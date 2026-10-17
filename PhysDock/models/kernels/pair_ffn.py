"""Triton kernels of the SwiGLU feed-forward on the atom-pair features, forward and backward, with the residual:

    out = x + w2(silu(w1 x) * (w3 x))        x [M, C] with C = 16 channels (M = N_atom^2 = 9.4 million pairs), hidden 128

(PhysDock's AtomEmbedder: ap + ffn(ap)). torch runs this as three matmuls with K = 16 or N = 16, a silu and a product and keeps
the hidden activations for the backward: three [M, 128] tensors = 7.2 GiB at 3072 atoms (bfloat16), and its backward spends
7 ms on a concatenation of the two hidden gradients. The kernels keep a tile of x in registers, compute the hidden units for
it and write only the [M, C] result; the backward recomputes the hidden units from x, so only x is saved.

Precision, per mode the same as the torch expression (never lower): "bf16" (autocast): matmul inputs rounded to bfloat16, float32
accumulation, the residual x kept in its own dtype (float32 in the model); "tf32" (float32 inputs, TF32 matmuls like torch's default): inputs rounded to the nearest TF32 value; "fp32":
float32 FMAs. The hidden units, the silu and the products are float32 (torch rounds them to the matmul dtype), the result is
x + ffn(x) in float32, rounded once to the dtype of x. Weight gradients are summed in float32 from per-program partial sums.
"""
from typing import Tuple

import torch
import triton
import triton.language as tl

MODES = {"fp32": 0, "tf32": 1, "bf16": 2}
_NUM_SMS = {}


def _nsm() -> int:
    dev = torch.cuda.current_device()
    if dev not in _NUM_SMS:
        _NUM_SMS[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    return _NUM_SMS[dev]


def can_fuse(x: torch.Tensor, hidden: int) -> bool:
    """CUDA float32 / bfloat16 x with 16 or 32 channels and a hidden size of 128"""
    return x.is_cuda and x.dtype in (torch.float32, torch.bfloat16) and x.shape[-1] in (16, 32) and hidden == 128


@triton.jit
def _round_tf32(x):
    return ((x.to(tl.int32, bitcast=True) + 0x1000) & -8192).to(tl.float32, bitcast=True)


@triton.jit
def _dot(a, b, MODE: tl.constexpr):
    if MODE == 0:
        return tl.dot(a, b, input_precision="ieee")
    elif MODE == 1:
        return tl.dot(_round_tf32(a), _round_tf32(b), input_precision="tf32")
    else:
        return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))


@triton.jit
def _fwd_kernel(X, W1, W3, W2, OUT, M, C: tl.constexpr, HID: tl.constexpr, MODE: tl.constexpr, BLOCK_P: tl.constexpr):
    pos = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    pm = pos < M
    oc, oh = tl.arange(0, C), tl.arange(0, HID)
    x = tl.load(X + pos[:, None] * C + oc[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
    w1 = tl.load(W1 + oh[None, :] * C + oc[:, None]).to(tl.float32)                  # [C, HID] = W1^T
    w3 = tl.load(W3 + oh[None, :] * C + oc[:, None]).to(tl.float32)
    w2 = tl.load(W2 + oc[None, :] * HID + oh[:, None]).to(tl.float32)                # [HID, C] = W2^T
    h1 = _dot(x, w1, MODE)
    h3 = _dot(x, w3, MODE)
    s = h1 * tl.sigmoid(h1) * h3
    y = _dot(s, w2, MODE) + x
    tl.store(OUT + pos[:, None] * C + oc[None, :], y.to(OUT.dtype.element_ty), mask=pm[:, None])


@triton.jit
def _bwd_kernel(X, DY, W1, W3, W2, DX, PART, M, ntiles, NPROG: tl.constexpr, C: tl.constexpr, HID: tl.constexpr, MODE: tl.constexpr,
                HC: tl.constexpr, BLOCK_P: tl.constexpr):
    """persistent programs; the hidden units are walked in two chunks of HC = HID / 2 (the [BLOCK_P, HID] float32 temporaries
    of the whole hidden width do not fit the registers); PART[pid] = [dW1 (HID x C) | dW3 (HID x C) | dW2 (C x HID)] partial sums"""
    pid = tl.program_id(0)
    oc, ohc = tl.arange(0, C), tl.arange(0, HC)
    dw1_0 = tl.zeros([HC, C], tl.float32)
    dw1_1 = tl.zeros([HC, C], tl.float32)
    dw3_0 = tl.zeros([HC, C], tl.float32)
    dw3_1 = tl.zeros([HC, C], tl.float32)
    dw2_0 = tl.zeros([C, HC], tl.float32)
    dw2_1 = tl.zeros([C, HC], tl.float32)
    for t in range(pid, ntiles, NPROG):
        pos = t * BLOCK_P + tl.arange(0, BLOCK_P)
        pm = pos < M
        x = tl.load(X + pos[:, None] * C + oc[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
        dy = tl.load(DY + pos[:, None] * C + oc[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
        dx = dy
        for c in tl.static_range(2):
            oh = c * HC + ohc
            w1 = tl.load(W1 + oh[None, :] * C + oc[:, None]).to(tl.float32)             # [C, HC]
            w3 = tl.load(W3 + oh[None, :] * C + oc[:, None]).to(tl.float32)
            w2 = tl.load(W2 + oc[:, None] * HID + oh[None, :]).to(tl.float32)           # [C, HC]
            h1 = _dot(x, w1, MODE)
            h3 = _dot(x, w3, MODE)
            sg = tl.sigmoid(h1)
            a = h1 * sg
            ds = _dot(dy, w2, MODE)                                                     # [BLOCK_P, HC]
            dh3 = ds * a
            dh1 = ds * h3 * sg * (1.0 + h1 * (1.0 - sg))
            s = a * h3
            w1n = tl.load(W1 + oh[:, None] * C + oc[None, :]).to(tl.float32)            # [HC, C]
            w3n = tl.load(W3 + oh[:, None] * C + oc[None, :]).to(tl.float32)
            dx += _dot(dh1, w1n, MODE) + _dot(dh3, w3n, MODE)
            g1 = _dot(tl.trans(dh1), x, MODE)                                           # [HC, C]
            g3 = _dot(tl.trans(dh3), x, MODE)
            g2 = _dot(tl.trans(dy), s, MODE)                                            # [C, HC]
            if c == 0:
                dw1_0 += g1
                dw3_0 += g3
                dw2_0 += g2
            else:
                dw1_1 += g1
                dw3_1 += g3
                dw2_1 += g2
        tl.store(DX + pos[:, None] * C + oc[None, :], dx.to(DX.dtype.element_ty), mask=pm[:, None])
    base = PART + pid * (3 * HID * C)
    tl.store(base + ohc[:, None] * C + oc[None, :], dw1_0)
    tl.store(base + (HC + ohc[:, None]) * C + oc[None, :], dw1_1)
    tl.store(base + HID * C + ohc[:, None] * C + oc[None, :], dw3_0)
    tl.store(base + HID * C + (HC + ohc[:, None]) * C + oc[None, :], dw3_1)
    tl.store(base + 2 * HID * C + oc[:, None] * HID + ohc[None, :], dw2_0)
    tl.store(base + 2 * HID * C + oc[:, None] * HID + HC + ohc[None, :], dw2_1)


def _split_weights(w13: torch.Tensor):
    h = w13.shape[0] // 2
    return w13[:h].contiguous(), w13[h:].contiguous()


@torch.library.custom_op("physdock::pair_ffn_fwd", mutates_args=())
def _pair_ffn_fwd(x: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor, mode: int) -> torch.Tensor:
    C, hid = x.shape[-1], w2.shape[1]
    x = x.contiguous()
    M = x.numel() // C
    w1, w3 = _split_weights(w13)
    out = torch.empty_like(x)
    bp = 64                                                     # fixed configuration (swept on an H20: 64 positions, 4 warps)
    _fwd_kernel[(triton.cdiv(M, bp),)](x, w1, w3, w2.contiguous(), out, M, C=C, HID=hid, MODE=mode, BLOCK_P=bp, num_warps=4, num_stages=2)
    return out


@_pair_ffn_fwd.register_fake
def _(x, w13, w2, mode):
    return torch.empty_like(x)


@torch.library.custom_op("physdock::pair_ffn_bwd", mutates_args=())
def _pair_ffn_bwd(dy: torch.Tensor, x: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor, mode: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    C, hid = x.shape[-1], w2.shape[1]
    x, dy = x.contiguous(), dy.contiguous()
    M = x.numel() // C
    w1, w3 = _split_weights(w13)
    dx = torch.empty_like(x)
    bp = 64
    ntiles = triton.cdiv(M, bp)
    nprog = min(2 * _nsm(), ntiles)
    part = torch.empty(nprog, 3 * hid * C, device=x.device, dtype=torch.float32)
    _bwd_kernel[(nprog,)](x, dy, w1, w3, w2.contiguous(), dx, part, M, ntiles, NPROG=nprog, C=C, HID=hid, MODE=mode, HC=hid // 2,
                          BLOCK_P=bp, num_warps=4, num_stages=2)
    tot = part.sum(0)
    dw13 = torch.cat([tot[:hid * C].view(hid, C), tot[hid * C:2 * hid * C].view(hid, C)])
    return dx, dw13, tot[2 * hid * C:].view(C, hid).clone()


@_pair_ffn_bwd.register_fake
def _(dy, x, w13, w2, mode):
    return torch.empty_like(x), x.new_empty(w13.shape, dtype=torch.float32), x.new_empty(w2.shape, dtype=torch.float32)


def _setup_context(ctx, inputs, output):
    x, w13, w2, mode = inputs
    ctx.save_for_backward(x, w13, w2)
    ctx.mode = mode


def _backward(ctx, dy):
    x, w13, w2 = ctx.saved_tensors
    dx, dw13, dw2 = _pair_ffn_bwd(dy, x, w13, w2, ctx.mode)
    return dx, dw13.to(w13.dtype), dw2.to(w2.dtype), None


_pair_ffn_fwd.register_autograd(_backward, setup_context=_setup_context)


def pair_ffn(x: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """x + w2(silu(w1 x) * (w3 x)) with w13 = [w1; w3] [2 * hidden, C], w2 [C, hidden]; differentiable w.r.t. x, w13 and w2.

    The residual is added to x itself, in the dtype of x (the float32 pair features under autocast: the residual stream is not
    rounded to bfloat16). The matmul precision follows torch's: bfloat16 inputs under autocast, else TF32 unless
    torch.set_float32_matmul_precision("highest")."""
    if torch.is_autocast_enabled():
        mode = "bf16"
    else:
        mode = "tf32" if torch.get_float32_matmul_precision() != "highest" else "fp32"
    return _pair_ffn_fwd(x, w13, w2, MODES[mode])
