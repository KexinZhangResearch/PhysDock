"""Triton kernels of RMSNormSwiGLUFFN: a residual SwiGLU feed-forward behind an RMSNorm, forward and backward.

    y = x + w2( silu(w1 n) * (w3 n) ),   n = RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * norm_weight

x [..., C], w13 = [w1; w3] [2 * HID, C] (the merged SwiGLUFFN.w13), w2 [C, HID]. The pair transition (C = 128, HID = 384, M = 147456
rows at 384 tokens), the MSA transition (256 / 768) and the other Transitions are this operator.

Why a kernel. torch runs it as an RMSNorm (several passes over x), a matmul to [M, 2 * HID], a silu * product, a matmul back, the
residual add, and keeps the [M, 2 * HID] output of the first matmul for the backward (361 MiB per pair transition at 384 tokens;
48 blocks: 17 GiB), and its backward spends 45% of its time outside the matmuls (a concatenation of the two hidden gradients, the
silu derivative, the norm backward, casts). Here
  forward   one kernel: a tile of rows is normalised in registers and the hidden units are produced chunk by chunk (HC at a time),
            multiplied by w2 and accumulated; the [M, 2 * HID] hidden activations never exist in memory.
  backward  the hidden units are recomputed from x (only x is saved): kernel A recomputes them per tile, forms the gradients of the
            hidden pre-activations dH = [dh1 | dh3] and of the output of the first stage s = silu(h1) * h3, and finishes the norm
            backward (dx); the three weight-gradient products dW13 = dH^T n, dW2 = dy^T s (contraction over all M rows) are cuBLAS
            matmuls on what kernel A wrote.
The matmuls are not recomputed in one place only: the recomputation costs ~1/3 more FLOPs than a backward that has the hidden
activations, which is why the backward gain (1.1x) is smaller than the forward gain (1.2x); the saved memory is the point.

Precision: see kernels/_common.py (bf16 under autocast, TF32 or float32 otherwise; norm, silu, products, residual in float32). The
hidden units are not rounded between the two stages beyond the matmul input rounding (torch rounds them to bfloat16 three times
under autocast), so the result is at least as accurate as torch's in every mode.

The configurations are fixed (no autotuning pass): swept on an H20 (sm_90) at the shapes of the model.
"""
from typing import Tuple

import torch
import triton
import triton.language as tl

from ._common import MODES, compute_dtype, matmul_mode, mm, prep_weight

# (C, HID, mode) -> (forward: block, hidden chunk, warps, stages; backward kernel A: block, hidden chunk, warps, stages), swept on an H20.
# Only the combinations that beat torch (compiled, cuBLAS matmuls) are listed:
#   (128, 384) bf16   the pair transition: forward 1.23x, backward 1.1x of torch, and no [M, 768] activations kept
#   (128, 384) TF32   the forward only (1.14x); the recomputing backward is slower than torch's (4.4 vs 3.3 ms), so with gradients the
#                     torch path runs
# not fused: the MSA transition (256, 768): the forward is as fast as torch (0.56 vs 0.55 ms) and the recomputing backward is slower;
# float32 matmuls ("highest"): 6 TFLOPS in Triton against cuBLAS's ~50; small M (the single and atom transitions: 384 / 3072 rows,
# 30 microseconds in torch, launch bound)
_CONFIGS = {
    (128, 384, MODES["bf16"]): ((64, 64, 4, 2), (64, 32, 4, 3)),
    (128, 384, MODES["tf32"]): ((128, 32, 8, 2), (32, 32, 4, 2)),
}
MIN_ROWS = 16384            # below this the grid does not fill the GPU and the launch-bound torch path is as fast


def can_fuse(x: torch.Tensor, hidden: int) -> bool:
    """the kernels apply: a CUDA float32 / bfloat16 x with a supported (channels, hidden) for the current matmul precision and enough
    rows (see _CONFIGS)"""
    c, mode = x.shape[-1], matmul_mode()
    if mode == MODES["tf32"] and torch.is_grad_enabled():
        return False
    return x.is_cuda and x.dtype in (torch.float32, torch.bfloat16) and (c, hidden, mode) in _CONFIGS and x.numel() // c >= MIN_ROWS


@triton.jit
def _fwd_kernel(X, NW, W13, W2, Y, M, eps,
                C: tl.constexpr, HID: tl.constexpr, HC: tl.constexpr, RES: tl.constexpr, MODE: tl.constexpr, BLOCK_P: tl.constexpr):
    pos = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    pm = pos < M
    oc, ohc = tl.arange(0, C), tl.arange(0, HC)
    x = tl.load(X + pos[:, None] * C + oc[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
    n = x * tl.math.rsqrt(tl.sum(x * x, 1) / C + eps)[:, None] * tl.load(NW + oc).to(tl.float32)[None, :]
    acc = tl.zeros([BLOCK_P, C], tl.float32)
    for h0 in tl.range(0, HID, HC):
        w1 = tl.load(W13 + (h0 + ohc)[None, :] * C + oc[:, None])                    # [C, HC]
        w3 = tl.load(W13 + (HID + h0 + ohc)[None, :] * C + oc[:, None])
        w2 = tl.load(W2 + oc[None, :] * HID + (h0 + ohc)[:, None])                    # [HC, C]
        h1 = mm(n, w1, MODE)
        h3 = mm(n, w3, MODE)
        acc = acc + mm(h1 * tl.sigmoid(h1) * h3, w2, MODE)
    if RES:
        acc = acc + x
    tl.store(Y + pos[:, None] * C + oc[None, :], acc.to(Y.dtype.element_ty), mask=pm[:, None])


@triton.jit
def _bwd_kernel(X, DY, NW, W13, W2, DX, DH, S, NO, DYC, DNW, M, eps,
                C: tl.constexpr, HID: tl.constexpr, HC: tl.constexpr, RES: tl.constexpr, MODE: tl.constexpr, STORE_DY: tl.constexpr,
                BLOCK_P: tl.constexpr):
    """one program per tile of rows; writes dX, the hidden gradients DH [M, 2 HID] = [dh1 | dh3], S [M, HID] = silu(h1) * h3, the
    normalised input NO [M, C] and dy in the matmul dtype DYC (STORE_DY, if dy is not in it already): the operands of the
    weight-gradient matmuls; and the tile's partial gradient of the norm weight"""
    tile = tl.program_id(0)
    pos = tile * BLOCK_P + tl.arange(0, BLOCK_P)
    pm = pos < M
    oc, ohc = tl.arange(0, C), tl.arange(0, HC)
    x = tl.load(X + pos[:, None] * C + oc[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
    dy = tl.load(DY + pos[:, None] * C + oc[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
    nw = tl.load(NW + oc).to(tl.float32)
    r = tl.math.rsqrt(tl.sum(x * x, 1) / C + eps)
    nhat = x * r[:, None]
    n = nhat * nw[None, :]
    dn = tl.zeros([BLOCK_P, C], tl.float32)
    cdt = DH.dtype.element_ty
    for h0 in tl.range(0, HID, HC):
        w1n = tl.load(W13 + (h0 + ohc)[:, None] * C + oc[None, :])                    # [HC, C]
        w3n = tl.load(W13 + (HID + h0 + ohc)[:, None] * C + oc[None, :])
        w2 = tl.load(W2 + oc[:, None] * HID + (h0 + ohc)[None, :])                    # [C, HC]
        h1 = mm(n, tl.trans(w1n), MODE)
        h3 = mm(n, tl.trans(w3n), MODE)
        sg = tl.sigmoid(h1)
        a = h1 * sg
        ds = mm(dy, w2, MODE)                                                         # d(silu(h1) * h3)  [BLOCK_P, HC]
        dh1 = ds * h3 * sg * (1.0 + h1 * (1.0 - sg))
        dh3 = ds * a
        tl.store(DH + pos[:, None] * (2 * HID) + (h0 + ohc)[None, :], dh1.to(cdt), mask=pm[:, None])
        tl.store(DH + pos[:, None] * (2 * HID) + (HID + h0 + ohc)[None, :], dh3.to(cdt), mask=pm[:, None])
        tl.store(S + pos[:, None] * HID + (h0 + ohc)[None, :], (a * h3).to(cdt), mask=pm[:, None])
        dn += mm(dh1, w1n, MODE) + mm(dh3, w3n, MODE)
    tl.store(NO + pos[:, None] * C + oc[None, :], n.to(NO.dtype.element_ty), mask=pm[:, None])
    if STORE_DY:
        tl.store(DYC + pos[:, None] * C + oc[None, :], dy.to(cdt), mask=pm[:, None])
    gw = dn * nw[None, :]
    dx = r[:, None] * (gw - nhat * (tl.sum(gw * nhat, 1) / C)[:, None])
    if RES:
        dx = dx + dy
    tl.store(DX + pos[:, None] * C + oc[None, :], dx.to(DX.dtype.element_ty), mask=pm[:, None])
    tl.store(DNW + tile * C + oc, tl.sum(tl.where(pm[:, None], dn * nhat, 0.0), 0))


@torch.library.custom_op("physdock::rmsnorm_swiglu_ffn_fwd", mutates_args=())
def _ffn_fwd(x: torch.Tensor, norm_w: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor, eps: float, residual: bool, mode: int) -> torch.Tensor:
    C, hid = x.shape[-1], w2.shape[1]
    bp, hc, warps, stages = _CONFIGS[(C, hid, mode)][0]
    x = x.contiguous()
    M = x.numel() // C
    y = torch.empty_like(x)
    _fwd_kernel[(triton.cdiv(M, bp),)](x, norm_w.contiguous(), prep_weight(w13, mode), prep_weight(w2, mode), y, M, eps, C=C, HID=hid, HC=hc,
                                       RES=residual, MODE=mode, BLOCK_P=bp, num_warps=warps, num_stages=stages)
    return y


@_ffn_fwd.register_fake
def _(x, norm_w, w13, w2, eps, residual, mode):
    return torch.empty_like(x)


@torch.library.custom_op("physdock::rmsnorm_swiglu_ffn_bwd", mutates_args=())
def _ffn_bwd(dy: torch.Tensor, x: torch.Tensor, norm_w: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor, eps: float, residual: bool,
             mode: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    C, hid = x.shape[-1], w2.shape[1]
    bp, hc, warps, stages = _CONFIGS[(C, hid, mode)][1]
    x, dy = x.contiguous(), dy.contiguous()
    M = x.numel() // C
    cdt = compute_dtype(mode)
    dx = torch.empty_like(x)
    dh = torch.empty(M, 2 * hid, device=x.device, dtype=cdt)
    s = torch.empty(M, hid, device=x.device, dtype=cdt)
    n = torch.empty(M, C, device=x.device, dtype=cdt)
    store_dy = dy.dtype != cdt
    dyc = torch.empty(M, C, device=x.device, dtype=cdt) if store_dy else dy.reshape(M, C)
    ntiles = triton.cdiv(M, bp)
    dnw = torch.empty(ntiles, C, device=x.device, dtype=torch.float32)
    _bwd_kernel[(ntiles,)](x, dy, norm_w.contiguous(), prep_weight(w13, mode), prep_weight(w2, mode), dx, dh, s, n, dyc, dnw, M, eps,
                           C=C, HID=hid, HC=hc, RES=residual, MODE=mode, STORE_DY=store_dy, BLOCK_P=bp, num_warps=warps, num_stages=stages)
    dw13 = (dh.t() @ n).float()                 # [2 HID, C]: contraction over the M rows (cuBLAS)
    dw2 = (dyc.t() @ s).float()
    return dx, dnw.sum(0), dw13, dw2


@_ffn_bwd.register_fake
def _(dy, x, norm_w, w13, w2, eps, residual, mode):
    return torch.empty_like(x), x.new_empty(norm_w.shape, dtype=torch.float32), x.new_empty(w13.shape, dtype=torch.float32), \
        x.new_empty(w2.shape, dtype=torch.float32)


def _setup_context(ctx, inputs, output):
    x, norm_w, w13, w2, eps, residual, mode = inputs
    ctx.save_for_backward(x, norm_w, w13, w2)
    ctx.conf = (eps, residual, mode)


def _backward(ctx, dy):
    x, norm_w, w13, w2 = ctx.saved_tensors
    dx, dnw, dw13, dw2 = _ffn_bwd(dy, x, norm_w, w13, w2, *ctx.conf)
    return dx, dnw.to(norm_w.dtype), dw13.to(w13.dtype), dw2.to(w2.dtype), None, None, None


_ffn_fwd.register_autograd(_backward, setup_context=_setup_context)


def rmsnorm_swiglu_ffn(x: torch.Tensor, norm_weight: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor, eps: float,
                       residual: bool = True) -> torch.Tensor:
    """x + w2(silu(w1 n) * (w3 n)) with n = RMSNorm(x) (without the leading x if residual is False); differentiable with respect to
    x, norm_weight, w13 and w2. The matmul precision follows torch's (see kernels/_common.py).

    Args:
        x: [..., C] (the residual stream, float32 in the model)
        norm_weight: [C], w13: [2 * HID, C] (rows w1 | w3), w2: [C, HID]
        eps: RMSNorm epsilon

    Returns:
        [..., C] in the dtype of x
    """
    return _ffn_fwd(x, norm_weight, w13, w2, eps, residual, matmul_mode())
