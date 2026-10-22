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

from ._common import MODES, cfg_mode, compute_dtype, is_half, matmul_mode, mm, prep_weight

# (C, HID, mode) -> (forward: block, hidden chunk, warps, stages; backward kernel A: block, hidden chunk, warps, stages), swept on an H20.
# Only the combinations that beat torch (compiled, cuBLAS matmuls) are listed:
#   (128, 384) bf16   the pair transition: forward 1.23x, backward 1.1x of torch, and no [M, 768] activations kept
#   (128, 384) TF32   the forward only (1.14x); the recomputing backward is slower than torch's (4.4 vs 3.3 ms), so with gradients the
#                     torch path runs
# not fused: the MSA transition (256, 768): the forward is as fast as torch (0.56 vs 0.55 ms) and the recomputing backward is slower;
# 3xTF32 (float32 matmuls, "highest": the kernels support it with the TF32 configurations) is not fused: 41 ms against torch's 3.9 at the pair transition;
# small M (the single and atom transitions: 384 / 3072 rows, 30 microseconds in torch, launch bound)
_CONFIGS = {
    (128, 384, MODES["bf16"]): ((64, 64, 4, 2), (64, 32, 4, 3)),
    (128, 384, MODES["tf32"]): ((128, 32, 8, 2), (32, 32, 4, 2)),
}
MIN_ROWS = 16384            # below this the grid does not fill the GPU and the launch-bound torch path is as fast


def can_fuse(x: torch.Tensor, hidden: int) -> bool:
    """the kernels apply: a CUDA float32 / bfloat16 / float16 x with a supported (channels, hidden) for the current matmul precision and enough
    rows (see _CONFIGS)"""
    c, mode = x.shape[-1], matmul_mode()
    if mode == MODES["tf32x3"] or (mode == MODES["tf32"] and torch.is_grad_enabled()):
        return False
    return x.is_cuda and x.dtype in (torch.float32, torch.bfloat16, torch.float16) and (c, hidden, cfg_mode(mode)) in _CONFIGS and x.numel() // c >= MIN_ROWS


@triton.jit
def _fwd_kernel(X, NW, W13, W2, Y, M, eps,
                C: tl.constexpr, HID: tl.constexpr, HC: tl.constexpr, MODE: tl.constexpr, BLOCK_P: tl.constexpr):
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
    tl.store(Y + pos[:, None] * C + oc[None, :], (acc + x).to(Y.dtype.element_ty), mask=pm[:, None])


@triton.jit
def _bwd_kernel(X, DY, NW, W13, W2, DX, DH, S, NO, DYC, DNW, M, eps,
                C: tl.constexpr, HID: tl.constexpr, HC: tl.constexpr, MODE: tl.constexpr, STORE_DY: tl.constexpr, BLOCK_P: tl.constexpr):
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
    dx = r[:, None] * (gw - nhat * (tl.sum(gw * nhat, 1) / C)[:, None]) + dy
    tl.store(DX + pos[:, None] * C + oc[None, :], dx.to(DX.dtype.element_ty), mask=pm[:, None])
    tl.store(DNW + tile * C + oc, tl.sum(tl.where(pm[:, None], dn * nhat, 0.0), 0))


@torch.library.custom_op("physdock::rmsnorm_swiglu_ffn_fwd", mutates_args=())
def _ffn_fwd(x: torch.Tensor, norm_w: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor, eps: float, mode: int) -> torch.Tensor:
    C, hid = x.shape[-1], w2.shape[1]
    bp, hc, warps, stages = _CONFIGS[(C, hid, cfg_mode(mode))][0]
    x = x.contiguous()
    M = x.numel() // C
    y = torch.empty_like(x)
    _fwd_kernel[(triton.cdiv(M, bp),)](x, norm_w.contiguous(), prep_weight(w13, mode), prep_weight(w2, mode), y, M, eps, C=C, HID=hid, HC=hc,
                                       MODE=mode, BLOCK_P=bp, num_warps=warps, num_stages=stages)
    return y


@_ffn_fwd.register_fake
def _(x, norm_w, w13, w2, eps, mode):
    return x.new_empty(x.shape)


@torch.library.custom_op("physdock::rmsnorm_swiglu_ffn_bwd", mutates_args=())
def _ffn_bwd(dy: torch.Tensor, x: torch.Tensor, norm_w: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor, eps: float,
             mode: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    C, hid = x.shape[-1], w2.shape[1]
    bp, hc, warps, stages = _CONFIGS[(C, hid, cfg_mode(mode))][1]
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
                           C=C, HID=hid, HC=hc, MODE=mode, STORE_DY=store_dy, BLOCK_P=bp, num_warps=warps, num_stages=stages)
    dw13 = (dh.t() @ n).float()                 # [2 HID, C]: contraction over the M rows (cuBLAS)
    dw2 = (dyc.t() @ s).float()
    return dx, dnw.sum(0), dw13, dw2


@_ffn_bwd.register_fake
def _(dy, x, norm_w, w13, w2, eps, mode):
    return x.new_empty(x.shape), x.new_empty(norm_w.shape, dtype=torch.float32), x.new_empty(w13.shape, dtype=torch.float32), \
        x.new_empty(w2.shape, dtype=torch.float32)


def _setup_context(ctx, inputs, output):
    x, norm_w, w13, w2, eps, mode = inputs
    ctx.save_for_backward(x, norm_w, w13, w2)
    ctx.conf = (eps, mode)


def _backward(ctx, dy):
    x, norm_w, w13, w2 = ctx.saved_tensors
    dx, dnw, dw13, dw2 = _ffn_bwd(dy, x, norm_w, w13, w2, *ctx.conf)
    return dx, dnw.to(norm_w.dtype), dw13.to(w13.dtype), dw2.to(w2.dtype), None, None


_ffn_fwd.register_autograd(_backward, setup_context=_setup_context)


def rmsnorm_swiglu_ffn(x: torch.Tensor, norm_weight: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor, eps: float) -> torch.Tensor:
    """x + w2(silu(w1 n) * (w3 n)) with n = RMSNorm(x); differentiable with respect to
    x, norm_weight, w13 and w2. The matmul precision follows torch's (see kernels/_common.py).

    Args:
        x: [..., C] (the residual stream, float32 in the model)
        norm_weight: [C], w13: [2 * HID, C] (rows w1 | w3), w2: [C, HID]
        eps: RMSNorm epsilon

    Returns:
        [..., C] in the dtype of x
    """
    return _ffn_fwd(x, norm_weight, w13, w2, eps, matmul_mode())


# ---------------------------------------------------------------------------------------------------------------------------------
# Second implementation: the matmuls are cuBLAS, the norm / activation / norm-backward passes are Triton kernels. For the shapes where the
# fused kernels above do not beat torch (the MSA transition: 256 channels, hidden 768, 49152 rows): torch's time there is half cuBLAS and half
# elementwise passes (the RMSNorm in float32 and its backward, silu, the product, the concatenation of the two hidden gradients, casts), and
# these are the passes that are replaced here by four kernels that touch each tensor once. The hidden pre-activations [M, 2 HID] are kept for
# the backward as in torch (in the matmul dtype), so the memory is torch's; the matmuls are torch's (cuBLAS) in every mode.
# ---------------------------------------------------------------------------------------------------------------------------------
@triton.jit
def _norm_fwd_kernel(X, NW, N, M, eps, C: tl.constexpr, BLOCK_M: tl.constexpr):
    pos = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    pm = pos < M
    oc = tl.arange(0, C)
    x = tl.load(X + pos[:, None] * C + oc[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
    n = x * tl.math.rsqrt(tl.sum(x * x, 1) / C + eps)[:, None] * tl.load(NW + oc).to(tl.float32)[None, :]
    tl.store(N + pos[:, None] * C + oc[None, :], n.to(N.dtype.element_ty), mask=pm[:, None])


@triton.jit
def _act_fwd_kernel(H, S, M, HID: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr):
    pos = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    oh = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mk = (pos < M)[:, None] & (oh < HID)[None, :]
    h1 = tl.load(H + pos[:, None] * (2 * HID) + oh[None, :], mask=mk, other=0.0).to(tl.float32)
    h3 = tl.load(H + pos[:, None] * (2 * HID) + HID + oh[None, :], mask=mk, other=0.0).to(tl.float32)
    tl.store(S + pos[:, None] * HID + oh[None, :], (h1 * tl.sigmoid(h1) * h3).to(S.dtype.element_ty), mask=mk)


@triton.jit
def _act_bwd_kernel(H, DS, S, DH, M, HID: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr):
    pos = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    oh = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mk = (pos < M)[:, None] & (oh < HID)[None, :]
    h1 = tl.load(H + pos[:, None] * (2 * HID) + oh[None, :], mask=mk, other=0.0).to(tl.float32)
    h3 = tl.load(H + pos[:, None] * (2 * HID) + HID + oh[None, :], mask=mk, other=0.0).to(tl.float32)
    ds = tl.load(DS + pos[:, None] * HID + oh[None, :], mask=mk, other=0.0).to(tl.float32)
    sg = tl.sigmoid(h1)
    a = h1 * sg
    cdt = DH.dtype.element_ty
    tl.store(DH + pos[:, None] * (2 * HID) + oh[None, :], (ds * h3 * sg * (1.0 + h1 * (1.0 - sg))).to(cdt), mask=mk)
    tl.store(DH + pos[:, None] * (2 * HID) + HID + oh[None, :], (ds * a).to(cdt), mask=mk)
    tl.store(S + pos[:, None] * HID + oh[None, :], (a * h3).to(S.dtype.element_ty), mask=mk)


@triton.jit
def _norm_bwd_kernel(X, DN, DY, NW, DX, DNW, M, eps, C: tl.constexpr, BLOCK_M: tl.constexpr):
    tile = tl.program_id(0)
    pos = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    pm = pos < M
    oc = tl.arange(0, C)
    off = pos[:, None] * C + oc[None, :]
    x = tl.load(X + off, mask=pm[:, None], other=0.0).to(tl.float32)
    dn = tl.load(DN + off, mask=pm[:, None], other=0.0).to(tl.float32)
    nw = tl.load(NW + oc).to(tl.float32)
    r = tl.math.rsqrt(tl.sum(x * x, 1) / C + eps)
    nhat = x * r[:, None]
    gw = dn * nw[None, :]
    dx = r[:, None] * (gw - nhat * (tl.sum(gw * nhat, 1) / C)[:, None]) + tl.load(DY + off, mask=pm[:, None], other=0.0).to(tl.float32)
    tl.store(DX + off, dx.to(DX.dtype.element_ty), mask=pm[:, None])
    tl.store(DNW + tile * C + oc, tl.sum(tl.where(pm[:, None], dn * nhat, 0.0), 0))


SPLIT_BLOCK_M = 32          # rows per program of the norm kernels (float32 rows of C channels)


def can_split(x: torch.Tensor, hidden: int) -> bool:
    """the cuBLAS + Triton path applies: bfloat16 / float16 matmuls (autocast) of a CUDA float32 x with a power-of-two channel count and a hidden
    size divisible by 128, and enough rows (below MIN_ROWS the launch-bound torch path is as fast)"""
    c = x.shape[-1]
    return (is_half(matmul_mode()) and not torch.compiler.is_compiling() and x.is_cuda and x.dtype == torch.float32 and 16 <= c <= 512
            and c & (c - 1) == 0 and hidden % 128 == 0 and x.numel() // c >= MIN_ROWS)


@torch.library.custom_op("physdock::rmsnorm_swiglu_ffn_split_fwd", mutates_args=())
def _split_fwd(x: torch.Tensor, norm_w: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor, eps: float, mode: int
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> y [..., C] (dtype of x), n [M, C] and h [M, 2 HID] (the matmul dtype of the mode, kept for the backward)"""
    C, hid = x.shape[-1], w2.shape[1]
    cdt = compute_dtype(mode)
    x = x.contiguous()
    M = x.numel() // C
    n = torch.empty(M, C, device=x.device, dtype=cdt)
    _norm_fwd_kernel[(triton.cdiv(M, SPLIT_BLOCK_M),)](x, norm_w.contiguous(), n, M, eps, C=C, BLOCK_M=SPLIT_BLOCK_M, num_warps=4)
    h = torch.mm(n, w13.to(cdt).t())
    s = torch.empty(M, hid, device=x.device, dtype=cdt)
    _act_fwd_kernel[(triton.cdiv(M, 16), triton.cdiv(hid, 128))](h, s, M, HID=hid, BLOCK_M=16, BLOCK_H=128, num_warps=4)
    o = torch.mm(s, w2.to(cdt).t())
    return (x.view(M, C) + o).view(x.shape), n, h


@_split_fwd.register_fake
def _(x, norm_w, w13, w2, eps, mode):
    M = x.numel() // x.shape[-1]
    return x.new_empty(x.shape), x.new_empty(M, x.shape[-1], dtype=compute_dtype(mode)), x.new_empty(M, w13.shape[0], dtype=compute_dtype(mode))


@torch.library.custom_op("physdock::rmsnorm_swiglu_ffn_split_bwd", mutates_args=())
def _split_bwd(dy: torch.Tensor, x: torch.Tensor, n: torch.Tensor, h: torch.Tensor, norm_w: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor,
               eps: float, mode: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    C, hid = x.shape[-1], w2.shape[1]
    cdt = compute_dtype(mode)
    x = x.contiguous()
    M = x.numel() // C
    dy = dy.reshape(M, C).contiguous()
    dyb = dy.to(cdt)
    ds = torch.mm(dyb, w2.to(cdt))                                              # [M, HID]
    dh = torch.empty(M, 2 * hid, device=x.device, dtype=cdt)
    s = torch.empty(M, hid, device=x.device, dtype=cdt)
    _act_bwd_kernel[(triton.cdiv(M, 16), triton.cdiv(hid, 128))](h, ds, s, dh, M, HID=hid, BLOCK_M=16, BLOCK_H=128, num_warps=4)
    dn = torch.mm(dh, w13.to(cdt))                                              # [M, C]
    dw13 = torch.mm(dh.t(), n).float()
    dw2 = torch.mm(dyb.t(), s).float()
    ntiles = triton.cdiv(M, SPLIT_BLOCK_M)
    dx = torch.empty_like(x)
    dnw = torch.empty(ntiles, C, device=x.device, dtype=torch.float32)
    _norm_bwd_kernel[(ntiles,)](x, dn, dy, norm_w.contiguous(), dx, dnw, M, eps, C=C, BLOCK_M=SPLIT_BLOCK_M, num_warps=4)
    return dx, dnw.sum(0), dw13, dw2


@_split_bwd.register_fake
def _(dy, x, n, h, norm_w, w13, w2, eps, mode):
    return x.new_empty(x.shape), x.new_empty(norm_w.shape, dtype=torch.float32), x.new_empty(w13.shape, dtype=torch.float32), \
        x.new_empty(w2.shape, dtype=torch.float32)


def _split_setup_context(ctx, inputs, output):
    x, norm_w, w13, w2, eps, mode = inputs
    _, n, h = output
    ctx.save_for_backward(x, n, h, norm_w, w13, w2)
    ctx.eps, ctx.mode = eps, mode


def _split_backward(ctx, dy, _dn, _dh):
    x, n, h, norm_w, w13, w2 = ctx.saved_tensors
    dx, dnw, dw13, dw2 = _split_bwd(dy, x, n, h, norm_w, w13, w2, ctx.eps, ctx.mode)
    return dx, dnw.to(norm_w.dtype), dw13.to(w13.dtype), dw2.to(w2.dtype), None, None


_split_fwd.register_autograd(_split_backward, setup_context=_split_setup_context)


def rmsnorm_swiglu_ffn_split(x: torch.Tensor, norm_weight: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor, eps: float) -> torch.Tensor:
    """The same function as rmsnorm_swiglu_ffn (every mode), with cuBLAS matmuls and Triton elementwise passes; see above."""
    return _split_fwd(x, norm_weight, w13, w2, eps, matmul_mode())[0]
