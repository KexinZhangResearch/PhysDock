"""Triton kernels of RMSNormLinear: an RMSNorm followed by one linear layer (with a bias on the last columns only), forward and backward.

    y = n @ W^T + [0 | b],   n = RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * norm_weight

x [..., K], W [NOUT, K], b [BIAS] added to the last BIAS columns of y. This is the q, k, v, g projection of every gated attention
(K = 128 / 256 / 512, NOUT = 4 K, the bias of g alone) and the q, k projection of the outer product mean (NOUT = 64, bias on all columns).

Why a kernel. torch runs the norm as several passes over x (square, mean, rsqrt, two products, a cast to the matmul dtype), a matmul,
and keeps the normalised and rounded copy of x for the backward, whose norm part is again several float32 passes over the
[M, K] tensors. Here
  forward   x is read once (float32), normalised in registers and multiplied by the weight tiles; y is written once;
  backward  only x is saved. One kernel recomputes n, forms dn = dy W, finishes the norm backward (dx) and writes n for the
            weight gradient dW = dy^T n (contraction over all M rows: a cuBLAS matmul), the bias gradient is a column sum of dy.
No [M, K] float32 intermediate is stored.

Precision: see kernels/_common.py (bf16 under autocast, TF32 or float32 otherwise; the norm in float32, the matmul inputs rounded as
cuBLAS does, float32 accumulation, the bias added before the single rounding of the output).

The launch configurations are fixed (no autotuning pass), swept on an H20 (sm_90) at the shapes of the model.
"""
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from ._common import MODES, compute_dtype, matmul_mode, mm, prep_weight


# (K, NOUT, mode) -> (forward: rows per program, output columns per step, warps, stages; backward kernel: the same), swept on an H20
# at the shapes of the model (kmod/sweep_rnl.py); other shapes use the nearest width below
BF, TF = MODES["bf16"], MODES["tf32"]
_CONFIGS = {
    (128, 512, BF): ((64, 32, 4, 2), (64, 128, 4, 2)), (128, 512, TF): ((64, 32, 8, 2), (32, 64, 4, 3)),
    (256, 1024, BF): ((64, 32, 8, 3), (32, 64, 4, 2)), (256, 1024, TF): ((64, 32, 8, 3), (32, 32, 4, 3)),
    (512, 2048, BF): ((16, 128, 4, 2), (16, 64, 4, 2)), (512, 2048, TF): ((16, 64, 8, 2), (16, 32, 4, 3)),
    (256, 64, BF): ((16, 64, 4, 3), (32, 64, 4, 2)), (256, 64, TF): ((32, 32, 4, 2), (32, 32, 4, 2)),
    (128, 64, BF): ((64, 64, 8, 3), (64, 64, 4, 2)), (128, 64, TF): ((64, 64, 4, 2), (32, 64, 4, 2)),
}


def _config(k: int, nout: int, mode: int):
    if (k, nout, mode) in _CONFIGS:
        return _CONFIGS[(k, nout, mode)]
    return ((64, 32, 4, 2), (32, 32, 4, 2)) if k <= 256 else ((16, 32, 4, 2), (16, 32, 4, 2))


MIN_ROWS = 16384        # below this (the atom stream: 3072 rows) the launch-bound torch path is as fast; the 512-channel single stream wins at 384 rows


def can_fuse(x: torch.Tensor, nout: int) -> bool:
    """the kernels apply: bfloat16 matmuls (autocast) of a CUDA float32 x with a power-of-two channel count in [16, 512] and an output
    width divisible by 16, and enough work. In TF32 the Triton matmuls are slower than cuBLAS's (pair projection 0.48 vs 0.60 ms forward but
    the backward 1.7 vs 2.2 ms only; MSA and single projections lose), so the TF32 and float32 paths run torch."""
    k = x.shape[-1]
    if torch.compiler.is_compiling():       # inductor fuses the norm / gate / residual around the matmul itself: 6% faster than this kernel
        return False
    return (matmul_mode() == MODES["bf16"] and x.is_cuda and x.dtype == torch.float32 and 16 <= k <= 512 and k & (k - 1) == 0
            and nout % 16 == 0 and (k >= 512 or x.numel() // k >= MIN_ROWS))


@triton.jit
def _fwd_kernel(X, NW, W, B, Y, M, eps,
                K: tl.constexpr, NOUT: tl.constexpr, BIAS: tl.constexpr, BN: tl.constexpr, MODE: tl.constexpr, BLOCK_M: tl.constexpr):
    pos = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    pm = pos < M
    ok, on = tl.arange(0, K), tl.arange(0, BN)
    x = tl.load(X + pos[:, None] * K + ok[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
    n = x * tl.math.rsqrt(tl.sum(x * x, 1) / K + eps)[:, None] * tl.load(NW + ok).to(tl.float32)[None, :]
    for n0 in tl.range(0, NOUT, BN):
        cols = n0 + on
        w = tl.load(W + cols[None, :] * K + ok[:, None])                                   # [K, BN]
        acc = mm(n, w, MODE)
        if BIAS > 0:
            bc = cols - (NOUT - BIAS)
            acc = acc + tl.load(B + bc, mask=bc >= 0, other=0.0).to(tl.float32)[None, :]
        tl.store(Y + pos[:, None] * NOUT + cols[None, :], acc.to(Y.dtype.element_ty), mask=pm[:, None])


@triton.jit
def _bwd_kernel(X, DY, NW, W, DX, NO, DNW, M, eps,
                K: tl.constexpr, NOUT: tl.constexpr, BN: tl.constexpr, MODE: tl.constexpr, BLOCK_M: tl.constexpr):
    """one program per tile of rows; writes dX, the normalised input NO [M, K] (matmul dtype: the operand of the weight gradient) and
    the tile's partial gradient of the norm weight"""
    tile = tl.program_id(0)
    pos = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    pm = pos < M
    ok, on = tl.arange(0, K), tl.arange(0, BN)
    x = tl.load(X + pos[:, None] * K + ok[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
    nw = tl.load(NW + ok).to(tl.float32)
    r = tl.math.rsqrt(tl.sum(x * x, 1) / K + eps)
    nhat = x * r[:, None]
    dn = tl.zeros([BLOCK_M, K], tl.float32)
    for n0 in tl.range(0, NOUT, BN):
        cols = n0 + on
        dy = tl.load(DY + pos[:, None] * NOUT + cols[None, :], mask=pm[:, None], other=0.0).to(tl.float32)     # [BLOCK_M, BN]
        w = tl.load(W + cols[:, None] * K + ok[None, :])                                  # [BN, K]
        dn += mm(dy, w, MODE)
    tl.store(NO + pos[:, None] * K + ok[None, :], (nhat * nw[None, :]).to(NO.dtype.element_ty), mask=pm[:, None])
    gw = dn * nw[None, :]
    dx = r[:, None] * (gw - nhat * (tl.sum(gw * nhat, 1) / K)[:, None])
    tl.store(DX + pos[:, None] * K + ok[None, :], dx.to(DX.dtype.element_ty), mask=pm[:, None])
    tl.store(DNW + tile * K + ok, tl.sum(tl.where(pm[:, None], dn * nhat, 0.0), 0))


@torch.library.custom_op("physdock::rmsnorm_linear_fwd", mutates_args=())
def _fwd(x: torch.Tensor, norm_w: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor], eps: float, mode: int) -> torch.Tensor:
    K, nout = x.shape[-1], w.shape[0]
    bp, bn, warps, stages = _config(K, nout, mode)[0]
    x = x.contiguous()
    M = x.numel() // K
    y = torch.empty(*x.shape[:-1], nout, device=x.device, dtype=compute_dtype(mode) if mode == MODES["bf16"] else x.dtype)
    bias = b if b is not None else norm_w
    _fwd_kernel[(triton.cdiv(M, bp),)](x, norm_w.contiguous(), prep_weight(w, mode), bias.contiguous(), y, M, eps, K=K, NOUT=nout,
                                       BIAS=0 if b is None else b.numel(), BN=bn, MODE=mode, BLOCK_M=bp, num_warps=warps,
                                       num_stages=stages)
    return y


@_fwd.register_fake
def _(x, norm_w, w, b, eps, mode):
    return x.new_empty(*x.shape[:-1], w.shape[0], dtype=torch.bfloat16 if mode == MODES["bf16"] else x.dtype)


@torch.library.custom_op("physdock::rmsnorm_linear_bwd", mutates_args=())
def _bwd(dy: torch.Tensor, x: torch.Tensor, norm_w: torch.Tensor, w: torch.Tensor, eps: float, mode: int,
         bias: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """dy [..., NOUT] -> dx (dtype of x), d norm_w [K], d w [NOUT, K], d b [bias] (float32; empty if bias == 0)"""
    K, nout = x.shape[-1], w.shape[0]
    bp, bn, warps, stages = _config(K, nout, mode)[1]
    x = x.contiguous()
    M = x.numel() // K
    dy = dy.reshape(M, nout)
    dy = dy if dy.stride(-1) == 1 and dy.stride(0) == nout else dy.contiguous()
    dx = torch.empty_like(x)
    cdt = compute_dtype(mode)
    n = torch.empty(M, K, device=x.device, dtype=cdt)
    ntiles = triton.cdiv(M, bp)
    part = torch.empty(ntiles, K, device=x.device, dtype=torch.float32)
    _bwd_kernel[(ntiles,)](x, dy, norm_w.contiguous(), prep_weight(w, mode), dx, n, part, M, eps, K=K, NOUT=nout, BN=bn, MODE=mode,
                           BLOCK_M=bp, num_warps=warps, num_stages=stages)
    dw = torch.mm(dy.to(cdt).t(), n).float()
    db = dy[:, nout - bias:].float().sum(0) if bias else x.new_empty(0, dtype=torch.float32)
    return dx, part.sum(0), dw, db


@_bwd.register_fake
def _(dy, x, norm_w, w, eps, mode, bias):
    return (torch.empty_like(x), x.new_empty(x.shape[-1], dtype=torch.float32), x.new_empty(w.shape, dtype=torch.float32),
            x.new_empty(bias, dtype=torch.float32))


def _setup_context(ctx, inputs, output):
    x, norm_w, w, b, eps, mode = inputs
    ctx.save_for_backward(x, norm_w, w)
    ctx.eps, ctx.mode, ctx.bias = eps, mode, 0 if b is None else b.numel()


def _backward(ctx, dy):
    x, norm_w, w = ctx.saved_tensors
    dx, dnw, dw, db = _bwd(dy, x, norm_w, w, ctx.eps, ctx.mode, ctx.bias)
    return dx, dnw.to(norm_w.dtype), dw.to(w.dtype), (db if ctx.bias else None), None, None


_fwd.register_autograd(_backward, setup_context=_setup_context)


def rmsnorm_linear(x: torch.Tensor, norm_weight: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
                   eps: float = 1e-6) -> torch.Tensor:
    """linear(RMSNorm(x)) with `bias` added to the last bias.numel() output columns (differentiable w.r.t. all tensors)

    Args:
        x: [..., K] float32
        norm_weight: [K], weight: [NOUT, K], bias: [B] or None
        eps: RMSNorm epsilon

    Returns:
        [..., NOUT]: bfloat16 under autocast, else the dtype of x
    """
    return _fwd(x, norm_weight, weight, bias, eps, matmul_mode())
