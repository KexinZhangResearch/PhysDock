"""Triton kernels of LinearGateResidual: the end of an attention, forward and backward.

    y = x + (o @ W^T + b) * g

x [B, N, C] the residual stream (float32), o [B, H, N, D] the attention output in the head layout of the attention kernels (C = H * D;
read with its strides, never copied to the merged [B, N, C] layout), g [B, N, C] the output gate (any strides, e.g. a column slice of
the merged q, k, v, g projection; a stride of 0 along N for the AdaLN gate of the DiT), W [C, C], b [C].

Why a kernel. torch merges the heads of o with a copy, runs the matmul, the gate product and the residual add as three more passes
over [M, C] tensors, and its backward spends most of its time in the same elementwise passes, a bias reduction and the copy of the merged
o for the weight gradient. Here the forward is one pass (x, o, g in, y out) and the backward one kernel plus a batched cuBLAS matmul: per tile of rows t = o W^T + b is recomputed, dg = dy * t, dt = dy * g and
do = dt W are formed (do written in the head layout of o), dt is written in the matmul dtype and the tile's partial column sum of dt
(the bias gradient) is stored; the weight gradient dW = dt^T o (contraction over all rows, per head) is the cuBLAS matmul of dt with o in
its head layout. dx = dy needs no kernel.

Precision: see kernels/_common.py (bf16 under autocast, TF32 or float32 otherwise). The linear output is not rounded to bfloat16 before
the gate product and the residual add (torch rounds twice), so the result is at least as accurate as torch's in every mode.

The launch configurations are fixed (no autotuning pass), swept on an H20 (sm_90) at the shapes of the model.
"""
from typing import Tuple

import torch
import triton
import triton.language as tl

from ._common import MODES, compute_dtype, matmul_mode, mm, prep_weight

HEAD_DIM = 32


def _config(c: int):
    """(forward: rows, output columns, warps, stages; backward kernel: the same) of the width C, swept on an H20"""
    if c <= 128:
        return (128, 64, 8, 2), (64, 64, 4, 2)
    if c <= 256:
        return (32, 32, 4, 2), (32, 32, 4, 2)
    return (16, 32, 4, 2), (16, 16, 4, 2)


MIN_ROWS = 16384        # below this (the single stream: 384 rows) the launch-bound torch path is as fast


def can_fuse(x: torch.Tensor, o: torch.Tensor) -> bool:
    """the kernels apply: bfloat16 matmuls (autocast; in TF32 cuBLAS wins the backward) of a CUDA float32 residual x [..., N, C] with
    C = 32 .. 512 a power of two, the attention output o [..., H, N, 32] with a unit stride along the head dimension, and enough rows"""
    c = x.shape[-1]
    if torch.compiler.is_compiling():       # inductor fuses the norm / gate / residual around the matmul itself: 6% faster than this kernel
        return False
    return (matmul_mode() == MODES["bf16"] and x.numel() // c >= MIN_ROWS and x.is_cuda and x.dtype == torch.float32 and x.dim() in (2, 3) and o.dim() == x.dim() + 1 and 32 <= c <= 512 and c & (c - 1) == 0
            and o.shape[-1] == HEAD_DIM and o.shape[-3] * HEAD_DIM == c and o.stride(-1) == 1 and x.stride(-1) == 1)


@triton.jit
def _load_o(O, b, n, pm, sob, soh, son, C: tl.constexpr):
    """o tile [BLOCK_M, C]: column c is (head c // 32, channel c % 32) of the head layout [B, H, N, 32]"""
    oc = tl.arange(0, C)
    return tl.load(O + b[:, None] * sob + (oc // 32)[None, :] * soh + n[:, None] * son + (oc % 32)[None, :], mask=pm[:, None], other=0.0)


@triton.jit
def _fwd_kernel(X, O, G, W, B, Y, M, N, sxb, sxn, sob, soh, son, sgb, sgn, syb, syn,
                C: tl.constexpr, BN: tl.constexpr, MODE: tl.constexpr, BLOCK_M: tl.constexpr):
    pos = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    pm = pos < M
    b = pos // N
    n = pos - b * N
    on = tl.arange(0, BN)
    oc = tl.arange(0, C)
    o = _load_o(O, b, n, pm, sob, soh, son, C).to(tl.float32)
    for n0 in tl.range(0, C, BN):
        cols = n0 + on
        w = tl.load(W + cols[None, :] * C + oc[:, None])                                   # [C, BN]  (W^T tile)
        t = mm(o, w, MODE) + tl.load(B + cols).to(tl.float32)[None, :]
        g = tl.load(G + b[:, None] * sgb + n[:, None] * sgn + cols[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
        x = tl.load(X + b[:, None] * sxb + n[:, None] * sxn + cols[None, :], mask=pm[:, None], other=0.0)
        tl.store(Y + b[:, None] * syb + n[:, None] * syn + cols[None, :], x + t * g, mask=pm[:, None])


@triton.jit
def _bwd_kernel(DY, O, G, W, B, DT, DG, DBP, DO, M, N, sdb, sdn, sob, soh, son, sgb, sgn, sdob, sdoh, sdon,
                C: tl.constexpr, BN: tl.constexpr, MODE: tl.constexpr, BLOCK_M: tl.constexpr):
    """dt = dy * g (matmul dtype) and its column sums per tile, dg = dy * (o W^T + b), do = dt W in the head layout of o"""
    tile = tl.program_id(0)
    pos = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    pm = pos < M
    b = pos // N
    n = pos - b * N
    on = tl.arange(0, BN)
    oc = tl.arange(0, C)
    o = _load_o(O, b, n, pm, sob, soh, son, C).to(tl.float32)
    do = tl.zeros([BLOCK_M, C], tl.float32)
    for n0 in tl.range(0, C, BN):
        cols = n0 + on
        w = tl.load(W + cols[None, :] * C + oc[:, None])                                   # [C, BN]
        t = mm(o, w, MODE) + tl.load(B + cols).to(tl.float32)[None, :]
        dy = tl.load(DY + b[:, None] * sdb + n[:, None] * sdn + cols[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
        g = tl.load(G + b[:, None] * sgb + n[:, None] * sgn + cols[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
        dt = dy * g
        tl.store(DT + pos[:, None] * C + cols[None, :], dt.to(DT.dtype.element_ty), mask=pm[:, None])
        tl.store(DG + pos[:, None] * C + cols[None, :], (dy * t).to(DG.dtype.element_ty), mask=pm[:, None])
        tl.store(DBP + tile * C + cols, tl.sum(dt, 0))
        wn = tl.load(W + cols[:, None] * C + oc[None, :])                                  # [BN, C]
        do += mm(dt, wn, MODE)
    tl.store(DO + b[:, None] * sdob + (oc // 32)[None, :] * sdoh + n[:, None] * sdon + (oc % 32)[None, :], do.to(DO.dtype.element_ty),
             mask=pm[:, None])


@torch.library.custom_op("physdock::linear_gate_residual_fwd", mutates_args=())
def _fwd(x: torch.Tensor, o: torch.Tensor, g: torch.Tensor, w: torch.Tensor, b: torch.Tensor, mode: int) -> torch.Tensor:
    B, N, C = x.shape
    bp, bn, warps, stages = _config(C)[0]
    y = torch.empty_like(x)
    M = B * N
    _fwd_kernel[(triton.cdiv(M, bp),)](x, o, g, prep_weight(w, mode), b.contiguous(), y, M, N, x.stride(0), x.stride(1), o.stride(0),
                                       o.stride(1), o.stride(2), g.stride(0), g.stride(1), y.stride(0), y.stride(1), C=C, BN=bn,
                                       MODE=mode, BLOCK_M=bp, num_warps=warps, num_stages=stages)
    return y


@_fwd.register_fake
def _(x, o, g, w, b, mode):
    return torch.empty_like(x)


@torch.library.custom_op("physdock::linear_gate_residual_bwd", mutates_args=())
def _bwd(dy: torch.Tensor, o: torch.Tensor, g: torch.Tensor, w: torch.Tensor, b: torch.Tensor,
         mode: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """dy [B, N, C] -> dg [B, N, C], do [B, H, N, 32], dW [C, C], db [C] (the last two float32)"""
    B, N, C = dy.shape
    dy = dy if dy.stride(-1) == 1 else dy.contiguous()
    bp, bn, warps, stages = _config(C)[1]
    M = B * N
    cdt = compute_dtype(mode)
    dg = torch.empty(B, N, C, device=dy.device, dtype=g.dtype)
    dt = torch.empty(B, N, C, device=dy.device, dtype=cdt)
    do = torch.empty(o.shape, device=o.device, dtype=o.dtype if mode == MODES["bf16"] else torch.float32)
    ntiles = triton.cdiv(M, bp)
    dbp = torch.empty(ntiles, C, device=dy.device, dtype=torch.float32)
    _bwd_kernel[(ntiles,)](dy, o, g, prep_weight(w, mode), b.contiguous(), dt, dg, dbp, do, M, N, dy.stride(0), dy.stride(1), o.stride(0),
                           o.stride(1), o.stride(2), g.stride(0), g.stride(1), do.stride(0), do.stride(1), do.stride(2), C=C, BN=bn,
                           MODE=mode, BLOCK_M=bp, num_warps=warps, num_stages=stages)
    dw = torch.einsum("bnc,bhnd->hcd", dt, o.to(cdt)).permute(1, 0, 2).reshape(C, C).float()
    return dg, do, dw, dbp.sum(0)


@_bwd.register_fake
def _(dy, o, g, w, b, mode):
    return (dy.new_empty(dy.shape, dtype=g.dtype), o.new_empty(o.shape, dtype=o.dtype if mode == MODES["bf16"] else torch.float32),
            dy.new_empty(w.shape, dtype=torch.float32), dy.new_empty(b.shape, dtype=torch.float32))


def _setup_context(ctx, inputs, output):
    x, o, g, w, b, mode = inputs
    ctx.save_for_backward(o, g, w, b)
    ctx.mode, ctx.o_dtype = mode, o.dtype


def _backward(ctx, dy):
    o, g, w, b = ctx.saved_tensors
    dg, do, dw, db = _bwd(dy, o, g, w, b, ctx.mode)
    return dy, do.to(ctx.o_dtype), dg, dw.to(w.dtype), db.to(b.dtype), None


_fwd.register_autograd(_backward, setup_context=_setup_context)


def linear_gate_residual(x: torch.Tensor, o: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """x + (merge_heads(o) @ weight^T + bias) * gate (differentiable w.r.t. all tensors)

    Args:
        x: [N, C] or [B, N, C] float32
        o: [H, N, 32] or [B, H, N, 32] attention output, C = H * 32
        gate: broadcastable to x
        weight: [C, C], bias: [C]
    """
    squeeze = x.dim() == 2
    if squeeze:
        x, o = x[None], o[None]
    y = _fwd(x, o, gate.expand(x.shape), weight, bias, matmul_mode())
    return y[0] if squeeze else y
