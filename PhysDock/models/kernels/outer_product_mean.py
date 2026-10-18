"""Triton kernels of OuterProductMean, forward and backward:

    y[i, j] = z[i, j] + RMSNorm( sum_{c, d} W[:, c, d] * sum_s q[s, i, c] * k[s, j, d]  + b ) * norm_weight

qk [S, N, 64] = [q | k] (the output of the RMSNormLinear that precedes: S MSA rows, N tokens, 32 channels each), z [N, N, CZ], W [CZ, 1024]
(the output Linear of the outer product mean, columns (c, d) = c * 32 + d), b [CZ].

Why a kernel. torch forms the outer product [N, N, 32, 32] (150M elements at 384 tokens: 300 MiB in bfloat16) with a batched matmul and
copies it to the layout of the output projection, runs the projection, the RMSNorm (several float32 passes) and the residual add, and the
backward does the same in the opposite direction with three more large copies (a quarter of its time) and a contraction over the
N^2 positions for the weight gradient. Here the outer product only ever exists as a [BI, BJ * 32] register tile:
  forward    program (i block of BI rows, one j): for each c, T_c = q[:, i, c]^T k[:, j, :] ([BI, 32], contraction over the MSA rows, tensor cores)
             multiplied by W_c ([32, CZ]) and accumulated over the 32 values of c; the epilogue adds the bias, applies the RMSNorm and the
             residual and writes y; the projection u (before the norm) is also kept, in the matmul dtype, for the backward;
  backward   1. du = the RMSNorm backward from dy and u (one elementwise pass; also the partial gradients of the norm weight and of the bias);
             2. dW_c = sum over the positions of du^T T_c (T_c recomputed per tile; programs over (c, a slice of the tiles));
             3. dq[:, i, c] = sum over j of k[:, j, :] dO_c^T with dO_c = du W_c^T ([BI, 32], formed per tile, never stored);
             4. dk[:, j, :] = sum over i, c of q[:, i, c]^T dO_c, accumulated in partial buffers over slices of the i blocks.
             Nothing of size N^2 * 1024 is ever written.

Precision: see kernels/_common.py. The outer product is rounded to the matmul dtype once before the projection (as torch's bfloat16 einsum
output is), the projection output is rounded to it once (as torch's linear is), the RMSNorm is float32 and the residual add float32.

The launch configurations are fixed (no autotuning pass), swept on an H20 (sm_90) at the shapes of the model.
"""
from typing import Tuple

import torch
import triton
import triton.language as tl

from ._common import MODES, compute_dtype, matmul_mode, mm, mm_act, prep_weight

HEAD = tl.constexpr(32)
_H = 32                # the same, for the Python side
# (forward: BI, warps, stages), (dW: BI, warps, stages, slices), (dq: BI, warps, stages), (dk: BI, warps, stages, slices), swept on an H20
_CONFIGS = {
    MODES["bf16"]: ((64, 4, 2), (128, 8, 3, 16), (64, 4, 2), (64, 4, 2, 4)),
    MODES["tf32"]: ((32, 4, 3), (64, 4, 3, 16), (32, 4, 3), (128, 8, 3, 1)),
}


def _config(mode: int):
    return _CONFIGS.get(mode, _CONFIGS[MODES["tf32"]])


def can_fuse(qk: torch.Tensor, z: torch.Tensor) -> bool:
    """the kernels apply: CUDA qk [S, N, 64] (float32 or bfloat16, unit stride in the last dimension) and a float32 z [N, N, CZ] with CZ a
    power of two in [32, 128]; in TF32 only without gradients (the forward is 1.1x faster than torch's, the backward 1.6x slower)"""
    cz = z.shape[-1]
    if matmul_mode() != MODES["bf16"] and torch.is_grad_enabled():
        return False
    return (qk.is_cuda and qk.dim() == 3 and qk.shape[-1] == 64 and qk.stride(-1) == 1 and qk.dtype in (torch.float32, torch.bfloat16)
            and z.dtype == torch.float32 and z.dim() == 3 and z.shape[0] == z.shape[1] == qk.shape[1] and 32 <= cz <= 128 and cz & (cz - 1) == 0)


@triton.jit
def _t_tile(QT, K, c, oi, imask, j, S, sqc, sqs, sks, skn, BS: tl.constexpr, NSC: tl.constexpr, MODE: tl.constexpr, BI: tl.constexpr):
    """T_c [BI, 32] = q[:, i, c]^T k[:, j, :] for the rows i of the block, contraction over the S rows"""
    od = tl.arange(0, HEAD)
    T = tl.zeros([BI, HEAD], tl.float32)
    for sc in range(NSC):
        os_ = sc * BS + tl.arange(0, BS)
        sm = os_ < S
        kt = tl.load(K + os_[:, None] * sks + j * skn + od[None, :], mask=sm[:, None], other=0.0)                  # [BS, 32]
        qt = tl.load(QT + c * sqc + os_[:, None] * sqs + oi[None, :], mask=sm[:, None] & imask[None, :], other=0.0)  # [BS, BI]
        T += mm_act(tl.trans(qt), kt, MODE)
    return T


@triton.jit
def _fwd_kernel(QT, K, WP, BO, NW, Z, Y, U, S, N, sqc, sqs, sks, skn, eps,
                CZ: tl.constexpr, BI: tl.constexpr, BS: tl.constexpr, NSC: tl.constexpr, MODE: tl.constexpr):
    pi = tl.program_id(0)
    j = tl.program_id(1)
    oi = pi * BI + tl.arange(0, BI)
    imask = oi < N
    ocz, od = tl.arange(0, CZ), tl.arange(0, HEAD)
    acc = tl.zeros([BI, CZ], tl.float32)
    for c in range(HEAD):
        T = _t_tile(QT, K, c, oi, imask, j, S, sqc, sqs, sks, skn, BS, NSC, MODE, BI)
        w = tl.load(WP + ocz[None, :] * (HEAD * HEAD) + c * HEAD + od[:, None])                       # [32, CZ]
        acc += mm(T, w, MODE)
    u = acc + tl.load(BO + ocz).to(tl.float32)[None, :]
    if MODE == 2:
        u = u.to(tl.bfloat16).to(tl.float32)
    off = (oi * N + j)[:, None] * CZ + ocz[None, :]
    tl.store(U + off, u.to(U.dtype.element_ty), mask=imask[:, None])
    n = u * tl.math.rsqrt(tl.sum(u * u, 1) / CZ + eps)[:, None] * tl.load(NW + ocz).to(tl.float32)[None, :]
    z = tl.load(Z + off, mask=imask[:, None], other=0.0)
    tl.store(Y + off, z + n, mask=imask[:, None])


@triton.jit
def _du_kernel(DY, U, NW, DU, PART, M, eps, CZ: tl.constexpr, BLOCK_M: tl.constexpr):
    tile = tl.program_id(0)
    pos = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    pm = pos < M
    ocz = tl.arange(0, CZ)
    off = pos[:, None] * CZ + ocz[None, :]
    dy = tl.load(DY + off, mask=pm[:, None], other=0.0).to(tl.float32)
    u = tl.load(U + off, mask=pm[:, None], other=0.0).to(tl.float32)
    nw = tl.load(NW + ocz).to(tl.float32)
    r = tl.math.rsqrt(tl.sum(u * u, 1) / CZ + eps)
    nh = u * r[:, None]
    gw = dy * nw[None, :]
    du = r[:, None] * (gw - nh * (tl.sum(gw * nh, 1) / CZ)[:, None])
    tl.store(DU + off, du.to(DU.dtype.element_ty), mask=pm[:, None])
    tl.store(PART + tile * 2 * CZ + ocz, tl.sum(tl.where(pm[:, None], dy * nh, 0.0), 0))
    tl.store(PART + tile * 2 * CZ + CZ + ocz, tl.sum(tl.where(pm[:, None], du, 0.0), 0))


@triton.jit
def _dw_kernel(QT, K, DU, DWP, S, N, sqc, sqs, sks, skn, nti, NSPLIT,
               CZ: tl.constexpr, BI: tl.constexpr, BS: tl.constexpr, NSC: tl.constexpr, MODE: tl.constexpr):
    """program (c, slice): dW_c [CZ, 32] over the (i block, j) tiles t = slice, slice + NSPLIT, ..."""
    c = tl.program_id(0)
    sp = tl.program_id(1)
    ocz, od = tl.arange(0, CZ), tl.arange(0, HEAD)
    acc = tl.zeros([CZ, HEAD], tl.float32)
    for t in range(sp, nti * N, NSPLIT):
        pi = t // N
        j = t % N
        oi = pi * BI + tl.arange(0, BI)
        imask = oi < N
        T = _t_tile(QT, K, c, oi, imask, j, S, sqc, sqs, sks, skn, BS, NSC, MODE, BI)
        du = tl.load(DU + (oi * N + j)[:, None] * CZ + ocz[None, :], mask=imask[:, None], other=0.0)              # [BI, CZ]
        acc += mm_act(tl.trans(du), T, MODE)
    tl.store(DWP + (c * NSPLIT + sp) * CZ * HEAD + ocz[:, None] * HEAD + od[None, :], acc)


@triton.jit
def _dq_kernel(QT, K, WP, DU, DQT, S, N, sqc, sqs, sks, skn,
               CZ: tl.constexpr, BI: tl.constexpr, BS: tl.constexpr, NSC: tl.constexpr, MODE: tl.constexpr):
    """program (c, i block): dq[:, i, c] = sum over j of k[:, j, :] dO_c^T, dO_c = du W_c^T  ([BI, 32])"""
    c = tl.program_id(0)
    pi = tl.program_id(1)
    oi = pi * BI + tl.arange(0, BI)
    imask = oi < N
    ocz, od = tl.arange(0, CZ), tl.arange(0, HEAD)
    wc = tl.load(WP + ocz[:, None] * (HEAD * HEAD) + c * HEAD + od[None, :])                         # [CZ, 32]
    for sc in range(NSC):
        os_ = sc * BS + tl.arange(0, BS)
        sm = os_ < S
        acc = tl.zeros([BS, BI], tl.float32)
        for j in range(N):
            du = tl.load(DU + (oi * N + j)[:, None] * CZ + ocz[None, :], mask=imask[:, None], other=0.0)          # [BI, CZ]
            do = mm(du, wc, MODE)                                                                    # [BI, 32]
            kt = tl.load(K + os_[:, None] * sks + j * skn + od[None, :], mask=sm[:, None], other=0.0)         # [BS, 32]
            acc += mm_act(kt, tl.trans(do), MODE)
        tl.store(DQT + c * sqc + os_[:, None] * sqs + oi[None, :], acc.to(DQT.dtype.element_ty), mask=sm[:, None] & imask[None, :])


@triton.jit
def _dk_kernel(QT, WP, DU, DKP, S, N, sqc, sqs, nti, NSPLIT, dks,
               CZ: tl.constexpr, BI: tl.constexpr, BS: tl.constexpr, NSC: tl.constexpr, MODE: tl.constexpr):
    """program (j, slice): dk[:, j, :] = sum over i, c of q[:, i, c]^T dO_c over the i blocks pi = slice, slice + NSPLIT, ..."""
    j = tl.program_id(0)
    sp = tl.program_id(1)
    ocz, od = tl.arange(0, CZ), tl.arange(0, HEAD)
    for sc in range(NSC):
        os_ = sc * BS + tl.arange(0, BS)
        sm = os_ < S
        acc = tl.zeros([BS, HEAD], tl.float32)
        for pi in range(sp, nti, NSPLIT):
            oi = pi * BI + tl.arange(0, BI)
            imask = oi < N
            du = tl.load(DU + (oi * N + j)[:, None] * CZ + ocz[None, :], mask=imask[:, None], other=0.0)
            for c in range(HEAD):
                wc = tl.load(WP + ocz[:, None] * (HEAD * HEAD) + c * HEAD + od[None, :])
                do = mm(du, wc, MODE)                                                                # [BI, 32]
                qt = tl.load(QT + c * sqc + os_[:, None] * sqs + oi[None, :], mask=sm[:, None] & imask[None, :], other=0.0)  # [BS, BI]
                acc += mm_act(qt, do, MODE)
        tl.store(DKP + (sp * S + os_)[:, None] * dks + j * HEAD + od[None, :], acc, mask=sm[:, None])


def _s_chunks(S: int):
    bs = max(64, min(128, triton.next_power_of_2(S)))        # (a TF32 dot of K = 32 crashes the Triton 3.2 compiler)
    return bs, triton.cdiv(S, bs)


@torch.library.custom_op("physdock::outer_product_mean_fwd", mutates_args=())
def _fwd(qk: torch.Tensor, z: torch.Tensor, w: torch.Tensor, b: torch.Tensor, nw: torch.Tensor, eps: float, mode: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """-> y [N, N, CZ] float32, u [N, N, CZ] (matmul dtype; the projection before the norm)"""
    S, N, _ = qk.shape
    cz = z.shape[-1]
    bi, warps, stages = _config(mode)[0]
    bs, nsc = _s_chunks(S)
    qt = qk[..., :_H].permute(2, 0, 1).contiguous()                       # [32, S, N]: the columns of q the kernels read, contiguous
    k = qk[..., _H:]
    z = z.contiguous()
    y = torch.empty_like(z)
    u = torch.empty(N, N, cz, device=z.device, dtype=compute_dtype(mode) if mode == MODES["bf16"] else torch.float32)
    _fwd_kernel[(triton.cdiv(N, bi), N)](qt, k, prep_weight(w, mode), b.contiguous(), nw.contiguous(), z, y, u, S, N, qt.stride(0), qt.stride(1),
                                         k.stride(0), k.stride(1), eps, CZ=cz, BI=bi, BS=bs, NSC=nsc, MODE=mode, num_warps=warps, num_stages=stages)
    return y, u


@_fwd.register_fake
def _(qk, z, w, b, nw, eps, mode):
    return torch.empty_like(z), z.new_empty(z.shape, dtype=torch.bfloat16 if mode == MODES["bf16"] else torch.float32)


@torch.library.custom_op("physdock::outer_product_mean_bwd", mutates_args=())
def _bwd(dy: torch.Tensor, qk: torch.Tensor, u: torch.Tensor, w: torch.Tensor, nw: torch.Tensor, eps: float,
         mode: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """dy [N, N, CZ] -> dqk [S, N, 64] (dtype of qk), d norm_w [CZ], d W [CZ, 1024], d b [CZ] (float32)"""
    S, N, _ = qk.shape
    cz = dy.shape[-1]
    dy = dy.contiguous()
    wp = prep_weight(w, mode)
    bs, nsc = _s_chunks(S)
    _, (wbi, ww, ws, wsplit), (qbi, qw, qs), (kbi, kw, ks, ksplit) = _config(mode)
    qt = qk[..., :_H].permute(2, 0, 1).contiguous()
    k = qk[..., _H:]
    M = N * N
    # 1. du
    du = torch.empty_like(u)
    nt = triton.cdiv(M, 64)
    part = torch.empty(nt, 2 * cz, device=dy.device, dtype=torch.float32)
    _du_kernel[(nt,)](dy, u, nw.contiguous(), du, part, M, eps, CZ=cz, BLOCK_M=64, num_warps=4)
    part = part.sum(0)
    # 2. dW
    dwp = torch.empty(_H, wsplit, cz, _H, device=dy.device, dtype=torch.float32)
    _dw_kernel[(_H, wsplit)](qt, k, du, dwp, S, N, qt.stride(0), qt.stride(1), k.stride(0), k.stride(1), triton.cdiv(N, wbi), wsplit, CZ=cz, BI=wbi,
                             BS=bs, NSC=nsc, MODE=mode, num_warps=ww, num_stages=ws)
    dw = dwp.sum(1).permute(1, 0, 2).reshape(cz, _H * _H)
    # 3. dq, 4. dk
    dqt = torch.empty_like(qt)
    _dq_kernel[(_H, triton.cdiv(N, qbi))](qt, k, wp, du, dqt, S, N, qt.stride(0), qt.stride(1), k.stride(0), k.stride(1), CZ=cz, BI=qbi, BS=bs,
                                          NSC=nsc, MODE=mode, num_warps=qw, num_stages=qs)
    dkp = torch.empty(ksplit, S, N * _H, device=dy.device, dtype=torch.float32)
    _dk_kernel[(N, ksplit)](qt, wp, du, dkp, S, N, qt.stride(0), qt.stride(1), triton.cdiv(N, kbi), ksplit, dkp.stride(1), CZ=cz, BI=kbi, BS=bs,
                            NSC=nsc, MODE=mode, num_warps=kw, num_stages=ks)
    dqk = torch.cat([dqt.permute(1, 2, 0), dkp.sum(0).view(S, N, _H).to(qk.dtype)], dim=-1)
    return dqk, part[:cz].clone(), dw, part[cz:].clone()


@_bwd.register_fake
def _(dy, qk, u, w, nw, eps, mode):
    return (qk.new_empty(qk.shape), dy.new_empty(nw.shape, dtype=torch.float32), dy.new_empty(w.shape, dtype=torch.float32),
            dy.new_empty(nw.shape, dtype=torch.float32))


def _setup_context(ctx, inputs, output):
    qk, z, w, b, nw, eps, mode = inputs
    ctx.save_for_backward(qk, output[1], w, nw)
    ctx.eps, ctx.mode = eps, mode


def _backward(ctx, dy, du_unused):
    qk, u, w, nw = ctx.saved_tensors
    dqk, dnw, dw, db = _bwd(dy, qk, u, w, nw, ctx.eps, ctx.mode)
    return dqk, dy, dw.to(w.dtype), db.to(w.dtype), dnw.to(nw.dtype), None, None


_fwd.register_autograd(_backward, setup_context=_setup_context)


def outer_product_mean(qk: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, norm_weight: torch.Tensor,
                       eps: float) -> torch.Tensor:
    """z + RMSNorm(linear(outer product of q and k summed over the rows)), see the module docstring (differentiable w.r.t. all tensors)

    Args:
        qk: [S, N, 64] the q (columns 0..31) and k (32..63) projections of the S MSA rows
        z: [N, N, CZ] float32; weight [CZ, 1024]; bias [CZ]; norm_weight [CZ]
    """
    return _fwd(qk, z, weight, bias, norm_weight, eps, matmul_mode())[0]
