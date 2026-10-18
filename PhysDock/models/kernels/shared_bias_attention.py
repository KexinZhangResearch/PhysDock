"""Triton attention with a pair bias shared by the batch (SharedBiasAttention): the DiT attentions of the decoder (atoms, tokens),
the triangle attention of the pair stack, the single attention and the MSA row attention.

    o[b, h] = softmax(q[b, h] k[b, h]^T / sqrt(D) + bias[h]) v[b, h]

    q, k, v: [B, H, N, D] (any strides, D <= 128), bias: [H, N, N] float32 or bfloat16 (pair features + key mask), out: [B, H, N, D]

The bias is the same for every b: the B noisy samples of the DiT, or the N rows of triangle attention. Gradients for
q, k, v and bias (summed over b).

Precision (float32 inputs): TF32 matmuls with the inputs rounded to nearest, like cuBLAS does for torch's TF32 matmuls
(the Triton default truncates the low 13 mantissa bits, which makes the gradients ~6x less accurate than torch's TF32);
softmax and accumulators in float32. bfloat16 inputs use plain bf16 matmuls. Scores are kept in log2 units (exp2 is a hardware instruction).

Backward, one fused kernel instead of the usual three passes: grid (query block, key block, head [x batch split]),
looping over the batch inside. The bias tile is loaded once for all b, the scores are recomputed once, and dbias is
accumulated in registers (no atomics). dq, dk and dv are accumulated across blocks with relaxed atomics. When the
grid is too small to fill the GPU (triangle attention, few heads and rows) the batch is split over extra programs and
the partial dbias are summed.

Padding: n_valid (an int32 device scalar, no host synchronisation) is the length of the leading part of the sequence that holds real
positions; the keys from n_valid on must be masked by the bias (they are skipped, which is the same as masking them), and the query
rows from n_valid on are not computed (output 0, no gradient). The training crops are padded at the end to a fixed size (a mean of
60% real atom pairs and 74% real token pairs at crop 384 / 3072 atoms), and the score tiles of the padding are skipped in the forward
and the backward.

Distance decay (decay_* arguments, off by default): a physics-style switching factor on the attention of distant atoms, for the atom DiTs.
With positions pos [B, N, 3] and a per-sample radius r [B] the score of a pair becomes

    s[i, j] -= lambda[h] * w * softplus((|pos_i - pos_j| / r[b] - 1) / w)        (w = decay_width)

i.e. the attention weight is multiplied by a factor that is ~1 well inside r and decays beyond it (a Boltzmann factor in log space;
12 A radius, lambda 10: 0.49 at 10 A, 0.25 at 12 A, 0.09 at 14 A). The caller passes r in the units of pos; for the preconditioned
input of the decoder, x_in = (x + t e) / sqrt(1 + t^2), a true radius r_A corresponds to
r[b]^2 = ((r_A / sigma_data)^2 + 6 t_b^2) / (1 + t_b^2), which keeps the factor meaningful at every noise level and switches the decay
off by itself at high noise (every pair then has the same distance, so the factor is a constant the softmax ignores).
lambda is [H] (its gradient is returned: a learnable strength per head) or a scalar broadcast over the heads (fixed strength).
The positions receive no gradient.

The kernels run behind torch.library custom ops (opaque for torch.compile, with fake implementations).
"""
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from ._common import num_sms, round_tf32_ as _round_tf32

_MAX_SPLIT = 8


def _programs(B: int, H: int, N: int) -> int:
    """number of 64-row query tiles of the forward: the tile size that is best depends on how full the GPU is"""
    return B * triton.cdiv(N, 64) * H


def _fwd_config(B: int, H: int, N: int, bf16: bool):
    """(BLOCK_M, BLOCK_N, num_warps, num_stages) of the forward, fixed (no autotuning pass: it costs minutes of compilation
    per shape). Found by sweeping all configurations on an H20 (sm_90) at the shapes of the model (DiT atoms / tokens,
    triangle attention): with a head dim of 32 the score tile dominates the registers, so small tiles with few warps win
    when there are few programs (32 x 16 with 2 warps is 1.5x faster than 64 x 64 with 4), bigger ones when the GPU is
    full anyway."""
    full = _programs(B, H, N) >= 3000
    if bf16:
        if full:
            return (64, 64, 4, 3) if N > 1024 else (64, 32, 4, 3)
        return (64, 32, 4, 3) if N > 1024 else (32, 32, 1, 2)
    if full:
        return (64, 32, 4, 3)
    if N > 1024:        # the atoms of the decoder (inputs rounded outside the kernel, see prepare_tf32)
        return (64, 32, 4, 3)
    return (32, 16, 2, 3)


def _bwd_config(N: int, bf16: bool):
    """(BLOCK_M, BLOCK_N, num_warps, num_stages) of the fused backward, fixed (see _fwd_config)"""
    if N <= 512:
        return (64, 64, 4, 2) if bf16 else (32, 32, 2, 1)
    return (64, 64, 4, 3)


@triton.jit
def _softplus(x, w):
    """w * softplus(x / w), stable for a large x / w (the switch of the distance decay, see the module docstring)"""
    z = x / w
    return tl.where(z > 20.0, x, w * tl.math.log2(1.0 + tl.math.exp2(z * 1.4426950408889634)) * 0.6931471805599453)


# ----------------------------------------------------------------------------------------------- TF32 preparation
PRE_MIN_N = 1024            # from this sequence length rounding the matmul inputs outside the kernels pays off: the pass costs
                            # ~B * H * N, the in-kernel rounding it saves ~B * H * N * (N / BLOCK_M) (K, V tiles are re-read per query block)


@triton.jit
def _prep_one(X, Y, scale, N, H, bh, pid_n, sxb, sxh, sxn, D_HEAD: tl.constexpr, BLOCK_D: tl.constexpr, SCALE: tl.constexpr,
              BLOCK_N: tl.constexpr):
    b, h = bh // H, bh % H
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask = (offs_n < N)[:, None] & (offs_d < D_HEAD)[None, :]
    x = tl.load(X + b * sxb + h * sxh + offs_n[:, None] * sxn + offs_d[None, :], mask=mask, other=0.0)
    if SCALE:
        x = x * scale
    tl.store(Y + (bh * N + offs_n[:, None]) * D_HEAD + offs_d[None, :], _round_tf32(x), mask=mask)


@triton.jit
def _prep_kernel(X0, X1, X2, Y, scale, N, H, BH, s0b, s0h, s0n, s1b, s1h, s1n, s2b, s2h, s2n,
                 NT: tl.constexpr, D_HEAD: tl.constexpr, BLOCK_D: tl.constexpr, SCALE0: tl.constexpr, BLOCK_N: tl.constexpr):
    """Y[t] = round_tf32(X_t * scale if t == 0 and SCALE0 else X_t), contiguous [NT, B, H, N, D]; one launch for all NT tensors"""
    pid_n, bh, t = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2)
    if t == 0:
        _prep_one(X0, Y, scale, N, H, bh, pid_n, s0b, s0h, s0n, D_HEAD, BLOCK_D, SCALE0, BLOCK_N)
    elif t == 1:
        _prep_one(X1, Y + BH * N * D_HEAD, scale, N, H, bh, pid_n, s1b, s1h, s1n, D_HEAD, BLOCK_D, False, BLOCK_N)
    else:
        _prep_one(X2, Y + 2 * BH * N * D_HEAD, scale, N, H, bh, pid_n, s2b, s2h, s2n, D_HEAD, BLOCK_D, False, BLOCK_N)


def prepare_tf32(tensors, scale: float = None):
    """tensors: 1-3 float32 [B, H, N, D] (any strides) -> list of contiguous tensors, rounded to the nearest TF32 value (the
    rounding the kernels apply to their matmul inputs, here done once instead of once per tile); the first is multiplied by
    `scale` before the rounding. One kernel launch."""
    x0 = tensors[0]
    B, H, N, D = x0.shape
    xs = list(tensors) + [x0] * (3 - len(tensors))
    y = torch.empty(len(tensors), B, H, N, D, device=x0.device, dtype=torch.float32)
    strides = [v for x in xs for v in x.stride()[:3]]
    _prep_kernel[(triton.cdiv(N, 128), B * H, len(tensors))](*xs, y, 1.0 if scale is None else scale, N, H, B * H, *strides, NT=len(tensors),
                                                           D_HEAD=D, BLOCK_D=max(16, triton.next_power_of_2(D)), SCALE0=scale is not None,
                                                           BLOCK_N=128)
    return list(y.unbind(0))


# ----------------------------------------------------------------------------------------------- forward
@triton.jit
def _fwd(Q, K, V, Bias, NV, POS, RAD, LAM, O, LSE, scale_log2e, H, N, spb, spn, decay_w,
         sqb, sqh, sqn, skb, skh, skn, svb, svh, svn, sob, soh, son, sbh, sbm,
         D_HEAD: tl.constexpr, BLOCK_D: tl.constexpr, PREC: tl.constexpr, RN: tl.constexpr, PRE: tl.constexpr,
         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HAS_NV: tl.constexpr, DECAY: tl.constexpr):
    """PRE: q (already multiplied by scale * log2(e)), k and v are TF32-rounded by the caller (prepare_tf32), so the kernel
    loads them straight into the matmuls (the in-kernel rounding forced plain register loads instead of asynchronous,
    pipelined copies)."""
    b = tl.program_id(0)                       # fastest axis: the programs sharing a bias tile run together (L2)
    pid_m = tl.program_id(1)
    h = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_in, d_in = offs_m < N, offs_d < D_HEAD
    b64, h64 = b.to(tl.int64), h.to(tl.int64)
    q = tl.load(Q + b64 * sqb + h64 * sqh + offs_m[:, None] * sqn + offs_d[None, :],
                mask=m_in[:, None] & d_in[None, :], other=0.0)
    if not PRE:
        q = (q * scale_log2e).to(q.dtype)                   # keep the input dtype for the dot
        if RN:
            q = _round_tf32(q)
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    k_base, v_base = K + b64 * skb + h64 * skh, V + b64 * svb + h64 * svh
    bias_base = Bias + h64 * sbh + offs_m[:, None] * sbm
    if DECAY:
        pbase = POS + b64 * spb + offs_m * spn
        xq0 = tl.load(pbase + 0, mask=m_in, other=0.0)
        xq1 = tl.load(pbase + 1, mask=m_in, other=0.0)
        xq2 = tl.load(pbase + 2, mask=m_in, other=0.0)
        inv_r = 1.0 / tl.load(RAD + b64)
        lam = tl.load(LAM + h64) * 1.4426950408889634                  # the scores are in log2 units
    if HAS_NV:
        nv = tl.load(NV)
        hi = tl.where(pid_m * BLOCK_M < nv, tl.minimum(N, (nv + BLOCK_N - 1) // BLOCK_N * BLOCK_N), 0)   # a query tile of padding attends to nothing
    else:
        nv = N
        hi = N
    for start_n in tl.range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_in = offs_n < N
        kt = tl.load(k_base + offs_n[None, :] * skn + offs_d[:, None], mask=n_in[None, :] & d_in[:, None], other=0.0)
        # the mask of the bias load uses the row stride (a multiple of 16: the padded layout of aligned_bias), not N: a mask
        # that is constant over 4 neighbouring columns lets Triton issue 16-byte loads (the columns N .. stride are padding,
        # their scores are discarded below)
        bias = tl.load(bias_base + offs_n[None, :], mask=m_in[:, None] & (offs_n < sbm)[None, :], other=0.0).to(tl.float32)
        if RN and not PRE:
            kt = _round_tf32(kt)
        s = tl.dot(q, kt, input_precision=PREC) + bias * 1.4426950408889634
        if DECAY:
            kbase = POS + b64 * spb + offs_n * spn
            e0 = xq0[:, None] - tl.load(kbase + 0, mask=n_in, other=0.0)[None, :]
            e1 = xq1[:, None] - tl.load(kbase + 1, mask=n_in, other=0.0)[None, :]
            e2 = xq2[:, None] - tl.load(kbase + 2, mask=n_in, other=0.0)[None, :]
            dd = tl.sqrt(e0 * e0 + e1 * e1 + e2 * e2) * inv_r
            s -= lam * _softplus(dd - 1.0, decay_w)
        s = tl.where(n_in[None, :], s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_base + offs_n[:, None] * svn + offs_d[None, :], mask=n_in[:, None] & d_in[None, :], other=0.0)
        acc = acc * alpha[:, None]
        pp = p.to(v.dtype)
        if RN:
            pp = _round_tf32(pp)
            if not PRE:
                v = _round_tf32(v)
        acc += tl.dot(pp, v, input_precision=PREC)
        m_i = m_new
    live = (l_i > 0.0) & (offs_m < nv)                          # the rows of the padding: output 0, lse 0 (nothing reads them)
    acc = tl.where(live[:, None], acc / tl.where(live, l_i, 1.0)[:, None], 0.0)
    tl.store(O + b64 * sob + h64 * soh + offs_m[:, None] * son + offs_d[None, :], acc,
             mask=m_in[:, None] & d_in[None, :])
    tl.store(LSE + (b64 * H + h64) * N + offs_m, tl.where(live, m_i + tl.math.log2(tl.where(live, l_i, 1.0)), 0.0), mask=m_in)


@triton.jit
def _delta(O, DO, Delta, N, sob, soh, son, sdb, sdh, sdn, H,
           D_HEAD: tl.constexpr, BLOCK_D: tl.constexpr, RN: tl.constexpr, BLOCK_M: tl.constexpr):
    """delta[b, h, m] = sum_d o * do (do rounded like the dO V^T matmul sees it: the softmax backward subtracts delta
    from dO V^T and a mismatch between the two does not cancel)"""
    pid_m, bh = tl.program_id(0), tl.program_id(1).to(tl.int64)
    b, h = bh // H, bh % H
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask = (offs_m < N)[:, None] & (offs_d < D_HEAD)[None, :]
    o = tl.load(O + b * sob + h * soh + offs_m[:, None] * son + offs_d[None, :], mask=mask, other=0.0)
    do = tl.load(DO + b * sdb + h * sdh + offs_m[:, None] * sdn + offs_d[None, :], mask=mask, other=0.0)
    if RN:
        do = _round_tf32(do)
    tl.store(Delta + bh * N + offs_m, tl.sum(o.to(tl.float32) * do.to(tl.float32), 1), mask=offs_m < N)


# ----------------------------------------------------------------------------------------------- backward
@triton.jit
def _bwd(Q, K, V, Bias, NV, POS, RAD, LAM, DLAM, DO, LSE, Delta, DQ, DK, DV, DBias, scale, scale_log2e, B, H, N, spb, spn, decay_w,
         sqb, sqh, sqn, skb, skh, skn, svb, svh, svn, sdob, sdoh, sdon, sdqb, sdqh, sdqn, sdkb, sdkh, sdkn, sdvb, sdvh, sdvn,
         sbh, sbm, sdbs, sdbh, sdbm,
         D_HEAD: tl.constexpr, BLOCK_D: tl.constexpr, PREC: tl.constexpr, RN: tl.constexpr, SPLIT: tl.constexpr,
         PRE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HAS_NV: tl.constexpr, DECAY: tl.constexpr):
    """One pass over the score tiles: recomputes p once and produces dq, dk, dv and dbias (see the module docstring).
    program_id(2) enumerates (head, batch chunk); chunk c sums the batch entries [c * ceil(B / SPLIT), ...)."""
    pid_m, pid_n, pid_hs = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    h, sid = pid_hs // SPLIT, pid_hs % SPLIT
    chunk = (B + SPLIT - 1) // SPLIT
    b_lo, b_hi = sid * chunk, tl.minimum(B, sid * chunk + chunk)
    if HAS_NV:
        nv = tl.load(NV)
        active = (pid_m * BLOCK_M < nv) & (pid_n * BLOCK_N < nv)      # a tile that touches padding only: dbias = 0, nothing else
        b_hi = tl.where(active, b_hi, b_lo)
    else:
        nv = N
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    m_in, n_in, d_in = offs_m < N, offs_n < N, offs_d < D_HEAD
    md, nd = m_in[:, None] & d_in[None, :], n_in[:, None] & d_in[None, :]
    h64 = h.to(tl.int64)
    tile = offs_m[:, None] * sbm + offs_n[None, :]
    bias = tl.load(Bias + h64 * sbh + tile, mask=m_in[:, None] & (offs_n < sbm)[None, :], other=0.0).to(tl.float32) * 1.4426950408889634
    dbias = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    if DECAY:
        lam = tl.load(LAM + h64) * 1.4426950408889634
        dlam = tl.zeros([1], tl.float32)
    for b in tl.range(b_lo, b_hi):
        b64 = b.to(tl.int64)
        qo, ko, vo, doo = b64 * sqb + h64 * sqh, b64 * skb + h64 * skh, b64 * svb + h64 * svh, b64 * sdob + h64 * sdoh
        dqo, dko, dvo = b64 * sdqb + h64 * sdqh, b64 * sdkb + h64 * sdkh, b64 * sdvb + h64 * sdvh
        q = tl.load(Q + qo + offs_m[:, None] * sqn + offs_d[None, :], mask=md, other=0.0)
        if not PRE:
            q = (q * scale_log2e).to(q.dtype)
        kt = tl.load(K + ko + offs_n[None, :] * skn + offs_d[:, None], mask=n_in[None, :] & d_in[:, None], other=0.0)
        vt = tl.load(V + vo + offs_n[None, :] * svn + offs_d[:, None], mask=n_in[None, :] & d_in[:, None], other=0.0)
        do = tl.load(DO + doo + offs_m[:, None] * sdon + offs_d[None, :], mask=md, other=0.0)
        if RN and not PRE:
            q, kt, vt, do = _round_tf32(q), _round_tf32(kt), _round_tf32(vt), _round_tf32(do)
        lse = tl.load(LSE + (b64 * H + h64) * N + offs_m, mask=m_in, other=0.0)
        delta = tl.load(Delta + (b64 * H + h64) * N + offs_m, mask=m_in, other=0.0)
        s = tl.dot(q, kt, input_precision=PREC) + bias
        if DECAY:                                           # the same term the forward subtracted (recomputed, nothing saved)
            qbase, kbase = POS + b64 * spb + offs_m * spn, POS + b64 * spb + offs_n * spn
            e0 = tl.load(qbase + 0, mask=m_in, other=0.0)[:, None] - tl.load(kbase + 0, mask=n_in, other=0.0)[None, :]
            e1 = tl.load(qbase + 1, mask=m_in, other=0.0)[:, None] - tl.load(kbase + 1, mask=n_in, other=0.0)[None, :]
            e2 = tl.load(qbase + 2, mask=m_in, other=0.0)[:, None] - tl.load(kbase + 2, mask=n_in, other=0.0)[None, :]
            dd = tl.sqrt(e0 * e0 + e1 * e1 + e2 * e2) / tl.load(RAD + b64)
            fu = _softplus(dd - 1.0, decay_w)
            s -= lam * fu
        p = tl.where((m_in & (offs_m < nv))[:, None] & (n_in & (offs_n < nv))[None, :], tl.math.exp2(s - lse[:, None]), 0.0)
        dp = tl.dot(do, vt, input_precision=PREC).to(tl.float32)
        ds = p * (dp - delta[:, None])
        dbias += ds
        if DECAY:                                           # d/d lambda of (-lambda * f) summed over the tile and the batch
            dlam += tl.sum(tl.sum(-ds * fu, 1), 0)
        pc, dsc = p.to(do.dtype), ds.to(do.dtype)
        if RN:
            pc, dsc = _round_tf32(pc), _round_tf32(dsc)
        dv = tl.dot(tl.trans(pc), do, input_precision=PREC)
        dk = tl.dot(tl.trans(dsc), q, input_precision=PREC)
        dq = tl.dot(dsc, tl.trans(kt), input_precision=PREC)
        dq = dq * scale                                     # q was pre-scaled by scale * log2(e); dk picks it up via q
        dk = dk * 0.6931471805599453                        # 1 / log2(e)
        tl.atomic_add(DQ + dqo + offs_m[:, None] * sdqn + offs_d[None, :], dq, mask=md, sem="relaxed")
        tl.atomic_add(DK + dko + offs_n[:, None] * sdkn + offs_d[None, :], dk, mask=nd, sem="relaxed")
        tl.atomic_add(DV + dvo + offs_n[:, None] * sdvn + offs_d[None, :], dv, mask=nd, sem="relaxed")
    tl.store(DBias + sid.to(tl.int64) * sdbs + h64 * sdbh + offs_m[:, None] * sdbm + offs_n[None, :], dbias,
             mask=m_in[:, None] & (offs_n < sdbm)[None, :])      # vectorised stores: the padding columns are written too
    if DECAY:
        tl.atomic_add(DLAM + h64 + tl.zeros([1], tl.int32), dlam, sem="relaxed")


# ----------------------------------------------------------------------------------------------- torch ops
def _split(B, H, N):
    """batch chunks of the backward: enough programs (~4 per SM, counted for 64 x 64 tiles) to fill the GPU"""
    programs = triton.cdiv(N, 64) ** 2 * H
    return max(1, min(B, _MAX_SPLIT, triton.cdiv(4 * num_sms(), programs)))


def _pad16(n: int) -> int:
    return (n + 15) // 16 * 16


def aligned_bias(bias: torch.Tensor) -> torch.Tensor:
    """bias [..., N, N] with a row stride that is a multiple of 16 elements: Triton only vectorises (16-byte) loads of a
    tile whose row stride it knows to be divisible by 16, and the kernels are ~10% faster with them for N that is not a
    multiple of 16 (N = 2407 atoms). Returns bias itself if it is aligned already; DiT.pair_bias emits the aligned
    layout once per structure, so the attention calls of the denoising steps do not copy."""
    N = bias.shape[-1]
    if bias.stride(-1) == 1 and all(s % 16 == 0 for s in bias.stride()[:-1]):
        return bias
    out = torch.empty(*bias.shape[:-1], _pad16(N), device=bias.device, dtype=bias.dtype)
    out[..., :N] = bias
    return out[..., :N]


def _prep(dtype):
    """(tl.dot input_precision, round the inputs to TF32?)"""
    if dtype == torch.bfloat16:
        return "ieee", False                                # bf16 inputs: the plain tensor-core product
    return "tf32", True


def radius_for_noise(t: torch.Tensor, radius_angstrom: float, sigma_data: float) -> torch.Tensor:
    """Radius of the decay in the preconditioned input x_in = (x + t e) / sqrt(1 + t^2) that corresponds to `radius_angstrom` of the真 structure:
    r_in^2 = ((r_A / sigma_data)^2 + 6 t^2) / (1 + t^2), the same relation the distances of two atoms follow (6 = E|e_i - e_j|^2 over 3 axes).

    Args:
        t: [B] noise level in normalised units, radius_angstrom: the radius in Angstrom, sigma_data: the coordinate scale

    Returns:
        [B] radius in the units of x_in
    """
    r = radius_angstrom / sigma_data
    t2 = t.detach().float() ** 2
    return torch.sqrt((r * r + 6.0 * t2) / (1.0 + t2))


def _decay_args(pos, radius, lam, q):
    """(pos, radius, lam, on) with placeholders when the decay is off (the kernels then compile without it)"""
    if pos is None:
        z = q.new_zeros(1)
        return z, z, z, False
    return pos, radius, lam, True


@torch.library.custom_op("physdock::shared_bias_attention_fwd", mutates_args=())
def _attention_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, n_valid: Optional[torch.Tensor],
                   pos: Optional[torch.Tensor], radius: Optional[torch.Tensor], lam: Optional[torch.Tensor],
                   decay_width: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """-> o [B, N, H, D] contiguous (the caller transposes to [B, H, N, D]), lse [B, H, N] (log2 units)"""
    B, H, N, D = q.shape
    q, k, v = (x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v))
    bias = aligned_bias(bias)
    prec, rn = _prep(q.dtype)
    o = torch.empty(B, N, H, D, device=q.device, dtype=q.dtype)
    ov = o.transpose(1, 2)
    lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
    bm, bn, warps, stages = _fwd_config(B, H, N, q.dtype == torch.bfloat16)
    pre = rn and N >= PRE_MIN_N
    if pre:
        q, k, v = prepare_tf32([q, k, v], D ** -0.5 * 1.4426950408889634)
    ps, rs, ls, decay = _decay_args(pos, radius, lam, q)
    _fwd[(B, triton.cdiv(N, bm), H)](
        q, k, v, bias, n_valid, ps, rs, ls, ov, lse, D ** -0.5 * 1.4426950408889634, H, N,
        ps.stride(0) if decay else 0, ps.stride(1) if decay else 0, decay_width,
        *q.stride()[:3], *k.stride()[:3], *v.stride()[:3], *ov.stride()[:3], bias.stride(0), bias.stride(1),
        D_HEAD=D, BLOCK_D=max(16, triton.next_power_of_2(D)), PREC=prec, RN=rn, PRE=pre,
        BLOCK_M=bm, BLOCK_N=bn, HAS_NV=n_valid is not None, DECAY=decay, num_warps=warps, num_stages=stages)
    return o, lse


@_attention_fwd.register_fake
def _(q, k, v, bias, n_valid, pos, radius, lam, decay_width):
    B, H, N, D = q.shape
    return q.new_empty(B, N, H, D), q.new_empty(B, H, N, dtype=torch.float32)


@torch.library.custom_op("physdock::shared_bias_attention_bwd", mutates_args=())
def _attention_bwd(do: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, n_valid: Optional[torch.Tensor],
                   pos: Optional[torch.Tensor], radius: Optional[torch.Tensor], lam: Optional[torch.Tensor], decay_width: float,
                   o: torch.Tensor, lse: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """do, o: [B, N, H, D] -> dq, dk, dv [B, N, H, D], dbias [H, N, N], dlambda [H] (zeros when the decay is off)"""
    B, H, N, D = q.shape
    q, k, v = (x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v))
    bias = aligned_bias(bias)
    if do.stride(-1) != 1:
        do = do.contiguous()
    prec, rn = _prep(q.dtype)
    block_d = max(16, triton.next_power_of_2(D))
    ov, dov = o.transpose(1, 2), do.transpose(1, 2)
    delta = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
    _delta[lambda c: (triton.cdiv(N, 128), B * H)](ov, dov, delta, N, *ov.stride()[:3], *dov.stride()[:3], H,
                                                   D_HEAD=D, BLOCK_D=block_d, RN=rn, BLOCK_M=128)
    dq, dk, dv = (torch.zeros(B, N, H, D, device=q.device, dtype=torch.float32) for _ in range(3))
    split = _split(B, H, N)
    dbias = torch.empty(split, H, N, _pad16(N), device=q.device, dtype=torch.float32)[..., :N]
    dqv, dkv, dvv = dq.transpose(1, 2), dk.transpose(1, 2), dv.transpose(1, 2)
    scale = D ** -0.5
    bm, bn, warps, stages = _bwd_config(N, q.dtype == torch.bfloat16)
    dtype = q.dtype
    pre = rn and N >= PRE_MIN_N
    if pre:                                         # TF32 rounding of the matmul inputs once, not once per tile
        q, k, v = prepare_tf32([q, k, v], scale * 1.4426950408889634)
        dov = prepare_tf32([dov])[0]
    ps, rs, ls, decay = _decay_args(pos, radius, lam, q)
    dlam = torch.zeros(H, device=q.device, dtype=torch.float32)
    _bwd[(triton.cdiv(N, bm), triton.cdiv(N, bn), H * split)](
        q, k, v, bias, n_valid, ps, rs, ls, dlam, dov, lse, delta, dqv, dkv, dvv, dbias, scale, scale * 1.4426950408889634, B, H, N,
        ps.stride(0) if decay else 0, ps.stride(1) if decay else 0, decay_width,
        *q.stride()[:3], *k.stride()[:3], *v.stride()[:3], *dov.stride()[:3], *dqv.stride()[:3], *dkv.stride()[:3],
        *dvv.stride()[:3], bias.stride(0), bias.stride(1), dbias.stride(0), dbias.stride(1), dbias.stride(2), D_HEAD=D, BLOCK_D=block_d, PREC=prec, RN=rn,
        SPLIT=split, PRE=pre, BLOCK_M=bm, BLOCK_N=bn, HAS_NV=n_valid is not None, DECAY=decay, num_warps=warps, num_stages=stages)
    return dq.to(dtype), dk.to(dtype), dv.to(dtype), dbias[0] if split == 1 else dbias.sum(0), dlam


@_attention_bwd.register_fake
def _(do, q, k, v, bias, n_valid, pos, radius, lam, decay_width, o, lse):
    B, H, N, D = q.shape
    return (q.new_empty(B, N, H, D), q.new_empty(B, N, H, D), q.new_empty(B, N, H, D),
            bias.new_empty(H, N, N, dtype=torch.float32), q.new_empty(H, dtype=torch.float32))


def _setup_context(ctx, inputs, output):
    q, k, v, bias, n_valid, pos, radius, lam, decay_width = inputs
    ctx.save_for_backward(q, k, v, bias, output[0], output[1])
    ctx.n_valid, ctx.pos, ctx.radius, ctx.lam, ctx.decay_width = n_valid, pos, radius, lam, decay_width


def _backward(ctx, do, dlse):
    q, k, v, bias, o, lse = ctx.saved_tensors
    dq, dk, dv, dbias, dlam = _attention_bwd(do, q, k, v, bias, ctx.n_valid, ctx.pos, ctx.radius, ctx.lam, ctx.decay_width, o, lse)
    dlam = dlam.to(ctx.lam.dtype) if ctx.lam is not None and ctx.lam.requires_grad else None
    return dq.transpose(1, 2), dk.transpose(1, 2), dv.transpose(1, 2), dbias.to(bias.dtype), None, None, None, dlam, None


_attention_fwd.register_autograd(_backward, setup_context=_setup_context)


def shared_bias_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, n_valid: Optional[torch.Tensor] = None,
                          pos: Optional[torch.Tensor] = None, radius: Optional[torch.Tensor] = None,
                          lam: Optional[torch.Tensor] = None, decay_width: float = 0.2) -> torch.Tensor:
    """softmax(q k^T / sqrt(D) + bias) v with a bias shared by the batch.

    Args:
        q, k, v: [B, H, N, D] float32 or bfloat16
        bias: [H, N, N] attention logit bias (pair features + key mask), shared by the B batch entries
        n_valid: int32 tensor of one element on the device, the number of leading positions that are real (the rest is padding, see the
            module docstring); None: all N
        pos: [B, N, 3] positions of the distance decay (None: no decay, nothing is computed for it), radius: [B] its radius in the units
            of pos, lam: [H] or [1] its strength per head (differentiable), decay_width: width of the switch (see the module docstring)

    Returns:
        [B, H, N, D]
    """
    B, H, N, D = q.shape
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError(f"q, k, v must have the same shape [B, H, N, D], got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}")
    if tuple(bias.shape) != (H, N, N):
        raise ValueError(f"bias must be [H, N, N] = {(H, N, N)}, got {tuple(bias.shape)}")
    if q.dtype not in (torch.float32, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError(f"q, k, v must be float32 or bfloat16 of the same dtype, got {q.dtype}, {k.dtype}, {v.dtype}")
    if D > 128:
        raise ValueError(f"head dim must be <= 128, got {D}")
    if pos is not None:
        if pos.shape[:2] != (B, N) or pos.shape[2] != 3 or radius is None or lam is None:
            raise ValueError(f"the decay needs pos [B, N, 3] = {(B, N, 3)} (got {tuple(pos.shape)}), radius [B] and lam [H] or [1]")
        pos, radius = pos.detach().float().contiguous(), radius.detach().float().reshape(-1).contiguous()
        lam = lam.float().reshape(-1)
        if lam.numel() == 1:
            lam = lam.expand(H)
        lam = lam.contiguous()
    return _attention_fwd(q, k, v, bias, n_valid, pos, radius, lam, float(decay_width))[0].transpose(1, 2)
