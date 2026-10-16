"""Triton attention with a pair bias shared by the batch: the DiT attentions of the decoder (atoms, tokens) and the
triangle attention of the pair stack.

    o[b, h] = softmax(q[b, h] k[b, h]^T / sqrt(D) + bias[h]) v[b, h]

    q, k, v: [B, H, N, D] (any strides, D <= 128), bias: [H, N, N] float32 or bfloat16 (pair features + key mask), out: [B, H, N, D]

The bias is the same for every b: the B noisy samples of the DiT, or the N rows of triangle attention. Gradients for
q, k, v and bias (summed over b).

Precision (float32 inputs): TF32 matmuls with the inputs rounded to nearest, like cuBLAS does for torch's TF32 matmuls
(the Triton default truncates the low 13 mantissa bits, which makes the gradients ~6x less accurate than torch's TF32);
softmax and accumulators in float32. precision="tf32x3" (3xTF32, fp32-grade) is available at ~3x the cost. bfloat16
inputs use plain bf16 matmuls. Scores are kept in log2 units (exp2 is a hardware instruction).

Backward, one fused kernel instead of the usual three passes: grid (query block, key block, head [x batch split]),
looping over the batch inside. The bias tile is loaded once for all b, the scores are recomputed once, and dbias is
accumulated in registers (no atomics). dq, dk and dv are accumulated across blocks with relaxed atomics. When the
grid is too small to fill the GPU (triangle attention, few heads and rows) the batch is split over extra programs and
the partial dbias are summed.

The kernels run behind torch.library custom ops (opaque for torch.compile, with fake implementations).
"""
from typing import Tuple

import torch
import triton
import triton.language as tl

PRECISIONS = ("tf32", "tf32x3")
_MAX_SPLIT = 8


@triton.jit
def _round_tf32(x):
    """x rounded to the nearest TF32 value (10 mantissa bits): add half an ulp, clear the low 13 bits"""
    return ((x.to(tl.int32, bitcast=True) + 0x1000) & -8192).to(tl.float32, bitcast=True)


def _configs(blocks):
    """(BLOCK_M, BLOCK_N, num_warps, num_stages) combinations that compile on triton 3.2 / sm_90 (some crash the
    compiler: 64x64 with 8 warps, 128x128 with 3 stages)."""
    out = []
    for m, n in blocks:
        for w in (4, 8):
            for st in (1, 2, 3):
                if (m == n == 128 and st == 3) or (m == n == 64 and w == 8):
                    continue
                out.append(triton.Config({"BLOCK_M": m, "BLOCK_N": n}, num_warps=w, num_stages=st))
    return out


_FWD_BLOCKS = ((64, 64), (128, 64), (64, 128), (128, 128), (32, 64))
_BWD_BLOCKS = ((64, 64), (64, 32), (32, 64), (128, 32), (32, 128), (128, 64), (64, 128))


# ----------------------------------------------------------------------------------------------- forward
@triton.autotune(_configs(_FWD_BLOCKS), key=["N", "D_HEAD", "PREC", "RN"])
@triton.jit
def _fwd(Q, K, V, Bias, O, LSE, scale_log2e, H, N,
         sqb, sqh, sqn, skb, skh, skn, svb, svh, svn, sob, soh, son, sbh, sbm,
         D_HEAD: tl.constexpr, BLOCK_D: tl.constexpr, PREC: tl.constexpr, RN: tl.constexpr,
         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    b = tl.program_id(0)                       # fastest axis: the programs sharing a bias tile run together (L2)
    pid_m = tl.program_id(1)
    h = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_in, d_in = offs_m < N, offs_d < D_HEAD
    b64, h64 = b.to(tl.int64), h.to(tl.int64)
    q = tl.load(Q + b64 * sqb + h64 * sqh + offs_m[:, None] * sqn + offs_d[None, :],
                mask=m_in[:, None] & d_in[None, :], other=0.0)
    q = (q * scale_log2e).to(q.dtype)                       # keep the input dtype for the dot
    if RN:
        q = _round_tf32(q)
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    k_base, v_base = K + b64 * skb + h64 * skh, V + b64 * svb + h64 * svh
    bias_base = Bias + h64 * sbh + offs_m[:, None] * sbm
    for start_n in tl.range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_in = offs_n < N
        kt = tl.load(k_base + offs_n[None, :] * skn + offs_d[:, None], mask=n_in[None, :] & d_in[:, None], other=0.0)
        bias = tl.load(bias_base + offs_n[None, :], mask=m_in[:, None] & n_in[None, :], other=0.0).to(tl.float32)
        if RN:
            kt = _round_tf32(kt)
        s = tl.dot(q, kt, input_precision=PREC) + bias * 1.4426950408889634
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
            v = _round_tf32(v)
        acc += tl.dot(pp, v, input_precision=PREC)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(O + b64 * sob + h64 * soh + offs_m[:, None] * son + offs_d[None, :], acc,
             mask=m_in[:, None] & d_in[None, :])
    tl.store(LSE + (b64 * H + h64) * N + offs_m, m_i + tl.math.log2(l_i), mask=m_in)


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
@triton.autotune(_configs(_BWD_BLOCKS), key=["N", "D_HEAD", "PREC", "RN", "SPLIT"],
                 reset_to_zero=["DQ", "DK", "DV"])          # the tuning runs repeat the kernel: clear the atomic sums
@triton.jit
def _bwd(Q, K, V, Bias, DO, LSE, Delta, DQ, DK, DV, DBias, scale, scale_log2e, B, H, N,
         sqb, sqh, sqn, skb, skh, skn, svb, svh, svn, sdob, sdoh, sdon, sbh, sbm, sbs,
         D_HEAD: tl.constexpr, BLOCK_D: tl.constexpr, PREC: tl.constexpr, RN: tl.constexpr, SPLIT: tl.constexpr,
         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """One pass over the score tiles: recomputes p once and produces dq, dk, dv and dbias (see the module docstring).
    program_id(2) enumerates (head, batch chunk); chunk c sums the batch entries [c * ceil(B / SPLIT), ...)."""
    pid_m, pid_n, pid_hs = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    h, sid = pid_hs // SPLIT, pid_hs % SPLIT
    chunk = (B + SPLIT - 1) // SPLIT
    b_lo, b_hi = sid * chunk, tl.minimum(B, sid * chunk + chunk)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    m_in, n_in, d_in = offs_m < N, offs_n < N, offs_d < D_HEAD
    md, nd = m_in[:, None] & d_in[None, :], n_in[:, None] & d_in[None, :]
    h64 = h.to(tl.int64)
    tile = offs_m[:, None] * sbm + offs_n[None, :]
    bias = tl.load(Bias + h64 * sbh + tile, mask=m_in[:, None] & n_in[None, :], other=0.0).to(tl.float32) * 1.4426950408889634
    dbias = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    for b in tl.range(b_lo, b_hi):
        b64 = b.to(tl.int64)
        qo, ko, vo, doo = b64 * sqb + h64 * sqh, b64 * skb + h64 * skh, b64 * svb + h64 * svh, b64 * sdob + h64 * sdoh
        q = tl.load(Q + qo + offs_m[:, None] * sqn + offs_d[None, :], mask=md, other=0.0)
        q = (q * scale_log2e).to(q.dtype)
        kt = tl.load(K + ko + offs_n[None, :] * skn + offs_d[:, None], mask=n_in[None, :] & d_in[:, None], other=0.0)
        vt = tl.load(V + vo + offs_n[None, :] * svn + offs_d[:, None], mask=n_in[None, :] & d_in[:, None], other=0.0)
        do = tl.load(DO + doo + offs_m[:, None] * sdon + offs_d[None, :], mask=md, other=0.0)
        if RN:
            q, kt, vt, do = _round_tf32(q), _round_tf32(kt), _round_tf32(vt), _round_tf32(do)
        lse = tl.load(LSE + (b64 * H + h64) * N + offs_m, mask=m_in, other=0.0)
        delta = tl.load(Delta + (b64 * H + h64) * N + offs_m, mask=m_in, other=0.0)
        s = tl.dot(q, kt, input_precision=PREC) + bias
        p = tl.where(m_in[:, None] & n_in[None, :], tl.math.exp2(s - lse[:, None]), 0.0)
        dp = tl.dot(do, vt, input_precision=PREC).to(tl.float32)
        ds = p * (dp - delta[:, None])
        dbias += ds
        pc, dsc = p.to(do.dtype), ds.to(do.dtype)
        if RN:
            pc, dsc = _round_tf32(pc), _round_tf32(dsc)
        dv = tl.dot(tl.trans(pc), do, input_precision=PREC)
        dk = tl.dot(tl.trans(dsc), q, input_precision=PREC)
        dq = tl.dot(dsc, tl.trans(kt), input_precision=PREC)
        dq = dq * scale                                     # q was pre-scaled by scale * log2(e); dk picks it up via q
        dk = dk * 0.6931471805599453                        # 1 / log2(e)
        tl.atomic_add(DQ + qo + offs_m[:, None] * sqn + offs_d[None, :], dq, mask=md, sem="relaxed")
        tl.atomic_add(DK + ko + offs_n[:, None] * skn + offs_d[None, :], dk, mask=nd, sem="relaxed")
        tl.atomic_add(DV + vo + offs_n[:, None] * svn + offs_d[None, :], dv, mask=nd, sem="relaxed")
    tl.store(DBias + sid.to(tl.int64) * sbs + h64 * sbh + tile, dbias, mask=m_in[:, None] & n_in[None, :])


# ----------------------------------------------------------------------------------------------- torch ops
_NUM_SMS = {}


def _split(B, H, N):
    """batch chunks of the backward: enough programs (~4 per SM, counted for 64 x 64 tiles) to fill the GPU"""
    dev = torch.cuda.current_device()
    sms = _NUM_SMS.setdefault(dev, torch.cuda.get_device_properties(dev).multi_processor_count)
    programs = triton.cdiv(N, 64) ** 2 * H
    return max(1, min(B, _MAX_SPLIT, triton.cdiv(4 * sms, programs)))


def _prep(dtype, precision):
    """(tl.dot input_precision, round the inputs to TF32?)"""
    if dtype == torch.bfloat16:
        return "ieee", False                                # bf16 inputs: the plain tensor-core product
    return ("tf32", True) if precision == "tf32" else ("tf32x3", False)


@torch.library.custom_op("physdock::shared_bias_attention_fwd", mutates_args=())
def _attention_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor,
                   precision: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """-> o [B, N, H, D] contiguous (the caller transposes to [B, H, N, D]), lse [B, H, N] (log2 units)"""
    B, H, N, D = q.shape
    q, k, v = (x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v))
    if bias.stride(-1) != 1:
        bias = bias.contiguous()
    prec, rn = _prep(q.dtype, precision)
    o = torch.empty(B, N, H, D, device=q.device, dtype=q.dtype)
    ov = o.transpose(1, 2)
    lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
    _fwd[lambda c: (B, triton.cdiv(N, c["BLOCK_M"]), H)](
        q, k, v, bias, ov, lse, D ** -0.5 * 1.4426950408889634, H, N,
        *q.stride()[:3], *k.stride()[:3], *v.stride()[:3], *ov.stride()[:3], bias.stride(0), bias.stride(1),
        D_HEAD=D, BLOCK_D=max(16, triton.next_power_of_2(D)), PREC=prec, RN=rn)
    return o, lse


@_attention_fwd.register_fake
def _(q, k, v, bias, precision):
    B, H, N, D = q.shape
    return q.new_empty(B, N, H, D), q.new_empty(B, H, N, dtype=torch.float32)


@torch.library.custom_op("physdock::shared_bias_attention_bwd", mutates_args=())
def _attention_bwd(do: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor,
                   o: torch.Tensor, lse: torch.Tensor,
                   precision: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """do, o: [B, N, H, D] -> dq, dk, dv [B, N, H, D], dbias [H, N, N]"""
    B, H, N, D = q.shape
    q, k, v = (x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v))
    if bias.stride(-1) != 1:
        bias = bias.contiguous()
    if do.stride(-1) != 1:
        do = do.contiguous()
    prec, rn = _prep(q.dtype, precision)
    block_d = max(16, triton.next_power_of_2(D))
    ov, dov = o.transpose(1, 2), do.transpose(1, 2)
    delta = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
    _delta[lambda c: (triton.cdiv(N, 128), B * H)](ov, dov, delta, N, *ov.stride()[:3], *dov.stride()[:3], H,
                                                   D_HEAD=D, BLOCK_D=block_d, RN=rn, BLOCK_M=128)
    dq, dk, dv = (torch.zeros(B, N, H, D, device=q.device, dtype=torch.float32) for _ in range(3))
    split = _split(B, H, N)
    dbias = torch.empty(split, H, N, N, device=q.device, dtype=torch.float32)
    dqv, dkv, dvv = dq.transpose(1, 2), dk.transpose(1, 2), dv.transpose(1, 2)
    scale = D ** -0.5
    _bwd[lambda c: (triton.cdiv(N, c["BLOCK_M"]), triton.cdiv(N, c["BLOCK_N"]), H * split)](
        q, k, v, bias, dov, lse, delta, dqv, dkv, dvv, dbias, scale, scale * 1.4426950408889634, B, H, N,
        *q.stride()[:3], *k.stride()[:3], *v.stride()[:3], *dov.stride()[:3], bias.stride(0), bias.stride(1),
        dbias.stride(0), D_HEAD=D, BLOCK_D=block_d, PREC=prec, RN=rn, SPLIT=split)
    return dq.to(q.dtype), dk.to(q.dtype), dv.to(q.dtype), dbias[0] if split == 1 else dbias.sum(0)


@_attention_bwd.register_fake
def _(do, q, k, v, bias, o, lse, precision):
    B, H, N, D = q.shape
    return (q.new_empty(B, N, H, D), q.new_empty(B, N, H, D), q.new_empty(B, N, H, D),
            bias.new_empty(H, N, N, dtype=torch.float32))


def _setup_context(ctx, inputs, output):
    q, k, v, bias, precision = inputs
    ctx.save_for_backward(q, k, v, bias, output[0], output[1])
    ctx.precision = precision


def _backward(ctx, do, dlse):
    q, k, v, bias, o, lse = ctx.saved_tensors
    dq, dk, dv, dbias = _attention_bwd(do, q, k, v, bias, o, lse, ctx.precision)
    return dq.transpose(1, 2), dk.transpose(1, 2), dv.transpose(1, 2), dbias.to(bias.dtype), None


_attention_fwd.register_autograd(_backward, setup_context=_setup_context)


def shared_bias_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor,
                          precision: str = "tf32") -> torch.Tensor:
    """softmax(q k^T / sqrt(D) + bias) v with a bias shared by the batch.

    Args:
        q, k, v: [B, H, N, D] float32 or bfloat16
        bias: [H, N, N] attention logit bias (pair features + key mask), shared by the B batch entries
        precision: float32 matmuls, "tf32" (like torch's TF32) or "tf32x3" (fp32-grade); ignored for bfloat16

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
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    return _attention_fwd(q, k, v, bias, precision)[0].transpose(1, 2)
