"""The attention of Pomelo's DiT blocks as Triton kernels (forward and backward), for a batch of noisy samples of one structure:

    s[b, h, i, j] = scale * q_i . k_j  -  w[b, h] |p_i - p_j|^2  +  wc[h] [chain_i == chain_j]  +  we[h] [entity_i == entity_j]      (keys j < n_valid)
    o[b, h, i]    = sum_j softmax_j(s) v_j

    q, k: [B, H, N, Dqk], v: [B, H, N, Dv] (any widths up to 256; Pomelo: the typed channels of a head flattened, v with the positions appended: the direction
    messages come out of the same sum), p: [B, N, 3] (the positions of the attention: noisy atoms / token centres), w: [B, H] >= 0, ids: [N, 2] (chain, entity),
    wc / we: [H, 2] (code_weight)

What the torch path (PhysDock.pomelo.primitives.attentions) builds as extra columns of q and k (4 for the distance, 2 x code_dim for the chain / entity codes) is
computed here from the positions and the ids, in float32: the softmax sees -w |p_i - p_j|^2 where the torch path has 2 w p_i . p_j - w |p_j|^2 (the term of the
query is a constant the softmax drops: the same probabilities and gradients), and the exact [same chain] / [same entity] where the torch path has the dot product of
random orthonormal codes (exact up to code_dim chains; the kernel has no such limit). The head widths are not padded to a power of two: a width is split into two
blocks of power-of-two sizes (76 = 64 + 16, the atoms of Pomelo mini; the extra columns of the torch path took the q / k width to 112, computed at 128).

Gradients: q, k, v, w (per sample and head) and the code weights (per head); the positions and the ids get none (they are inputs of the network). Backward as
FlashAttention-2: one kernel per key block (dk, dv and the gradients of the weights, accumulated in registers, one atomic add per program) and one per query block (dq).

Padding: n_valid (an int32 device tensor, no host synchronisation): the leading positions that are real (PhysDock pads the crop at the end); the keys from n_valid on are
masked, the query rows from n_valid on are not computed (output 0, no gradient: the model never reads them; the torch path lets them attend to themselves).

Precision (the matmuls; the scores, the softmax, the bias terms and the accumulators are float32 in every mode), following the dtype of q, k, v and torch's float32
matmul precision as PhysDock's kernels (PhysDock.models.kernels._common): float16 / bfloat16 inputs (autocast): half matmuls; float32 inputs: 3xTF32 (the accuracy
of float32) under torch.set_float32_matmul_precision("highest"), TF32 with the inputs rounded to nearest under "high". The launch configurations are fixed (no
autotuning), swept on an H20 at the shapes of Pomelo mini (scripts/diagnose/pomelo_kernel_speed.py).

The kernels run behind torch.library custom ops (opaque for torch.compile, with fake implementations).
"""
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634
MODES = {"tf32x3": 0, "tf32": 1, "bf16": 2, "fp16": 3}


# ------------------------------------------------------------------------------------------------------------------------------------------- helpers
def split_width(d: int) -> Tuple[int, int]:
    """a head width as two power-of-two blocks (a, b), a + b >= d, b = 0 or >= 16: 76 -> (64, 16), 79 -> (64, 16), 54 -> (32, 32), 32 -> (32, 0)"""
    if d <= 16:
        return 16, 0
    a = 1 << (d.bit_length() - 1)
    if a == d:
        return a, 0
    rest = d - a
    b = max(16, 1 << (rest - 1).bit_length())
    return (2 * a, 0) if b >= a else (a, b)


@triton.jit
def _rnd(x):
    """float32 rounded to the nearest TF32 value (the Triton TF32 path truncates)"""
    return ((x.to(tl.int32, bitcast=True) + 0x1000) & -8192).to(tl.float32, bitcast=True)


@triton.jit
def _dot(a, b, MODE: tl.constexpr):
    """a @ b with float32 accumulation: 3xTF32 / TF32 (inputs rounded to nearest) for float32 operands, the plain half product otherwise"""
    if MODE == 0:
        return tl.dot(a, b, input_precision="tf32x3")
    elif MODE == 1:
        return tl.dot(_rnd(a), _rnd(b), input_precision="tf32")
    else:
        return tl.dot(a, b)


@triton.jit
def _load_rows(base, offs, stride, row_mask, D: tl.constexpr, START: tl.constexpr, WIDTH: tl.constexpr):
    """the columns START .. START + WIDTH of rows offs: [len(offs), WIDTH], 0 beyond D or outside row_mask"""
    cols = START + tl.arange(0, WIDTH)
    return tl.load(base + offs[:, None] * stride + cols[None, :], mask=row_mask[:, None] & (cols < D)[None, :], other=0.0)


@triton.jit
def _load_cols(base, offs, stride, row_mask, D: tl.constexpr, START: tl.constexpr, WIDTH: tl.constexpr):
    """the transposed block: [WIDTH, len(offs)]"""
    cols = START + tl.arange(0, WIDTH)
    return tl.load(base + offs[None, :] * stride + cols[:, None], mask=row_mask[None, :] & (cols < D)[:, None], other=0.0)


@triton.jit
def _store_rows(base, offs, stride, row_mask, x, D: tl.constexpr, START: tl.constexpr, WIDTH: tl.constexpr):
    cols = START + tl.arange(0, WIDTH)
    tl.store(base + offs[:, None] * stride + cols[None, :], x, mask=row_mask[:, None] & (cols < D)[None, :])


@triton.jit
def _bias(s, offs_m, offs_n, m_in, n_in, POS, spb, b64, IDS, N, w, cwa, cwe, DIST: tl.constexpr, CODES: tl.constexpr):
    """s [BM, BN] (log2 units) + the distance and the chain / entity terms (w, cwa, cwe already in log2 units); also returns |p_i - p_j|^2 and the two indicators
    (for the gradients of the weights)"""
    d2 = s * 0.0
    sa = s * 0.0
    se = s * 0.0
    if DIST:
        pq, pk = POS + b64 * spb + offs_m, POS + b64 * spb + offs_n
        e = tl.load(pq, mask=m_in, other=0.0)[:, None] - tl.load(pk, mask=n_in, other=0.0)[None, :]
        d2 = e * e
        e = tl.load(pq + N, mask=m_in, other=0.0)[:, None] - tl.load(pk + N, mask=n_in, other=0.0)[None, :]
        d2 += e * e
        e = tl.load(pq + 2 * N, mask=m_in, other=0.0)[:, None] - tl.load(pk + 2 * N, mask=n_in, other=0.0)[None, :]
        d2 += e * e
        s = s - w * d2
    if CODES:
        ca_q = tl.load(IDS + offs_m, mask=m_in, other=-1)
        ca_k = tl.load(IDS + offs_n, mask=n_in, other=-2)
        ce_q = tl.load(IDS + N + offs_m, mask=m_in, other=-1)
        ce_k = tl.load(IDS + N + offs_n, mask=n_in, other=-2)
        sa = (ca_q[:, None] == ca_k[None, :]).to(tl.float32)
        se = (ce_q[:, None] == ce_k[None, :]).to(tl.float32)
        s = s + cwa * sa + cwe * se
    return s, d2, sa, se


# ------------------------------------------------------------------------------------------------------------------------------------------- forward
@triton.jit
def _fwd(Q, K, V, POS, W, IDS, CW, NV, O, LSE, qk_scale, H, N, spb,
         sqb, sqh, sqn, skb, skh, skn, svb, svh, svn, sob, soh, son,
         DQK: tl.constexpr, QA: tl.constexpr, QB: tl.constexpr, DV: tl.constexpr, VA: tl.constexpr, VB: tl.constexpr,
         DIST: tl.constexpr, CODES: tl.constexpr, MODE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    b64, h64 = b.to(tl.int64), h.to(tl.int64)
    nv = tl.load(NV)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_in = offs_m < nv
    qb, kb, vb = Q + b64 * sqb + h64 * sqh, K + b64 * skb + h64 * skh, V + b64 * svb + h64 * svh
    q_a = _load_rows(qb, offs_m, sqn, m_in, DQK, 0, QA)
    if QB > 0:
        q_b = _load_rows(qb, offs_m, sqn, m_in, DQK, QA, QB)
    w = 0.0
    cwa = 0.0
    cwe = 0.0
    if DIST:
        w = tl.load(W + b64 * H + h64) * 1.4426950408889634
    if CODES:
        cwa = tl.load(CW + 2 * h64) * 1.4426950408889634
        cwe = tl.load(CW + 2 * h64 + 1) * 1.4426950408889634
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc_a = tl.zeros([BLOCK_M, VA], tl.float32)
    if VB > 0:
        acc_b = tl.zeros([BLOCK_M, VB], tl.float32)
    hi = tl.where(pid_m * BLOCK_M < nv, nv, 0)                                     # a query tile of padding attends to nothing
    for start_n in tl.range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_in = offs_n < nv
        s = _dot(q_a, _load_cols(kb, offs_n, skn, n_in, DQK, 0, QA), MODE)
        if QB > 0:
            s += _dot(q_b, _load_cols(kb, offs_n, skn, n_in, DQK, QA, QB), MODE)
        s, d2, sa, se = _bias(s * qk_scale, offs_m, offs_n, m_in, n_in, POS, spb, b64, IDS, N, w, cwa, cwe, DIST, CODES)
        s = tl.where(n_in[None, :], s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v_a = _load_rows(vb, offs_n, svn, n_in, DV, 0, VA)
        acc_a = acc_a * alpha[:, None] + _dot(p.to(v_a.dtype), v_a, MODE)
        if VB > 0:
            v_b = _load_rows(vb, offs_n, svn, n_in, DV, VA, VB)
            acc_b = acc_b * alpha[:, None] + _dot(p.to(v_b.dtype), v_b, MODE)
        m_i = m_new
    live = m_in & (l_i > 0.0)
    inv = tl.where(live, 1.0 / tl.where(live, l_i, 1.0), 0.0)
    ob = O + b64 * sob + h64 * soh
    m_all = offs_m < N
    _store_rows(ob, offs_m, son, m_all, (acc_a * inv[:, None]).to(O.dtype.element_ty), DV, 0, VA)
    if VB > 0:
        _store_rows(ob, offs_m, son, m_all, (acc_b * inv[:, None]).to(O.dtype.element_ty), DV, VA, VB)
    tl.store(LSE + (b64 * H + h64) * N + offs_m, tl.where(live, m_i + tl.math.log2(tl.where(live, l_i, 1.0)), 0.0), mask=m_all)


# ------------------------------------------------------------------------------------------------------------------------------------------- backward
@triton.jit
def _bwd_dkdv(Q, K, V, POS, W, IDS, CW, NV, DO, LSE, DELTA, DK, DV_, DW, DCW, qk_scale, scale, H, N, spb,
              sqb, sqh, sqn, skb, skh, skn, svb, svh, svn, sdob, sdoh, sdon, sdkb, sdkh, sdkn, sdvb, sdvh, sdvn,
              DQK: tl.constexpr, QA: tl.constexpr, QB: tl.constexpr, DV: tl.constexpr, VA: tl.constexpr, VB: tl.constexpr,
              DIST: tl.constexpr, CODES: tl.constexpr, GRAD_W: tl.constexpr, GRAD_CW: tl.constexpr, MODE: tl.constexpr,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """one key block: loops over the query blocks; dk, dv and the weight gradients of the block (scores in the [key, query] orientation)"""
    pid_n, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    b64, h64 = b.to(tl.int64), h.to(tl.int64)
    nv = tl.load(NV)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_in = offs_n < nv
    qb, kb, vb, dob = Q + b64 * sqb + h64 * sqh, K + b64 * skb + h64 * skh, V + b64 * svb + h64 * svh, DO + b64 * sdob + h64 * sdoh
    k_a = _load_rows(kb, offs_n, skn, n_in, DQK, 0, QA)
    v_a = _load_rows(vb, offs_n, svn, n_in, DV, 0, VA)
    if QB > 0:
        k_b = _load_rows(kb, offs_n, skn, n_in, DQK, QA, QB)
    if VB > 0:
        v_b = _load_rows(vb, offs_n, svn, n_in, DV, VA, VB)
    w = 0.0
    cwa = 0.0
    cwe = 0.0
    if DIST:
        w = tl.load(W + b64 * H + h64) * 1.4426950408889634
    if CODES:
        cwa = tl.load(CW + 2 * h64) * 1.4426950408889634
        cwe = tl.load(CW + 2 * h64 + 1) * 1.4426950408889634
    dk_a = tl.zeros([BLOCK_N, QA], tl.float32)
    dv_a = tl.zeros([BLOCK_N, VA], tl.float32)
    if QB > 0:
        dk_b = tl.zeros([BLOCK_N, QB], tl.float32)
    if VB > 0:
        dv_b = tl.zeros([BLOCK_N, VB], tl.float32)
    gw = tl.zeros([BLOCK_N], tl.float32)
    ga = tl.zeros([BLOCK_N], tl.float32)
    ge = tl.zeros([BLOCK_N], tl.float32)
    hi = tl.where(pid_n * BLOCK_N < nv, nv, 0)
    for start_m in tl.range(0, hi, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        m_in = offs_m < nv
        qt_a = _load_cols(qb, offs_m, sqn, m_in, DQK, 0, QA)                      # [QA, BM]
        st = _dot(k_a, qt_a, MODE)                                                 # [BN, BM]
        if QB > 0:
            qt_b = _load_cols(qb, offs_m, sqn, m_in, DQK, QA, QB)
            st += _dot(k_b, qt_b, MODE)
        st, d2, sa, se = _bias(st * qk_scale, offs_n, offs_m, n_in, m_in, POS, spb, b64, IDS, N, w, cwa, cwe, DIST, CODES)
        lse = tl.load(LSE + (b64 * H + h64) * N + offs_m, mask=m_in, other=0.0)
        delta = tl.load(DELTA + (b64 * H + h64) * N + offs_m, mask=m_in, other=0.0)
        # p <= 1 exactly (the forward and this kernel may round the scores differently)
        pt = tl.where(n_in[:, None] & m_in[None, :], tl.math.exp2(tl.minimum(st - lse[None, :], 0.0)), 0.0)
        do_a = _load_rows(dob, offs_m, sdon, m_in, DV, 0, VA)                      # [BM, VA]
        dv_a += _dot(pt.to(do_a.dtype), do_a, MODE)
        dpt = _dot(v_a, tl.trans(do_a), MODE)                                      # [BN, BM]
        if VB > 0:
            do_b = _load_rows(dob, offs_m, sdon, m_in, DV, VA, VB)
            dv_b += _dot(pt.to(do_b.dtype), do_b, MODE)
            dpt += _dot(v_b, tl.trans(do_b), MODE)
        dst = pt * (dpt - delta[None, :])                                          # d loss / d s (natural units)
        dk_a += _dot(dst.to(qt_a.dtype), tl.trans(qt_a), MODE)
        if QB > 0:
            dk_b += _dot(dst.to(qt_b.dtype), tl.trans(qt_b), MODE)
        if GRAD_W:
            gw += tl.sum(-dst * d2, 1)
        if GRAD_CW:
            ga += tl.sum(dst * sa, 1)
            ge += tl.sum(dst * se, 1)
    n_all = offs_n < N
    dkb, dvb = DK + b64 * sdkb + h64 * sdkh, DV_ + b64 * sdvb + h64 * sdvh
    _store_rows(dkb, offs_n, sdkn, n_all, (dk_a * scale).to(DK.dtype.element_ty), DQK, 0, QA)
    _store_rows(dvb, offs_n, sdvn, n_all, dv_a.to(DV_.dtype.element_ty), DV, 0, VA)
    if QB > 0:
        _store_rows(dkb, offs_n, sdkn, n_all, (dk_b * scale).to(DK.dtype.element_ty), DQK, QA, QB)
    if VB > 0:
        _store_rows(dvb, offs_n, sdvn, n_all, dv_b.to(DV_.dtype.element_ty), DV, VA, VB)
    if GRAD_W:
        tl.atomic_add(DW + b64 * H + h64, tl.sum(gw, 0), sem="relaxed")
    if GRAD_CW:
        tl.atomic_add(DCW + 2 * h64, tl.sum(ga, 0), sem="relaxed")
        tl.atomic_add(DCW + 2 * h64 + 1, tl.sum(ge, 0), sem="relaxed")


@triton.jit
def _bwd_dq(Q, K, V, POS, W, IDS, CW, NV, DO, LSE, DELTA, DQ, qk_scale, scale, H, N, spb,
            sqb, sqh, sqn, skb, skh, skn, svb, svh, svn, sdob, sdoh, sdon, sdqb, sdqh, sdqn,
            DQK: tl.constexpr, QA: tl.constexpr, QB: tl.constexpr, DV: tl.constexpr, VA: tl.constexpr, VB: tl.constexpr,
            DIST: tl.constexpr, CODES: tl.constexpr, MODE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """one query block: loops over the key blocks; dq"""
    pid_m, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    b64, h64 = b.to(tl.int64), h.to(tl.int64)
    nv = tl.load(NV)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_in = offs_m < nv
    qb, kb, vb, dob = Q + b64 * sqb + h64 * sqh, K + b64 * skb + h64 * skh, V + b64 * svb + h64 * svh, DO + b64 * sdob + h64 * sdoh
    q_a = _load_rows(qb, offs_m, sqn, m_in, DQK, 0, QA)
    do_a = _load_rows(dob, offs_m, sdon, m_in, DV, 0, VA)
    if QB > 0:
        q_b = _load_rows(qb, offs_m, sqn, m_in, DQK, QA, QB)
    if VB > 0:
        do_b = _load_rows(dob, offs_m, sdon, m_in, DV, VA, VB)
    lse = tl.load(LSE + (b64 * H + h64) * N + offs_m, mask=m_in, other=0.0)
    delta = tl.load(DELTA + (b64 * H + h64) * N + offs_m, mask=m_in, other=0.0)
    w = 0.0
    cwa = 0.0
    cwe = 0.0
    if DIST:
        w = tl.load(W + b64 * H + h64) * 1.4426950408889634
    if CODES:
        cwa = tl.load(CW + 2 * h64) * 1.4426950408889634
        cwe = tl.load(CW + 2 * h64 + 1) * 1.4426950408889634
    dq_a = tl.zeros([BLOCK_M, QA], tl.float32)
    if QB > 0:
        dq_b = tl.zeros([BLOCK_M, QB], tl.float32)
    hi = tl.where(pid_m * BLOCK_M < nv, nv, 0)
    for start_n in tl.range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_in = offs_n < nv
        k_a = _load_rows(kb, offs_n, skn, n_in, DQK, 0, QA)                        # [BN, QA]
        s = _dot(q_a, tl.trans(k_a), MODE)
        if QB > 0:
            k_b = _load_rows(kb, offs_n, skn, n_in, DQK, QA, QB)
            s += _dot(q_b, tl.trans(k_b), MODE)
        s, d2, sa, se = _bias(s * qk_scale, offs_m, offs_n, m_in, n_in, POS, spb, b64, IDS, N, w, cwa, cwe, DIST, CODES)
        p = tl.where(m_in[:, None] & n_in[None, :], tl.math.exp2(tl.minimum(s - lse[:, None], 0.0)), 0.0)
        dp = _dot(do_a, _load_cols(vb, offs_n, svn, n_in, DV, 0, VA), MODE)        # [BM, BN]
        if VB > 0:
            dp += _dot(do_b, _load_cols(vb, offs_n, svn, n_in, DV, VA, VB), MODE)
        ds = p * (dp - delta[:, None])
        dq_a += _dot(ds.to(k_a.dtype), k_a, MODE)
        if QB > 0:
            dq_b += _dot(ds.to(k_b.dtype), k_b, MODE)
    m_all = offs_m < N
    dqb = DQ + b64 * sdqb + h64 * sdqh
    _store_rows(dqb, offs_m, sdqn, m_all, (dq_a * scale).to(DQ.dtype.element_ty), DQK, 0, QA)
    if QB > 0:
        _store_rows(dqb, offs_m, sdqn, m_all, (dq_b * scale).to(DQ.dtype.element_ty), DQK, QA, QB)


# ------------------------------------------------------------------------------------------------------------------------------------------- launch
def _config(half: bool, n: int, width: int, kernel: str):
    """(BLOCK_M, BLOCK_N, num_warps, num_stages) of kernel "fwd", "dkdv" or "dq", fixed (see the module docstring)"""
    if kernel == "fwd":
        return (128, 32, 4, 2) if half else (64, 32, 4, 2)
    return (64, 64, 4, 2) if half else (32, 32, 4, 2)


def _mode(dtype: torch.dtype) -> int:
    if dtype == torch.float16:
        return MODES["fp16"]
    if dtype == torch.bfloat16:
        return MODES["bf16"]
    return MODES["tf32x3"] if torch.get_float32_matmul_precision() == "highest" else MODES["tf32"]


def _dummy(device) -> torch.Tensor:
    return torch.zeros(1, device=device, dtype=torch.float32)


def _widths(q, v):
    return (q.shape[-1], *split_width(q.shape[-1]), v.shape[-1], *split_width(v.shape[-1]))


@torch.library.custom_op("pomelo::attention_fwd", mutates_args=())
def _attention_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, n_valid: torch.Tensor, pos: Optional[torch.Tensor], w: Optional[torch.Tensor],
                   ids: Optional[torch.Tensor], code_weight: Optional[torch.Tensor], scale: float, mode: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """-> o [B, H, N, Dv] (contiguous), lse [B, H, N] (log2 units)"""
    B, H, N, _ = q.shape
    dqk, qa, qb, dv, va, vb = _widths(q, v)
    o = torch.empty(B, H, N, dv, device=q.device, dtype=q.dtype)
    lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
    bm, bn, warps, stages = _config(mode >= MODES["bf16"], N, qa + qb, "fwd")
    z = _dummy(q.device)
    _fwd[(triton.cdiv(N, bm), B * H)](
        q, k, v, pos if pos is not None else z, w if w is not None else z, ids if ids is not None else z, code_weight if code_weight is not None else z,
        n_valid, o, lse, scale * LOG2E, H, N, pos.stride(0) if pos is not None else 0,
        *q.stride()[:3], *k.stride()[:3], *v.stride()[:3], *o.stride()[:3],
        DQK=dqk, QA=qa, QB=qb, DV=dv, VA=va, VB=vb, DIST=pos is not None, CODES=ids is not None, MODE=mode,
        BLOCK_M=bm, BLOCK_N=bn, num_warps=warps, num_stages=stages)
    return o, lse


@_attention_fwd.register_fake
def _(q, k, v, n_valid, pos, w, ids, code_weight, scale, mode):
    B, H, N, _ = q.shape
    return q.new_empty(B, H, N, v.shape[-1]), q.new_empty(B, H, N, dtype=torch.float32)


@torch.library.custom_op("pomelo::attention_bwd", mutates_args=())
def _attention_bwd(do: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, lse: torch.Tensor, n_valid: torch.Tensor,
                   pos: Optional[torch.Tensor], w: Optional[torch.Tensor], ids: Optional[torch.Tensor], code_weight: Optional[torch.Tensor],
                   scale: float, mode: int, grad_w: bool, grad_cw: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> dq, dk, dv (the shapes and dtype of q, k, v), dw [B, H], dcode_weight [H, 2] (float32; zeros when not requested)"""
    B, H, N, _ = q.shape
    dqk, qa, qb, dv, va, vb = _widths(q, v)
    do = do if do.stride(-1) == 1 else do.contiguous()
    delta = (o.float() * do.float()).sum(-1).contiguous()                              # [B, H, N]
    dq, dk = torch.empty_like(q), torch.empty_like(k)
    dvv = torch.empty_like(v)
    gw = torch.zeros(B, H, device=q.device, dtype=torch.float32)
    gcw = torch.zeros(H, 2, device=q.device, dtype=torch.float32)
    z = _dummy(q.device)
    common = (pos if pos is not None else z, w if w is not None else z, ids if ids is not None else z, code_weight if code_weight is not None else z, n_valid)
    flags = dict(DQK=dqk, QA=qa, QB=qb, DV=dv, VA=va, VB=vb, DIST=pos is not None, CODES=ids is not None, MODE=mode)
    sp = pos.stride(0) if pos is not None else 0
    half = mode >= MODES["bf16"]
    bm, bn, warps, stages = _config(half, N, qa + qb, "dkdv")
    flags.update(BLOCK_M=bm, BLOCK_N=bn)
    _bwd_dkdv[(triton.cdiv(N, bn), B * H)](
        q, k, v, *common, do, lse, delta, dk, dvv, gw, gcw, scale * LOG2E, scale, H, N, sp,
        *q.stride()[:3], *k.stride()[:3], *v.stride()[:3], *do.stride()[:3], *dk.stride()[:3], *dvv.stride()[:3],
        GRAD_W=grad_w and pos is not None, GRAD_CW=grad_cw and ids is not None, num_warps=warps, num_stages=stages, **flags)
    bm, bn, warps, stages = _config(half, N, qa + qb, "dq")
    flags.update(BLOCK_M=bm, BLOCK_N=bn)
    _bwd_dq[(triton.cdiv(N, bm), B * H)](
        q, k, v, *common, do, lse, delta, dq, scale * LOG2E, scale, H, N, sp,
        *q.stride()[:3], *k.stride()[:3], *v.stride()[:3], *do.stride()[:3], *dq.stride()[:3],
        num_warps=warps, num_stages=stages, **flags)
    return dq, dk, dvv, gw, gcw


@_attention_bwd.register_fake
def _(do, q, k, v, o, lse, n_valid, pos, w, ids, code_weight, scale, mode, grad_w, grad_cw):
    B, H = q.shape[:2]
    return (torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), q.new_empty(B, H, dtype=torch.float32), q.new_empty(H, 2, dtype=torch.float32))


def _setup_context(ctx, inputs, output):
    q, k, v, n_valid, pos, w, ids, code_weight, scale, mode = inputs
    ctx.save_for_backward(q, k, v, output[0], output[1], n_valid, pos, w, ids, code_weight)
    ctx.scale, ctx.mode = scale, mode
    ctx.w_dtype = w.dtype if w is not None else None
    ctx.cw_dtype = code_weight.dtype if code_weight is not None else None


def _backward(ctx, do, _dlse):
    q, k, v, o, lse, n_valid, pos, w, ids, code_weight = ctx.saved_tensors
    grad_w, grad_cw = ctx.needs_input_grad[5], ctx.needs_input_grad[7]
    dq, dk, dv, gw, gcw = _attention_bwd(do, q, k, v, o, lse, n_valid, pos, w, ids, code_weight, ctx.scale, ctx.mode, grad_w, grad_cw)
    return dq, dk, dv, None, None, gw if grad_w else None, None, gcw if grad_cw else None, None, None


_attention_fwd.register_autograd(_backward, setup_context=_setup_context)


def pomelo_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, n_valid: torch.Tensor, scale: float, pos: Optional[torch.Tensor] = None,
                     w: Optional[torch.Tensor] = None, ids: Optional[torch.Tensor] = None, code_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """the attention of the module docstring.

    Args:
        q, k: [B, H, N, Dqk], v: [B, H, N, Dv]: float32, bfloat16 or float16 of one dtype (the last dimension contiguous)
        n_valid: int32 tensor of one element: the real positions are 0 .. n_valid - 1
        scale: the factor of q . k
        pos: [B, N, 3] positions of the distance term (no gradient) and w [B, H] its weight (>= 0); None: no distance term
        ids: [N, 2] integer chain and entity ids and code_weight [H, 2] the weights of [same chain], [same entity]; None: no such terms

    Returns:
        [B, H, N, Dv] (0 at the rows from n_valid on)
    """
    B, H, N, dqk = q.shape
    if k.shape != q.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError(f"q, k must be [B, H, N, D] and v [B, H, N, Dv] of the same B, H, N: got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}")
    if q.dtype not in (torch.float32, torch.bfloat16, torch.float16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError(f"q, k, v must be float32, bfloat16 or float16 of one dtype, got {q.dtype}, {k.dtype}, {v.dtype}")
    if max(dqk, v.shape[-1]) > 256:
        raise ValueError("head widths up to 256")
    q, k, v = (x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v))
    if pos is not None:
        if pos.shape != (B, N, 3) or w is None or w.shape != (B, H):
            raise ValueError(f"the distance term needs pos [B, N, 3] = {(B, N, 3)} (got {tuple(pos.shape)}) and w [B, H]")
        pos = pos.detach().float().permute(0, 2, 1).contiguous()                        # [B, 3, N]: one coalesced load per axis
        w = w.float().contiguous()
    if ids is not None:
        if ids.shape != (N, 2) or code_weight is None or code_weight.shape != (H, 2):
            raise ValueError(f"the chain / entity terms need ids [N, 2] (got {tuple(ids.shape)}) and code_weight [H, 2]")
        ids = ids.detach().to(torch.int32).T.contiguous()                               # [2, N]
        code_weight = code_weight.float().contiguous()
    n_valid = n_valid.to(device=q.device, dtype=torch.int32).reshape(1)
    return _attention_fwd(q, k, v, n_valid, pos, w, ids, code_weight, float(scale), _mode(q.dtype))[0]
