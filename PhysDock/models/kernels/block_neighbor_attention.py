"""Block-neighbour attention: every atom attends only to the atoms that are close to it in the *current* (noisy) coordinates, decided per
sample and per denoising step, and still computed with dense tensor-core tiles.

    layout   = build_block_layout(pos, valid, leaf, blocks)       # per sample, from the (noisy) coordinates, no gradient
    o_sorted = block_neighbor_attention(q, k, v, bias, layout)    # q, k, v in the sorted order, one attention per query block

1. Sort. The atoms of every sample are ordered by a k-d tree (recursive bisection along the longest axis, split at a multiple of `leaf`), so
   that every run of `leaf` consecutive positions is a compact cluster of atoms. layout.perm[b, i] is the atom at sorted position i of sample
   b (positions >= N are virtual padding atoms, far away and invalid). The model sorts the atom stream once per DiT stack
   (sort_atoms / unsort_atoms): everything except the attention (LayerNorm, linears, feed-forward) acts on every atom alone.
2. Neighbour blocks. The `blocks` nearest blocks of every block (distance of the centroids of the valid atoms, the block itself first) are
   layout.kb[b, t, :]. Every atom of block t attends to all atoms of those blocks: blocks * leaf keys per query, the same for every query.
3. Attention. The kernels visit only those key blocks: forward and backward are the dense flash-attention loops over `blocks` tiles of
   leaf x leaf, with a bias tile that is gathered from the pair bias in the original atom order (bias[h, perm_q, perm_k], shared by all
   samples; -1e9 for invalid atoms) and a gradient of that bias that is scattered back with atomics.

Cost is blocks * leaf / N of dense attention (+ ~2 ms for the sort of 48 samples x 3072 atoms). The function is attention restricted to the
union of the neighbour blocks, not an exact per-atom top-k: with 32-atom blocks and 8 blocks (256 keys) 92 / 88 / 81% of the true 32 / 64 / 128
nearest atoms of a crop are inside the keys. The coordinates only choose the order and the blocks; they receive no gradient.

Precision (same as shared_bias_attention): float32 inputs use TF32 matmuls with the inputs rounded to nearest, bfloat16 inputs plain bf16 matmuls;
softmax and accumulators in float32. torch_block_attention is the same function in plain torch (also the torch backend of the model).

The launch configuration is fixed (no autotuning pass), swept on an H20 (sm_90).
"""
from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from ._common import round_tf32_ as _round_tf32


class BlockLayout(NamedTuple):
    """perm [B, Np] int32: the atom at every sorted position (>= N: virtual padding), Np = ceil(N / leaf) * leaf;
    kb [B, T, M] int32: the M key blocks of every query block (T = Np / leaf), the block itself first"""
    perm: torch.Tensor
    kb: torch.Tensor


# ------------------------------------------------------------------------------------------------------------------ layout
def kd_order(x: torch.Tensor, leaf: int) -> torch.Tensor:
    """x [B, Np, 3] (Np a multiple of leaf) -> order [B, Np]: sorted position -> index, such that every run of `leaf` positions is compact
    (recursive bisection along the longest axis; the segments of a level have at most two lengths, so a level is a few batched sorts)"""
    B, Np, _ = x.shape
    order = torch.arange(Np, device=x.device).expand(B, Np).contiguous()
    segs = [(0, Np // leaf)]                                                  # (first block, number of blocks)
    while any(n > 1 for _, n in segs):
        nxt, by_len = [], {}
        for a, n in segs:
            if n == 1:
                nxt.append((a, n))
            else:
                by_len.setdefault(n, []).append(a)
        for n, starts in by_len.items():
            idx = torch.stack([order[:, a * leaf:(a + n) * leaf] for a in starts], 1)                         # [B, S, n * leaf]
            pts = torch.gather(x.unsqueeze(1).expand(-1, len(starts), -1, -1), 2, idx.unsqueeze(-1).expand(-1, -1, -1, 3))
            axis = (pts.amax(2) - pts.amin(2)).argmax(-1)                                                      # [B, S] longest axis
            coord = torch.gather(pts, 3, axis[:, :, None, None].expand(-1, -1, pts.shape[2], 1)).squeeze(-1)  # [B, S, n * leaf]
            idx = torch.gather(idx, 2, torch.argsort(coord, dim=-1))
            for si, a in enumerate(starts):
                order[:, a * leaf:(a + n) * leaf] = idx[:, si]
            half = n // 2
            for a in starts:
                nxt += [(a, half), (a + half, n - half)]
        segs = nxt
    return order


@torch.no_grad()
def build_block_layout(pos: torch.Tensor, valid: Optional[torch.Tensor], leaf: int, blocks: int) -> BlockLayout:
    """Layout of the block-neighbour attention of pos [B, N, 3] (the noisy coordinates; any scale); valid [N] or [B, N] bool marks the real atoms
    (None: all). Differentiation does not pass through the coordinates."""
    B, N, _ = pos.shape
    T = -(-N // leaf)
    Np = T * leaf
    pos = pos.detach().float()
    valid = torch.ones(B, N, dtype=torch.bool, device=pos.device) if valid is None else valid.bool().expand(B, N)
    pos = F.pad(pos, (0, 0, 0, Np - N))
    valid = F.pad(valid, (0, Np - N), value=False)
    # invalid atoms: far away, in index order, so that they end up in their own blocks (at most one block mixes real and invalid atoms)
    real = pos.masked_fill(~valid[..., None], 0.0)
    hi = torch.where(valid[..., None], pos, torch.full_like(pos, -1e30)).amax(1, keepdim=True).clamp_min(0.0)
    lo = torch.where(valid[..., None], pos, torch.full_like(pos, 1e30)).amin(1, keepdim=True).clamp_max(0.0)
    span = (hi - lo).amax(-1, keepdim=True).clamp_min(1e-3)
    far = hi + 10.0 * span + span * torch.arange(Np, device=pos.device, dtype=pos.dtype)[None, :, None] * 1e-3 / Np
    x = torch.where(valid[..., None], real, far)
    order = kd_order(x, leaf)
    xs = torch.gather(x, 1, order[..., None].expand(-1, -1, 3))
    vs = torch.gather(valid, 1, order)
    cnt = vs.view(B, T, leaf).sum(-1)
    cen = (xs * vs[..., None]).view(B, T, leaf, 3).sum(2) / cnt.clamp_min(1)[..., None]
    cen = torch.where((cnt > 0)[..., None], cen, far.amax(1, keepdim=True).expand(-1, T, -1) * 4.0)
    d = torch.cdist(cen, cen)
    d.diagonal(dim1=1, dim2=2).fill_(-1.0)                                                                     # the block itself first
    kb = d.topk(min(blocks, T), dim=-1, largest=False).indices
    return BlockLayout(order.to(torch.int32).contiguous(), kb.to(torch.int32).contiguous())


def sort_atoms(x: torch.Tensor, layout: BlockLayout) -> torch.Tensor:
    """x [B, N, C] (original atom order) -> [B, Np, C] sorted (virtual padding atoms are zeros)"""
    Np = layout.perm.shape[1]
    x = F.pad(x, (0, 0, 0, Np - x.shape[1]))
    return torch.gather(x, 1, layout.perm.long()[..., None].expand(-1, -1, x.shape[-1]))


def unsort_atoms(xs: torch.Tensor, layout: BlockLayout, n: int) -> torch.Tensor:
    """sorted [B, Np, C] -> [B, n, C] in the original order"""
    inv = torch.argsort(layout.perm.long(), dim=1)
    return torch.gather(xs, 1, inv[..., None].expand(-1, -1, xs.shape[-1]))[:, :n]


# ------------------------------------------------------------------------------------------------------------------ torch version
def torch_block_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, layout: BlockLayout) -> torch.Tensor:
    """The same function in plain torch (differentiable; gathers the key blocks of every query block). q, k, v: [B, H, Np, D] sorted,
    bias [H, N, N] original order -> [B, H, Np, D]"""
    B, H, Np, D = q.shape
    N = bias.shape[-1]
    T, M = layout.kb.shape[1:]
    leaf = Np // T
    perm = layout.perm.long()
    flat = layout.kb.long() + (torch.arange(B, device=q.device) * T)[:, None, None]                          # [B, T, M] into B * T blocks

    def blocks_of(t):                                                                                         # [B, H, Np, D] -> [B, H, T, M * leaf, D]
        tb = t.view(B, H, T, leaf, D).permute(0, 2, 1, 3, 4).reshape(B * T, H, leaf, D)
        return tb[flat].permute(0, 1, 3, 2, 4, 5).reshape(B, T, H, M * leaf, D).permute(0, 2, 1, 3, 4)

    kg, vg = blocks_of(k), blocks_of(v)
    pq = perm.view(B, T, leaf)
    pk = perm.view(B * T, leaf)[flat].reshape(B, T, M * leaf)
    ok = (pq < N)[..., :, None] & (pk < N)[..., None, :]                                                      # [B, T, leaf, M * leaf]
    b = bias[:, pq.clamp(max=N - 1)[..., :, None], pk.clamp(max=N - 1)[..., None, :]]                         # [H, B, T, leaf, M * leaf]
    b = b.permute(1, 0, 2, 3, 4).masked_fill(~ok[:, None], -1e9)
    qb = q.view(B, H, T, leaf, D)
    s = torch.einsum("bhtid,bhtjd->bhtij", qb.float(), kg.float()) * D ** -0.5 + b.float()
    return torch.einsum("bhtij,bhtjd->bhtid", torch.softmax(s, -1), vg.float()).reshape(B, H, Np, D).to(q.dtype)


# ------------------------------------------------------------------------------------------------------------------ kernels
@triton.jit
def _fwd(Q, K, V, PERM, BIAS, KB, O, LSE, N, NP, H, T, scale_log2e,
         sqb, sqh, sqn, skb, skh, skn, svb, svh, svn, sob, soh, son, sbh, sbm,
         M: tl.constexpr, D: tl.constexpr, LEAF: tl.constexpr, PREC: tl.constexpr, RN: tl.constexpr):
    bh = tl.program_id(0)
    pm = tl.program_id(1)
    b = (bh // H).to(tl.int64)
    h = (bh % H).to(tl.int64)
    om = pm * LEAF + tl.arange(0, LEAF)
    od = tl.arange(0, D)
    pq = tl.load(PERM + b * NP + om)
    q = tl.load(Q + b * sqb + h * sqh + om[:, None] * sqn + od[None, :])
    q = (q * scale_log2e).to(q.dtype)
    if RN:
        q = _round_tf32(q)
    m_i = tl.full([LEAF], -float("inf"), tl.float32)
    l_i = tl.zeros([LEAF], tl.float32)
    acc = tl.zeros([LEAF, D], tl.float32)
    for it in tl.static_range(M):
        on = tl.load(KB + (b * T + pm) * M + it) * LEAF + tl.arange(0, LEAF)
        pk = tl.load(PERM + b * NP + on)
        kt = tl.load(K + b * skb + h * skh + on[None, :] * skn + od[:, None])
        bias = tl.load(BIAS + h * sbh + pq[:, None] * sbm + pk[None, :], mask=(pq < N)[:, None] & (pk < N)[None, :], other=-1e9).to(tl.float32)
        if RN:
            kt = _round_tf32(kt)
        s = tl.dot(q, kt, input_precision=PREC) + bias * 1.4426950408889634
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + b * svb + h * svh + on[:, None] * svn + od[None, :])
        pp = p.to(v.dtype)
        if RN:
            pp = _round_tf32(pp)
            v = _round_tf32(v)
        acc = acc * alpha[:, None] + tl.dot(pp, v, input_precision=PREC)
        m_i = m_new
    tl.store(O + b * sob + h * soh + om[:, None] * son + od[None, :], acc / l_i[:, None])
    tl.store(LSE + (b * H + h) * NP + om, m_i + tl.math.log2(l_i))


@triton.jit
def _bwd(Q, K, V, PERM, BIAS, KB, DO, LSE, DELTA, DQ, DK, DV, DBIAS, N, NP, H, T, scale, scale_log2e,
         sqb, sqh, sqn, skb, skh, skn, svb, svh, svn, sdob, sdoh, sdon, sdqb, sdqh, sdqn, sdkb, sdkh, sdkn, sdvb, sdvh, sdvn, sbh, sbm, sdbh, sdbm,
         M: tl.constexpr, D: tl.constexpr, LEAF: tl.constexpr, PREC: tl.constexpr, RN: tl.constexpr):
    bh = tl.program_id(0)
    pm = tl.program_id(1)
    b = (bh // H).to(tl.int64)
    h = (bh % H).to(tl.int64)
    om = pm * LEAF + tl.arange(0, LEAF)
    od = tl.arange(0, D)
    pq = tl.load(PERM + b * NP + om)
    q = tl.load(Q + b * sqb + h * sqh + om[:, None] * sqn + od[None, :])
    q = (q * scale_log2e).to(q.dtype)
    do = tl.load(DO + b * sdob + h * sdoh + om[:, None] * sdon + od[None, :])
    if RN:
        q, do = _round_tf32(q), _round_tf32(do)
    lse = tl.load(LSE + (b * H + h) * NP + om)
    delta = tl.load(DELTA + (b * H + h) * NP + om)
    dq = tl.zeros([LEAF, D], tl.float32)
    for it in tl.static_range(M):
        on = tl.load(KB + (b * T + pm) * M + it) * LEAF + tl.arange(0, LEAF)
        pk = tl.load(PERM + b * NP + on)
        kt = tl.load(K + b * skb + h * skh + on[None, :] * skn + od[:, None])
        vt = tl.load(V + b * svb + h * svh + on[None, :] * svn + od[:, None])
        inb = (pq < N)[:, None] & (pk < N)[None, :]
        bias = tl.load(BIAS + h * sbh + pq[:, None] * sbm + pk[None, :], mask=inb, other=-1e9).to(tl.float32)
        if RN:
            kt, vt = _round_tf32(kt), _round_tf32(vt)
        s = tl.dot(q, kt, input_precision=PREC) + bias * 1.4426950408889634
        p = tl.math.exp2(s - lse[:, None])
        dp = tl.dot(do, vt, input_precision=PREC).to(tl.float32)
        ds = p * (dp - delta[:, None])
        pc, dsc = p.to(do.dtype), ds.to(do.dtype)
        if RN:
            pc, dsc = _round_tf32(pc), _round_tf32(dsc)
        dq += tl.dot(dsc, tl.trans(kt), input_precision=PREC)
        dv = tl.dot(tl.trans(pc), do, input_precision=PREC)
        dk = tl.dot(tl.trans(dsc), q, input_precision=PREC) * 0.6931471805599453
        tl.atomic_add(DV + b * sdvb + h * sdvh + on[:, None] * sdvn + od[None, :], dv, sem="relaxed")
        tl.atomic_add(DK + b * sdkb + h * sdkh + on[:, None] * sdkn + od[None, :], dk, sem="relaxed")
        tl.atomic_add(DBIAS + h * sdbh + pq[:, None] * sdbm + pk[None, :], ds, mask=inb, sem="relaxed")
    tl.store(DQ + b * sdqb + h * sdqh + om[:, None] * sdqn + od[None, :], dq * scale)


# ------------------------------------------------------------------------------------------------------------------ torch ops
_CONFIG = {32: (2, 2), 64: (4, 2)}              # leaf -> (warps, stages), swept on an H20 (B = 48, N = 3072, 128-512 keys per query)


def _prep(dtype):
    return ("ieee", False) if dtype == torch.bfloat16 else ("tf32", True)


@torch.library.custom_op("physdock::block_neighbor_attention_fwd", mutates_args=())
def _attention_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, perm: torch.Tensor,
                   kb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """q, k, v [B, H, Np, D] -> o [B, H, Np, D], lse [B, H, Np] (log2 units)"""
    B, H, Np, D = q.shape
    T, M = kb.shape[1:]
    leaf = Np // T
    prec, rn = _prep(q.dtype)
    o = torch.empty(q.shape, device=q.device, dtype=q.dtype)               # contiguous, as the fake implementation says (not empty_like(q))
    lse = torch.empty(B, H, Np, device=q.device, dtype=torch.float32)
    warps, stages = _CONFIG.get(leaf, (4, 2))
    _fwd[(B * H, T)](q, k, v, perm, bias, kb, o, lse, bias.shape[-1], Np, H, T, D ** -0.5 * 1.4426950408889634,
                     *q.stride()[:3], *k.stride()[:3], *v.stride()[:3], *o.stride()[:3], bias.stride(0), bias.stride(1),
                     M=M, D=D, LEAF=leaf, PREC=prec, RN=rn, num_warps=warps, num_stages=stages)
    return o, lse


@_attention_fwd.register_fake
def _(q, k, v, bias, perm, kb):
    return q.new_empty(q.shape), q.new_empty(q.shape[:3], dtype=torch.float32)


@torch.library.custom_op("physdock::block_neighbor_attention_bwd", mutates_args=())
def _attention_bwd(do: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, perm: torch.Tensor, kb: torch.Tensor,
                   o: torch.Tensor, lse: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> dq, dk, dv [B, H, Np, D] (dtype of q), dbias [H, N, N] float32"""
    B, H, Np, D = q.shape
    T, M = kb.shape[1:]
    leaf = Np // T
    prec, rn = _prep(q.dtype)
    do = do if do.stride(-1) == 1 else do.contiguous()
    # delta = sum(o * do) with do rounded like the dO V^T matmul sees it (the softmax backward subtracts delta from dO V^T)
    delta = (o.float() * (((do.float().contiguous().view(torch.int32) + 0x1000) & -8192).view(torch.float32) if rn else do.float())).sum(-1)
    dq = torch.empty(B, H, Np, D, device=q.device, dtype=torch.float32)
    dk = torch.zeros_like(dq)
    dv = torch.zeros_like(dq)
    N = bias.shape[-1]
    dbias = torch.zeros(H, N, N, device=q.device, dtype=torch.float32)
    warps, stages = _CONFIG.get(leaf, (4, 2))
    _bwd[(B * H, T)](q, k, v, perm, bias, kb, do, lse, delta, dq, dk, dv, dbias, N, Np, H, T, D ** -0.5, D ** -0.5 * 1.4426950408889634,
                     *q.stride()[:3], *k.stride()[:3], *v.stride()[:3], *do.stride()[:3], *dq.stride()[:3], *dk.stride()[:3], *dv.stride()[:3],
                     bias.stride(0), bias.stride(1), dbias.stride(0), dbias.stride(1),
                     M=M, D=D, LEAF=leaf, PREC=prec, RN=rn, num_warps=warps, num_stages=stages)
    return dq.to(q.dtype).contiguous(), dk.to(q.dtype).contiguous(), dv.to(q.dtype).contiguous(), dbias


@_attention_bwd.register_fake
def _(do, q, k, v, bias, perm, kb, o, lse):
    return q.new_empty(q.shape), q.new_empty(q.shape), q.new_empty(q.shape), bias.new_empty(bias.shape, dtype=torch.float32)


def _setup_context(ctx, inputs, output):
    q, k, v, bias, perm, kb = inputs
    ctx.save_for_backward(q, k, v, bias, perm, kb, output[0], output[1])


def _backward(ctx, do, dlse):
    q, k, v, bias, perm, kb, o, lse = ctx.saved_tensors
    dq, dk, dv, dbias = _attention_bwd(do, q, k, v, bias, perm, kb, o, lse)
    return dq, dk, dv, dbias.to(bias.dtype), None, None


_attention_fwd.register_autograd(_backward, setup_context=_setup_context)


def block_neighbor_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, layout: BlockLayout) -> torch.Tensor:
    """softmax(q k^T / sqrt(D) + bias) v restricted to the key blocks of layout.kb (see the module docstring)

    Args:
        q, k, v: [B, H, Np, D] float32 or bfloat16, in the sorted order of the layout (sort_atoms); D a power of two <= 128
        bias: [H, N, N] logit bias in the original atom order, shared by the B samples (any row stride, unit column stride)
        layout: build_block_layout of the same coordinates

    Returns:
        [B, H, Np, D] in the sorted order
    """
    B, H, Np, D = q.shape
    if k.shape != q.shape or v.shape != q.shape or q.dtype not in (torch.float32, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q, k, v must have the same shape [B, H, Np, D] and a common dtype (float32 or bfloat16)")
    if layout.perm.shape != (B, Np) or layout.kb.shape[0] != B or Np % layout.kb.shape[1]:
        raise ValueError(f"layout {tuple(layout.perm.shape)} / {tuple(layout.kb.shape)} does not fit q {tuple(q.shape)}")
    if D & (D - 1) or D > 128 or bias.dim() != 3 or bias.shape[0] != H or bias.stride(-1) != 1:
        raise ValueError("D must be a power of two <= 128 and bias [H, N, N] with unit column stride")
    return _attention_fwd(q, k, v, bias, layout.perm, layout.kb)[0]
