"""The typed attention of Pomelo (torch only, parity-equivariant by construction), for a batch of noisy samples [B, N, ...] of one structure, without a pair representation.
Every attention is one F.scaled_dot_product_attention call; its only mask is boolean (no gradient: the padding); what has a learned weight is in q and k:

    logit_ij = q_i . k_j / sqrt(D)                q . k over all kinds (0e.0e, 0o.0o, 1o.1o, 1e.1e: even); RoPE (residue index, chain) on the 0e channels
             - w_h(t) |x_i - x_j|^2               the noisy sample (w >= 0 per head and noise level): q gets [x_i, 1], k gets [2 w x_j, -w |x_j|^2] (the term -w |x_i|^2 is
                                                  a constant of the query: the softmax drops it)

Every term is even, so the weights are even and every kind of value keeps its kind. The weights also aggregate the positions: the direction message sum_j alpha_ij (x_j - x_i)
(a polar vector) comes from the same call. The reference conformers and the template are not here: they are vectors of the streams (rotated into the frame of the sample,
PhysDock.pomelo.primitives.alignment).

GlobalAttention: all pairs (the atoms, the tokens), the mask [N, N] (the padding keys; the diagonal kept) shared by the samples and heads.
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from PhysDock.pomelo.primitives import geometry as G
from PhysDock.pomelo.relations import code_columns
from PhysDock.pomelo.primitives.irreps import Feats, Mult, TypedLinear

KINDS = G.KINDS
BACKENDS = {"efficient": [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],          # the memory-efficient kernel (fp32 on the CUDA cores)
            "math": [SDPBackend.MATH]}                                                 # matmul + softmax + matmul (TF32 tensor cores)
# "triton": PhysDock.pomelo.kernels.pomelo_attention, the distance and the chain / entity terms computed in the kernel (no extra columns of q and k)


def split_heads(x: torch.Tensor, kind: str, heads: int) -> torch.Tensor:
    """[..., C] -> [..., H, C/H], [..., C, 3] -> [..., H, C/H, 3]"""
    return x.reshape(*x.shape[:-1], heads, -1) if kind in G.SCALAR_KINDS else x.reshape(*x.shape[:-2], heads, -1, 3)


def merge_heads(x: torch.Tensor, kind: str) -> torch.Tensor:
    return x.reshape(*x.shape[:-2], -1) if kind in G.SCALAR_KINDS else x.reshape(*x.shape[:-3], -1, 3)


def head_norm(h: Dict[str, torch.Tensor], weight: nn.ParameterDict, eps: float = 1e-5) -> Dict[str, torch.Tensor]:
    """the query / key norm (PhysDock's RMSNorm of q and k per head): in every head the scalars (0e, 0o) divided by their RMS, the vectors (1o, 1e) by the RMS of their
    components, apart, then a learned scale per channel (the three components of a vector share it). h {kind: [..., H, d(, 3)]}"""
    out = {}
    for kinds in (G.SCALAR_KINDS, G.VECTOR_KINDS):
        part = {k: h[k] for k in kinds if k in h}
        if not part:
            continue
        ss = sum(v.square().sum(-1) if k in G.SCALAR_KINDS else v.square().sum((-1, -2)) for k, v in part.items())
        n = sum(v.shape[-1] if k in G.SCALAR_KINDS else 3 * v.shape[-2] for k, v in part.items())
        r = torch.rsqrt(ss / n + eps)                                                               # [..., H]
        out.update({k: v * (r[..., None] if k in G.SCALAR_KINDS else r[..., None, None]) * (weight[k] if k in G.SCALAR_KINDS else weight[k][:, None]) for k, v in part.items()})
    return out


def rope_angles(index: torch.Tensor, chain: torch.Tensor, n_res: int, n_chain: int, dtype) -> torch.Tensor:
    """the RoPE angles of every row [N, n_res + n_chain]: residue index on n_res frequencies (base 1000), chain index on n_chain frequencies (base 10)"""
    f_res = 1000.0 ** (-torch.arange(n_res, dtype=dtype, device=index.device) / max(1, n_res))
    f_chain = 10.0 ** (-torch.arange(n_chain, dtype=dtype, device=index.device) / max(1, n_chain))
    return torch.cat([index.to(dtype)[:, None] * f_res, chain.to(dtype)[:, None] * f_chain], -1)


def apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """x [B, N, H, d] (the 0e channels of q or k), angles [N, F] with 2F <= d: the first 2F channels rotated pairwise; a linear map of the 0e channels with even
    coefficients (the kind is kept), and q . k depends on the differences of the positions"""
    f = angles.shape[-1]
    if f == 0:
        return x
    c, s = torch.cos(angles)[None, :, None], torch.sin(angles)[None, :, None]
    a, b = x[..., 0:2 * f:2], x[..., 1:2 * f:2]
    rot = torch.stack([a * c - b * s, a * s + b * c], -1).flatten(-2)
    return torch.cat([rot, x[..., 2 * f:]], -1)


def pad8(x: torch.Tensor) -> torch.Tensor:
    """zero columns up to a multiple of 8 (the memory-efficient kernel wants the head width divisible by 4)"""
    p = (-x.shape[-1]) % 8
    return torch.cat([x, x.new_zeros(*x.shape[:-1], p)], -1) if p else x


def distance_columns(pq: torch.Tensor, pk: torch.Tensor, w: torch.Tensor, scale: float):
    """the columns of q and k for - w |p_i - p_j|^2 in a call with the given scale: q [pq, 1], k [2 w pk, - w |pk|^2] / scale (w broadcast to pk)"""
    return torch.cat([pq, torch.ones_like(pq[..., :1])], -1), torch.cat([2 * w * pk, -w * pk.square().sum(-1, keepdim=True)], -1) / scale


class _TypedAttention(nn.Module):
    """the shared part: typed q, k, v (one matmul per kind), the q / k norms, RoPE, the distance columns, the output gates and projection"""

    def __init__(self, mult: Mult, heads: int, rope_res: int, rope_chain: int, direction_length: float, zero_init: bool = False, chain_codes: int = 0) -> None:
        super().__init__()
        if set(mult) != set(KINDS) or any(c % heads for c in mult.values()):
            raise ValueError(f"the stream needs the four kinds with channels {dict(mult)} divisible by the heads {heads}")
        if 2 * (rope_res + rope_chain) > mult["0e"] // heads:
            raise ValueError("RoPE needs 2 * (rope_res + rope_chain) <= the 0e channels of a head")
        self.mult, self.heads, self.direction_length = dict(mult), heads, direction_length
        self.rope_res, self.rope_chain = rope_res, rope_chain
        self.backend = "math"                         # matmul + softmax + matmul on the TF32 tensor cores: the fastest here (the memory-efficient kernel: fp32 on CUDA cores)
        self.qkv = TypedLinear(self.mult, {k: 3 * c for k, c in self.mult.items()})
        self.q_norm = nn.ParameterDict({k: nn.Parameter(torch.ones(c // heads)) for k, c in self.mult.items()})
        self.k_norm = nn.ParameterDict({k: nn.Parameter(torch.ones(c // heads)) for k, c in self.mult.items()})
        self.dim = sum(c // heads * (1 if k in G.SCALAR_KINDS else 3) for k, c in self.mult.items())
        self.scale = self.dim ** -0.5
        self.mult_cat = {k: c + (heads if k == "1o" else 0) for k, c in self.mult.items()}          # + the direction message of every head
        self.gate = nn.Linear(self.mult["0e"], sum(self.mult_cat.values()))
        self.out = TypedLinear(self.mult_cat, self.mult, zero_init=zero_init)
        # w_h [same chain], w'_h [same entity] in the logits (PhysDock.pomelo.relations: codes in q and k); 0 at the start
        self.code_weight = nn.Parameter(torch.zeros(heads, 2)) if chain_codes else None

    def heads_qkv(self, x: Feats, angles: torch.Tensor):
        """-> q, k [B, H, N, D], v [B, H, N, Dv]: the kinds flattened (0e first), after the q / k norms and the RoPE"""
        H = self.heads
        qkv = self.qkv(x)
        part = lambda i: {kind: split_heads(t.chunk(3, -1 if kind in G.SCALAR_KINDS else -2)[i], kind, H) for kind, t in qkv.items()}
        q, k, v = head_norm(part(0), self.q_norm), head_norm(part(1), self.k_norm), part(2)
        q["0e"], k["0e"] = apply_rope(q["0e"], angles), apply_rope(k["0e"], angles)
        flat = lambda h: torch.cat([h[kd] if kd in G.SCALAR_KINDS else h[kd].flatten(-2) for kd in KINDS], -1).transpose(1, 2)
        return flat(q), flat(k), flat(v)

    def project(self, x: Feats, pos: torch.Tensor, w: torch.Tensor, angles: torch.Tensor, codes: torch.Tensor = None):
        """-> q, k [B, H, N, D + 4 (+ 2 code_dim)] (the kinds flattened, 0e first, then the distance columns, then the chain / entity codes), v [B, H, N, Dv + 3] (then
        the positions)"""
        H = self.heads
        b, n = pos.shape[:2]
        q, k, v = self.heads_qkv(x, angles)
        p = pos[:, None].expand(b, H, n, 3)
        qd, kd = distance_columns(p, p, w[:, :, None, None], self.scale)
        q, k = torch.cat([q, qd], -1), torch.cat([k, kd], -1)
        if self.code_weight is not None:
            qc, kc = code_columns(codes, self.code_weight, self.scale, b, q.dtype)
            q, k = torch.cat([q, qc], -1), torch.cat([k, kc], -1)
        return q, k, torch.cat([v, p.to(v.dtype)], -1)

    def sdpa(self, q, k, v, mask) -> torch.Tensor:
        dv = v.shape[-1]
        with sdpa_kernel(BACKENDS[self.backend]):
            return F.scaled_dot_product_attention(pad8(q), pad8(k), pad8(v), attn_mask=mask, scale=self.scale)[..., :dv]

    def finish(self, x: Feats, o: torch.Tensor, pos: torch.Tensor) -> Feats:
        """o [B, N, H, Dv + 3] (the values, then sum_j alpha_ij x_j) -> the update"""
        H = self.heads
        parts, i = {}, 0
        for kind in KINDS:
            d = self.mult[kind] // H
            width = d if kind in G.SCALAR_KINDS else 3 * d
            parts[kind] = merge_heads(o[..., i:i + width] if kind in G.SCALAR_KINDS else o[..., i:i + width].unflatten(-1, (d, 3)), kind)
            i += width
        direction = o[..., -3:] - pos[:, :, None]                                                       # [B, N, H, 3]: sum_j alpha_ij (x_j - x_i), polar
        parts["1o"] = torch.cat([parts["1o"], G.squash(direction, self.direction_length)], -2)
        gates = torch.sigmoid(self.gate(x["0e"])).split(list(self.mult_cat.values()), dim=-1)
        return self.out({kind: G.scale(parts[kind], kind, g, "0e")[0] for kind, g in zip(self.mult_cat, gates)})


class GlobalAttention(_TypedAttention):
    """all pairs (the atoms, the tokens)"""

    def forward(self, x: Feats, pos: torch.Tensor, w: torch.Tensor, geo: Dict[str, torch.Tensor]) -> Feats:
        """x {kind: [B, N, ...]} (normalised), pos [B, N, 3] (centred), w [B, H] >= 0, geo: angles [N, F], mask [N, N] bool (torch backends) or n_valid [1] and
        ids [N, 2] (triton) -> the update"""
        if self.backend == "triton":
            from PhysDock.pomelo.kernels import pomelo_attention
            q, k, v = self.heads_qkv(x, geo["angles"])
            b, n = pos.shape[:2]
            v = torch.cat([v, pos[:, None].expand(b, self.heads, n, 3).to(v.dtype)], -1)
            codes = self.code_weight is not None
            o = pomelo_attention(q, k, v, geo["n_valid"], self.scale, pos, w, geo["ids"] if codes else None, self.code_weight if codes else None)
            return self.finish(x, o.transpose(1, 2), pos)
        q, k, v = self.project(x, pos, w, geo["angles"], geo.get("codes"))
        return self.finish(x, self.sdpa(q, k, v, geo["mask"]).transpose(1, 2), pos)
