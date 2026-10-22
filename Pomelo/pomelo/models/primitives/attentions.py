"""Typed attentions, torch only, parity-equivariant by construction: TypedAttention (the single representation, with messages from the pair representation) and
TypedTriangleAttention (the pair representation, around the starting or the ending node).

TypedAttention:

One softmax per head, from the logits  q . k / sqrt(D) + b_h(z_0e) + <q_0o, W z_0o>  where q, k are the typed projections of s flattened and concatenated over
the kinds, so q . k is the sum of the dot products within each kind (0e.0e, 0o.0o, 1o.1o, 1e.1e: all even numbers). The weights alpha (even) aggregate three groups
of values, so that every kind of s can receive every kind of z:

    group  weights                         values of kind k                       what it sends
    W0     alpha                           v_k (of the neighbour j)               the neighbours
    We     alpha * sigmoid(z_0e)           v_k                                    the neighbours, gated by the even pair state
    Wo     alpha * tanh(z_0o)              v_flip(k)  (0e <-> 0o, 1o <-> 1e)      the neighbours, gated by the odd pair state: a gate that is a
                                                                                   pseudoscalar changes the parity of what it multiplies
    plus the direction messages  sum_j beta_ij (c_j - c_i)  with beta = alpha * sigmoid(z_0e): a polar vector (1o); with beta = alpha * tanh(z_0o): an
    axial vector (1e). The relative positions are never formed: sum_j beta_ij c_j - c_i sum_j beta_ij  (the centres c_j enter as an extra value).

The result is the update of s (the caller adds it to the stream). No structure yet (the first round of the trunk): pass centers = 0, the direction messages vanish.
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from pomelo.models.primitives import geometry as G
from pomelo.models.primitives.irreps import Feats, Mult, TypedLinear, TypedNorm, mask_pairs, pair_valid

KINDS = G.KINDS
FLIP = {"0e": "0o", "0o": "0e", "1o": "1e", "1e": "1o"}
MASKED = -1e9


def split_heads(x: torch.Tensor, kind: str, heads: int) -> torch.Tensor:
    """[N, C] -> [N, H, C/H], [N, C, 3] -> [N, H, C/H, 3]"""
    return x.reshape(*x.shape[:-1], heads, -1) if kind in G.SCALAR_KINDS else x.reshape(*x.shape[:-2], heads, -1, 3)


def merge_heads(x: torch.Tensor, kind: str) -> torch.Tensor:
    return x.reshape(*x.shape[:-2], -1) if kind in G.SCALAR_KINDS else x.reshape(*x.shape[:-3], -1, 3)


def weighted_sum(beta: torch.Tensor, v: torch.Tensor, kind: str) -> torch.Tensor:
    """o_i = sum_j beta[h, i, j] v_j for every head: beta [H, N, N], v [N, H, d] or [N, H, d, 3] -> same shape"""
    return torch.einsum("hij,jhd->ihd", beta, v) if kind in G.SCALAR_KINDS else torch.einsum("hij,jhdc->ihdc", beta, v)


def head_norm(h: Dict[str, torch.Tensor], weight: nn.ParameterDict, eps: float = 1e-5) -> Dict[str, torch.Tensor]:
    """the query / key norm of a typed attention (PhysDock's RMSNorm of q and k per head): in every head the scalars (0e, 0o) divided by their RMS, the vectors (1o, 1e) by the RMS of
    their components, apart (so that the scalar logit of a pair does not depend on the vectors, which only count between tokens of one frame), then a learned scale per channel (the
    three components of a vector share it). h {kind: [..., H, d(, 3)]}"""
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


DIRECTION_LENGTH = 2.0              # the direction messages are squashed to at most 2 (8 Angstrom in the units of the coordinates of the model, 4 Angstrom)


def direction_message(beta: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """sum_j beta[h, i, j] (c_j - c_i) -> [N, H, 3], without the [N, N, 3] relative positions: sum_j beta_ij c_j - c_i sum_j beta_ij.
    The centres should be centred (the layer does it) so that the two terms are of the size of the differences."""
    m = torch.einsum("hij,jc->ihc", beta, centers) - centers[:, None, :] * beta.sum(-1).T[..., None]
    return G.squash(m, DIRECTION_LENGTH)                    # a weighted mean of displacements is as long as the molecule: bounded, as every input of a layer


class TypedAttention(nn.Module):
    """mult_s: the kinds of the single representation (all four, every C divisible by `heads`); mult_z: the kinds of the pair representation that are read
    ("0e" and "0o")."""

    def __init__(self, mult_s: Mult, mult_z: Mult, heads: int, zero_init: bool = False) -> None:
        super().__init__()
        if set(mult_s) != set(KINDS) or not {"0e", "0o"} <= set(mult_z):
            raise ValueError("the single representation needs the four kinds and the pair representation 0e and 0o")
        if any(c % heads for c in mult_s.values()):
            raise ValueError(f"every channel count {dict(mult_s)} must be divisible by the number of heads {heads}")
        self.mult_s, self.heads = dict(mult_s), heads
        self.norm = TypedNorm(self.mult_s)
        self.norm_z = TypedNorm({k: mult_z[k] for k in ("0e", "0o")}, affine=False)               # the pair state of the logits and gates (PhysDock's LayerNormPairBias)
        self.q = TypedLinear(self.mult_s, self.mult_s, bias=False)
        self.k = TypedLinear(self.mult_s, self.mult_s, bias=False)
        self.v0, self.ve = TypedLinear(self.mult_s, self.mult_s), TypedLinear(self.mult_s, self.mult_s)
        self.vo = TypedLinear(self.mult_s, {FLIP[k]: c for k, c in self.mult_s.items()})        # values of the opposite parity, for the odd gate
        self.dim = sum(c // heads * (1 if k in G.SCALAR_KINDS else 3) for k, c in self.mult_s.items())     # D of q . k
        self.q_norm = nn.ParameterDict({k: nn.Parameter(torch.ones(c // heads)) for k, c in self.mult_s.items()})
        self.k_norm = nn.ParameterDict({k: nn.Parameter(torch.ones(c // heads)) for k, c in self.mult_s.items()})
        self.d_0o = self.mult_s["0o"] // heads
        self.bias_e = nn.Linear(mult_z["0e"], heads, bias=False)                                  # b_h(z_0e)
        self.key_o = nn.Linear(mult_z["0o"], heads * self.d_0o, bias=False)                         # 0o -> 0o: no bias
        self.gate_e = nn.Linear(mult_z["0e"], 2 * heads)                                          # even gates: values and direction
        self.gate_o = nn.Linear(mult_z["0o"], 2 * heads, bias=False)                              # odd gates (tanh is odd, so a pseudoscalar stays one)
        self.mult_cat = {k: 3 * c + (heads if k in G.VECTOR_KINDS else 0) for k, c in self.mult_s.items()}
        self.gate = nn.Linear(self.mult_s["0e"], sum(self.mult_cat.values()))                      # output gates (even), one per channel
        self.out = TypedLinear(self.mult_cat, self.mult_s, zero_init=zero_init)

    def forward(self, s: Feats, z: Feats, centers: torch.Tensor, mask: torch.Tensor, same: torch.Tensor = None, attn_mask: torch.Tensor = None, normed: bool = False) -> Feats:
        """normed: s is normalised already (by the AdaLN of a DiT block). z is normalised here (no affine: the linear maps that read it have their own scale).

        s: {kind: [N, C(, 3)]}; z: {"0e": [N, N, Cz0e], "0o": [N, N, Cz0o]}; centers [N, 3] (the centre of every unit; equivariant: Q c + t); mask [N] (1 = real).
        same [N, N] (bool): the pairs of units that are in the same frame (the same conformer). The vectors of two units can only be combined (dot products in the logit,
        aggregation of vector values, direction messages) when they are in one frame, so every vector term is multiplied by `same`; the scalars (0e, 0o) of all pairs are used.
        None: all pairs are in one frame (the noisy structure of the decoder).  attn_mask [N, N] (bool): the pairs that may attend at all (the atoms of one conformer).
        Padding (mask 0) is replaced, not multiplied: whatever the padded rows of s, z or centers hold does not reach the real rows, and the padded rows of the output are 0.
        """
        H, N = self.heads, mask.shape[0]
        valid = mask > 0
        pair = valid[:, None] & valid[None, :]
        if attn_mask is not None:
            pair = pair & attn_mask
        sm = torch.ones(1, N, N, dtype=centers.dtype, device=centers.device) if same is None else same.to(centers.dtype)[None]
        c = torch.where(valid[:, None], centers, torch.zeros_like(centers))
        c = c - c.sum(0) / valid.sum().clamp(min=1)                                               # the centre of the real units at 0
        c = torch.where(valid[:, None], c, torch.zeros_like(c))
        x = {k: torch.where(valid.reshape(N, *([1] * (v.dim() - 1))), v, torch.zeros_like(v)) for k, v in (s if normed else self.norm(s)).items()}
        z = mask_pairs(self.norm_z(mask_pairs({k: z[k] for k in ("0e", "0o")}, valid)), valid)

        q, k = self.q(x), self.k(x)
        qh = head_norm({kind: split_heads(q[kind], kind, H) for kind in KINDS}, self.q_norm)
        kh = head_norm({kind: split_heads(k[kind], kind, H) for kind in KINDS}, self.k_norm)
        flat = lambda h, kinds: torch.cat([h[kind].flatten(2) for kind in kinds], -1)                  # [N, H, D_kinds]
        zo = self.key_o(z["0o"]).reshape(N, N, H, self.d_0o)
        logit = ((torch.einsum("ihd,jhd->hij", flat(qh, G.SCALAR_KINDS), flat(kh, G.SCALAR_KINDS))
                  + sm * torch.einsum("ihd,jhd->hij", flat(qh, G.VECTOR_KINDS), flat(kh, G.VECTOR_KINDS))) * self.dim ** -0.5
                 + self.bias_e(z["0e"]).permute(2, 0, 1) + torch.einsum("ihd,ijhd->hij", qh["0o"], zo))
        alpha = torch.softmax(logit.masked_fill(~pair, MASKED), -1)

        ge = torch.sigmoid(self.gate_e(z["0e"])).permute(2, 0, 1)                                  # [2H, N, N]
        go = torch.tanh(self.gate_o(z["0o"])).permute(2, 0, 1)
        zero = torch.zeros_like(alpha)
        beta0 = torch.where(pair, alpha, zero)
        beta_e, beta_eo = (torch.where(pair, alpha * g, zero) for g in ge.split(H))
        beta_o, beta_oo = (torch.where(pair, alpha * g, zero) for g in go.split(H))

        v0, ve, vo = self.v0(x), self.ve(x), self.vo(x)
        cat: Dict[str, list] = {}
        for kind in KINDS:
            f = sm if kind in G.VECTOR_KINDS else 1.0                    # a vector value of a unit in another frame cannot be added to a vector of this unit
            cat[kind] = [merge_heads(weighted_sum(beta0 * f, split_heads(v0[kind], kind, H), kind), kind),
                         merge_heads(weighted_sum(beta_e * f, split_heads(ve[kind], kind, H), kind), kind),
                         # the odd gate changes the parity of its values: values of kind flip(k) are what arrives as kind k
                         merge_heads(weighted_sum(beta_o * f, split_heads(vo[FLIP[kind]], FLIP[kind], H), FLIP[kind]), FLIP[kind])]
        cat["1o"].append(direction_message(beta_eo * sm, c))             # 0e gate * polar vector = polar
        cat["1e"].append(direction_message(beta_oo * sm, c))                                         # 0o gate * polar vector = axial
        merged = {kind: torch.cat(parts, dim=-1 if kind in G.SCALAR_KINDS else -2) for kind, parts in cat.items()}

        gates = torch.sigmoid(self.gate(x["0e"])).split(list(self.mult_cat.values()), dim=-1)
        merged = {kind: G.scale(merged[kind], kind, g, "0e")[0] for kind, g in zip(self.mult_cat, gates)}
        out = self.out(merged)
        return {kind: torch.where(valid.reshape(N, *([1] * (v.dim() - 1))), v, torch.zeros_like(v)) for kind, v in out.items()}


class TypedTriangleAttention(nn.Module):
    """Attention over the third node of a triangle (starting node: for row i, the entries ij attend over ik with the bias from jk; ending node: the same on the transposed
    pair), as PhysDock's TriangleAttention: one softmax of  q . k / sqrt(D) + b_h(jk)  (q . k: the 0e and the 0o parts, both even), the bias of the edge jk from LN(z_0e) and
    RMS(z_0o)^2 (an even function of the pseudoscalars), the even weights aggregate the values of both kinds (each keeps its kind), a sigmoid output gate from z_0e. One call of
    the memory-efficient kernel (the rows i are the batch, the bias [H, N, N] is shared by them): no [H, N, N, N] tensor. The pair representation holds scalars only, so there are
    no vector terms; 0e and 0o exchange information in the triangle multiplications. The update of z."""

    def __init__(self, mult_z: Mult, heads: int, starting: bool = True, zero_init: bool = False) -> None:
        super().__init__()
        c0, co = mult_z["0e"], mult_z["0o"]
        if set(mult_z) != {"0e", "0o"} or c0 % heads or co % heads:
            raise ValueError("the pair representation needs 0e and 0o with channels divisible by the heads")
        self.heads, self.starting, self.mult = heads, starting, dict(mult_z)
        self.dim = (c0 + co) // heads
        self.norm = TypedNorm(mult_z)
        self.qkv = TypedLinear(mult_z, {k: 3 * c for k, c in mult_z.items()}, bias=False)
        self.q_norm = nn.ParameterDict({k: nn.Parameter(torch.ones(c // heads)) for k, c in mult_z.items()})
        self.k_norm = nn.ParameterDict({k: nn.Parameter(torch.ones(c // heads)) for k, c in mult_z.items()})
        self.bias = nn.Linear(c0 + co, heads, bias=False)
        self.gate = nn.Linear(c0, c0 + co)
        self.out = TypedLinear(mult_z, mult_z, bias=False, zero_init=zero_init)

    def forward(self, z: Feats, mask: torch.Tensor, pair_mask: torch.Tensor = None) -> Feats:
        """pair_mask [N, N] (bool, None: all): the edges that may be keys and that get an update (PhysDock's z_mask of the template stack)"""
        H, N = self.heads, mask.shape[0]
        valid = mask > 0
        pm = pair_valid(valid) if pair_mask is None else pair_valid(valid) & pair_mask
        z = mask_pairs(z, valid)
        if not self.starting:
            z = {k: v.transpose(0, 1) for k, v in z.items()}
            pm = pm.transpose(0, 1)
        x = mask_pairs(self.norm(z), valid)
        heads = lambda t: t.reshape(N, N, H, -1)
        qkv = self.qkv(x)
        q, k, v = ({kind: heads(t.chunk(3, -1)[i]) for kind, t in qkv.items()} for i in range(3))
        q, k = head_norm(q, self.q_norm), head_norm(k, self.k_norm)
        qf, kf = (torch.cat([t["0e"], t["0o"]], -1).permute(0, 2, 1, 3) for t in (q, k))                 # [i, H, j / k, D]
        vf = torch.cat([v["0e"], v["0o"]], -1).permute(0, 2, 1, 3)
        bias = self.bias(torch.cat([x["0e"], x["0o"].square()], -1)).permute(2, 0, 1)                      # [H, j, k]: the edge jk
        if bool(pm.all()):
            attn = bias[None].expand(N, -1, -1, -1)
        else:                                                                                               # the key edge ik must be valid (finite: a row with no key stays finite)
            attn = bias[None] + torch.where(pm, 0.0, MASKED).to(bias.dtype)[:, None, None, :]
        dv = vf.shape[-1]
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            o = F.scaled_dot_product_attention(_pad8(qf), _pad8(kf), _pad8(vf), attn_mask=attn.to(qf.dtype), scale=self.dim ** -0.5)
        o = o[..., :dv].permute(0, 2, 1, 3)                                                                 # [i, j, H, Dv]
        c0, co = self.mult["0e"] // H, self.mult["0o"] // H
        g = torch.sigmoid(self.gate(x["0e"])).split([self.mult["0e"], self.mult["0o"]], -1)
        y = self.out({"0e": o[..., :c0].reshape(N, N, -1) * g[0], "0o": o[..., c0:c0 + co].reshape(N, N, -1) * g[1]})
        y = {kind: torch.where(pm[..., None], t, torch.zeros_like(t)) for kind, t in y.items()}
        if not self.starting:
            y = {kind: t.transpose(0, 1) for kind, t in y.items()}
        return mask_pairs(y, valid)


# ---------------------------------------------------------------------------------------------------------------- the attention of the DiT: one shared bias, one fused kernel
class PairBias(nn.Module):
    """The logit biases of every block of a DiT stack from its pair representation, computed once per structure and shared by all noisy samples (PhysDock's LayerNormPairBias):
    bias[b] = linear_b[ LN(z_0e), RMS(z_0o)^2 ] -> [num_blocks, H, N, N]. A logit must be even: z_0o enters through its square (an even function of a pseudoscalar); its sign, the
    hand, reaches the decoder through the 0o channels of the atoms and tokens (q . k)."""

    def __init__(self, mult_z: Mult, heads: int, num_blocks: int) -> None:
        super().__init__()
        self.heads, self.num_blocks = heads, num_blocks
        self.norm = TypedNorm({k: mult_z[k] for k in ("0e", "0o")}, affine=False)
        self.linear = nn.Linear(mult_z["0e"] + mult_z["0o"], num_blocks * heads)

    def forward(self, z: Feats) -> torch.Tensor:
        n = self.norm({k: z[k] for k in ("0e", "0o")})
        b = self.linear(torch.cat([n["0e"], n["0o"].square()], -1))                          # [N, N, blocks * H]
        return b.permute(2, 0, 1).reshape(self.num_blocks, self.heads, *b.shape[:2]).contiguous()


def _pad8(x: torch.Tensor) -> torch.Tensor:
    """zero columns up to a multiple of 8 (the memory-efficient kernel wants the head width divisible by 4); a concatenation, not F.pad (whose backward fails to compile
    with dynamic shapes)"""
    p = (-x.shape[-1]) % 8
    return torch.cat([x, x.new_zeros(*x.shape[:-1], p)], -1) if p else x


class SharedBiasAttention(nn.Module):
    """The typed attention of a DiT block, for a batch of noisy samples [B, N, ...] that share the logit bias [H, N, N] of the structure (PhysDock's DiTAttention): one softmax
    of  q . k / sqrt(D) + bias,  q . k the sum of the dot products within every kind (the vectors flattened: three components, the same weights), RMS-normalised per head and
    part (scalars, vectors); the weights (even) aggregate the values of every kind and the positions c_j, so the direction message sum_j alpha_ij c_j - c_i (a polar vector,
    squashed) comes from the same call. Everything is one call of the memory-efficient kernel: no [B, H, N, N] tensor exists. All tokens / atoms are in one frame (the noisy
    structure). The parity of the kinds is kept (the weights are even); the kinds exchange information in the products of the block (GeometricProduct), not here.
    forward(x, bias, centers) is the update; x is normalised already (the AdaLN of the block)."""

    def __init__(self, mult: Mult, heads: int, zero_init: bool = False, direction_length: float = DIRECTION_LENGTH, distance: bool = False) -> None:
        """direction_length: the bound of the direction messages (in the units of the positions); distance: the logits get  - w_h |c_i - c_j|^2  of the positions passed to
        forward (w [B, H] >= 0 from the caller): the attention can look at the atoms near in the noisy sample"""
        super().__init__()
        if set(mult) != set(KINDS) or any(c % heads for c in mult.values()):
            raise ValueError(f"the stream needs the four kinds with channels {dict(mult)} divisible by the heads {heads}")
        self.mult, self.heads, self.direction_length, self.distance = dict(mult), heads, direction_length, distance
        self.qkv = TypedLinear(self.mult, {k: 3 * c for k, c in self.mult.items()})                     # one matmul per kind for q, k and v
        self.q_norm = nn.ParameterDict({k: nn.Parameter(torch.ones(c // heads)) for k, c in self.mult.items()})
        self.k_norm = nn.ParameterDict({k: nn.Parameter(torch.ones(c // heads)) for k, c in self.mult.items()})
        self.dim = sum(c // heads * (1 if k in G.SCALAR_KINDS else 3) for k, c in self.mult.items())
        self.mult_cat = {k: c + (heads if k == "1o" else 0) for k, c in self.mult.items()}
        self.gate = nn.Linear(self.mult["0e"], sum(self.mult_cat.values()))
        self.out = TypedLinear(self.mult_cat, self.mult, zero_init=zero_init)

    def flat(self, h: Dict[str, torch.Tensor]) -> torch.Tensor:
        """{kind: [B, N, H, d(, 3)]} -> [B, H, N, D]"""
        return torch.cat([h[k] if k in G.SCALAR_KINDS else h[k].flatten(-2) for k in KINDS], -1).transpose(1, 2)

    def forward(self, x: Feats, bias: torch.Tensor, centers: torch.Tensor, w: torch.Tensor = None) -> Feats:
        """x {kind: [B, N, C(, 3)]}, bias [H, N, N] (shared by the B samples; a mask is part of it: MASKED where a pair may not attend), centers [B, N, 3] (centred: the distance
        term expands |c_i - c_j|^2, small numbers cancel less), w [B, H] the weights of the distance term (distance=True) -> the update

        The distance term in the same kernel call:  - w |c_i - c_j|^2 = w (2 c_i . c_j - |c_j|^2) - w |c_i|^2, the last a constant of the query (the softmax drops it), so q gets
        the columns [c_i, 1] and k the columns [2 w c_j, - w |c_j|^2] (divided by the scale of the kernel). A function of distances: even, the kinds are kept."""
        H = self.heads
        b, n = x["0e"].shape[:2]
        heads = lambda t: {k: split_heads(v, k, H) for k, v in t.items()}
        qkv = self.qkv(x)
        split = lambda i: {kind: t.chunk(3, -1 if kind in G.SCALAR_KINDS else -2)[i] for kind, t in qkv.items()}
        q, k = head_norm(heads(split(0)), self.q_norm), head_norm(heads(split(1)), self.k_norm)
        v = heads(split(2))
        qf, kf = self.flat(q), self.flat(k)
        if self.distance:
            c = centers[:, None].expand(b, H, n, 3)
            wh = w[:, :, None, None] * self.dim ** 0.5                                                     # undo the scale of the kernel
            qf = torch.cat([qf, c, torch.ones_like(c[..., :1])], -1)
            kf = torch.cat([kf, 2 * wh * c, -wh * c.square().sum(-1, keepdim=True)], -1)
        vf = torch.cat([self.flat(v), centers[:, None].expand(b, H, n, 3)], -1)
        dv = vf.shape[-1]
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            o = F.scaled_dot_product_attention(_pad8(qf), _pad8(kf), _pad8(vf), attn_mask=bias[None].expand(b, -1, -1, -1).to(qf.dtype), scale=self.dim ** -0.5)
        o = o[..., :dv].transpose(1, 2)                                                               # [B, N, H, Dv]
        parts, i = {}, 0
        for kind in KINDS:
            d = self.mult[kind] // H
            w = d if kind in G.SCALAR_KINDS else 3 * d
            parts[kind] = merge_heads(o[..., i:i + w] if kind in G.SCALAR_KINDS else o[..., i:i + w].unflatten(-1, (d, 3)), kind)
            i += w
        direction = G.squash(o[..., i:i + 3] - centers[:, :, None], self.direction_length)             # [B, N, H, 3]: sum_j alpha_ij c_j - c_i, polar
        parts["1o"] = torch.cat([parts["1o"], direction], -2)
        gates = torch.sigmoid(self.gate(x["0e"])).split(list(self.mult_cat.values()), dim=-1)
        return self.out({kind: G.scale(parts[kind], kind, g, "0e")[0] for kind, g in zip(self.mult_cat, gates)})
