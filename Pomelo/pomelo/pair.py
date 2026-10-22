"""The pair representation z = {"0e": [N, N, C0e], "0o": [N, N, C0o]} of the trunk and its layers. Torch only, parity-equivariant by construction.

Only the even and the odd scalars are stored. Vectors appear inside the layers, built from the structure:  w_ij = c_j - c_i  (the centres of the units, a polar vector,
squashed to the bounded direction  w_hat = w / sqrt(l^2 + |w|^2))  and from the vectors u_k of the single representation. They are products of kinds by the rules of
geometry.py: an even coefficient times w_hat is polar (1o), an odd coefficient (a tanh of a pseudoscalar) times w_hat is axial (1e), w_jk x u_k is axial.

Note that the three points i, j, k of a triangle are coplanar, so no product of the w's of a triangle is a signed volume: the out-of-plane direction u_k of the single
representation is what the triangle layers use to see the handedness.
"""
from typing import Dict, List

import torch
import torch.nn as nn

from pomelo import geometry as G
from pomelo.blocks import TypedTransition, add, mask_rows
from pomelo.irreps import Feats, Mult, TypedLinear, TypedRMSNorm

MASKED = -1e9


def pair_valid(valid: torch.Tensor) -> torch.Tensor:
    return valid[:, None] & valid[None, :]


def mask_pairs(z: Feats, valid: torch.Tensor) -> Feats:
    """the entries of padded units replaced by zeros (replaced, not multiplied)"""
    p = pair_valid(valid)[..., None]
    return {k: torch.where(p, v, torch.zeros_like(v)) for k, v in z.items()}


def directions(centers: torch.Tensor, valid: torch.Tensor, length_scale: float) -> torch.Tensor:
    """w_hat[i, j] = (c_j - c_i) / sqrt(l^2 + |c_j - c_i|^2): [N, N, 3], a polar vector, bounded by 1, smooth at 0; zero for padded units"""
    c = torch.where(valid[:, None], centers, torch.zeros_like(centers))
    w = c[None, :, :] - c[:, None, :]
    w_hat = w / torch.sqrt(length_scale ** 2 + w.square().sum(-1, keepdim=True))
    return torch.where(pair_valid(valid)[..., None], w_hat, torch.zeros_like(w_hat))


def _need(mult_s: Mult, mult_z: Mult) -> None:
    if set(mult_s) != set(G.KINDS) or set(mult_z) != {"0e", "0o"}:
        raise ValueError("the single representation needs the four kinds, the pair representation 0e and 0o")


class PairFromSingle(nn.Module):
    """single -> pair: the products of a projection of s_i and one of s_j (all the allowed ones by kind), with the direction w_hat_ij for the pair-level ones, mixed into z. The
    update of z."""

    def __init__(self, mult_s: Mult, mult_z: Mult, channels: int = 16, length_scale: float = 8.0, zero_init: bool = False) -> None:
        super().__init__()
        _need(mult_s, mult_z)
        self.length_scale = length_scale
        self.norm = TypedRMSNorm(mult_s)
        proj = {k: channels for k in mult_s}
        self.proj_a, self.proj_b = TypedLinear(mult_s, proj), TypedLinear(mult_s, proj)
        self.out = TypedLinear({"0e": 6 * channels, "0o": 7 * channels}, mult_z, zero_init=zero_init)

    def forward(self, s: Feats, centers: torch.Tensor, mask: torch.Tensor) -> Feats:
        valid = mask > 0
        x = mask_rows(self.norm(mask_rows(s, valid)), valid)
        a, b = self.proj_a(x), self.proj_b(x)
        wh = directions(centers, valid, self.length_scale)[:, :, None, :]                    # [N, N, 1, 3]
        ai = {k: v[:, None] for k, v in a.items()}                                              # i side [N, 1, P(, 3)]
        bj = {k: v[None, :] for k, v in b.items()}                                              # j side [1, N, P(, 3)]
        out: Dict[str, List[torch.Tensor]] = {"0e": [], "0o": []}

        def put(r):
            out[r[1]].append(r[0])
        put(G.scale(ai["0e"], "0e", bj["0e"], "0e"))                                           # 0e * 0e
        put(G.scale(ai["0o"], "0o", bj["0o"], "0o"))                                           # 0o * 0o = 0e
        put(G.dot(ai["1o"], "1o", bj["1o"], "1o"))
        put(G.dot(ai["1e"], "1e", bj["1e"], "1e"))
        put(G.dot(ai["1o"], "1o", wh, "1o"))
        put(G.dot(bj["1o"], "1o", wh, "1o"))
        put(G.scale(ai["0e"], "0e", bj["0o"], "0o"))                                           # 0e * 0o = 0o
        put(G.scale(ai["0o"], "0o", bj["0e"], "0e"))
        put(G.dot(ai["1o"], "1o", bj["1e"], "1e"))                                              # 1o . 1e = 0o
        put(G.dot(ai["1e"], "1e", bj["1o"], "1o"))
        put(G.triple(ai["1o"], "1o", bj["1o"], "1o", wh, "1o"))                                # the signed volume of u_i, u_j, w_ij
        put(G.dot(ai["1e"], "1e", wh, "1o"))
        put(G.dot(bj["1e"], "1e", wh, "1o"))
        return mask_pairs(self.out({k: torch.cat(v, -1) for k, v in out.items()}), valid)


class TypedTriangleMultiplication(nn.Module):
    """The triangle multiplicative update, outgoing (sum_k a_ik b_jk) or incoming (sum_k a_ki b_kj), typed. The hidden features of an edge are scalars (even and odd, from z)
    and vectors (w_hat of the edge times a coefficient: an even one gives a polar vector, an odd one an axial vector; on the b side also w_jk x u_k). The sums of products
    that come out as scalars are the update: 0e from 0e0e, 0o0o, 1o.1o, 1e.1e, 0o from 0e0o, 0o0e, 1o.1e, 1e.1o. Of the last ones, w_ik . (w_jk x u_k) is the signed volume of the
    triangle with the direction of the unit k: the handedness of the triangle."""

    def __init__(self, mult_s: Mult, mult_z: Mult, outgoing: bool, channels: int = 16, length_scale: float = 8.0, zero_init: bool = False) -> None:
        super().__init__()
        _need(mult_s, mult_z)
        c, self.outgoing, self.length_scale = channels, outgoing, length_scale
        sc = {"0e": c, "0o": c}
        self.norm_z, self.norm_u = TypedRMSNorm(mult_z), TypedRMSNorm({"1o": mult_s["1o"]})          # of s only the polar vectors are read
        self.a_scalar, self.b_scalar = TypedLinear(mult_z, sc, bias=False), TypedLinear(mult_z, sc, bias=False)
        self.a_gate, self.b_gate = nn.Linear(mult_z["0e"], c), nn.Linear(mult_z["0e"], c)
        self.a_ce, self.b_ce = nn.Linear(mult_z["0e"], c, bias=False), nn.Linear(mult_z["0e"], c, bias=False)
        self.a_co = nn.Linear(mult_z["0o"], c, bias=False)
        self.b_cx = nn.Linear(mult_z["0e"], c, bias=False)
        self.proj_u = TypedLinear({"1o": mult_s["1o"]}, {"1o": c}, bias=False)
        cat = {"0e": 4 * c, "0o": 4 * c}
        self.norm_out = TypedRMSNorm(cat)
        self.lin_out = TypedLinear(cat, mult_z, bias=False, zero_init=zero_init)
        self.gate_out = nn.Linear(mult_z["0e"], sum(mult_z.values()))
        self.split = list(mult_z.values())

    def forward(self, s: Feats, z: Feats, centers: torch.Tensor, mask: torch.Tensor) -> Feats:
        valid = mask > 0
        pair = pair_valid(valid)
        zn = mask_pairs(self.norm_z(mask_pairs(z, valid)), valid)
        z_out = zn["0e"]                                                                       # the gate of the output reads z_ij itself
        wh = directions(centers, valid, self.length_scale)                                      # [N, N, 3]
        u = self.proj_u(mask_rows(self.norm_u(mask_rows({"1o": s["1o"]}, valid)), valid))["1o"]       # [N, c, 3]
        if not self.outgoing:                                                                  # the incoming update is the outgoing one on the transposed edges
            zn, wh = {k: v.transpose(0, 1) for k, v in zn.items()}, wh.transpose(0, 1)
        wv = wh[:, :, None, :]                                                                  # [N, N, 1, 3]
        pm = pair[..., None]

        def scalar_side(scalar, gate):
            g, sc = torch.sigmoid(gate(zn["0e"])), scalar(zn)
            return sc["0e"] * g * pm, sc["0o"] * g * pm
        a0e, a0o = scalar_side(self.a_scalar, self.a_gate)
        b0e, b0o = scalar_side(self.b_scalar, self.b_gate)
        a1o = wv * self.a_ce(zn["0e"])[..., None] * pm[..., None]                               # even coefficient * polar = polar
        a1e = wv * torch.tanh(self.a_co(zn["0o"]))[..., None] * pm[..., None]                   # odd coefficient * polar = axial
        b1o = wv * self.b_ce(zn["0e"])[..., None] * pm[..., None]
        b1e = torch.linalg.cross(wv, u[None], dim=-1) * self.b_cx(zn["0e"])[..., None] * pm[..., None]    # w_jk x u_k: axial

        e = lambda x, y: torch.einsum("ikc,jkc->ijc", x, y)
        ev = lambda x, y: torch.einsum("ikcx,jkcx->ijc", x, y)
        cat = {"0e": torch.cat([e(a0e, b0e), e(a0o, b0o), ev(a1o, b1o), ev(a1e, b1e)], -1),
               "0o": torch.cat([e(a0e, b0o), e(a0o, b0e), ev(a1o, b1e), ev(a1e, b1o)], -1)}
        y = self.lin_out(self.norm_out(cat))
        gates = torch.sigmoid(self.gate_out(z_out)).split(self.split, dim=-1)
        y = {k: G.scale(v, k, g, "0e")[0] for (k, v), g in zip(y.items(), gates)}
        return mask_pairs(y, valid)


class TypedTriangleAttention(nn.Module):
    """Attention over the third node of a triangle (starting node: for row i, the entries ij attend over ik with the bias from jk; ending node: the same on the transposed
    pair), typed like the attention of the single representation (one softmax, three weight groups: alpha, alpha * sigmoid(z_0e_jk), alpha * tanh(z_0o_jk) with the values of the
    opposite parity). The pair representation holds scalars only, so there are no vector terms. The update of z."""

    def __init__(self, mult_z: Mult, heads: int, starting: bool = True, zero_init: bool = False) -> None:
        super().__init__()
        c0, co = mult_z["0e"], mult_z["0o"]
        if set(mult_z) != {"0e", "0o"} or c0 % heads or co % heads:
            raise ValueError("the pair representation needs 0e and 0o with channels divisible by the heads")
        self.heads, self.starting, self.d_o = heads, starting, co // heads
        self.dim = (c0 + co) // heads
        self.norm = TypedRMSNorm(mult_z)
        self.q, self.k = TypedLinear(mult_z, mult_z, bias=False), TypedLinear(mult_z, mult_z, bias=False)
        self.v0, self.ve = TypedLinear(mult_z, mult_z, bias=False), TypedLinear(mult_z, mult_z, bias=False)
        self.vo = TypedLinear(mult_z, {"0o": c0, "0e": co}, bias=False)
        self.bias_e = nn.Linear(c0, heads, bias=False)
        self.key_o = nn.Linear(co, heads * self.d_o, bias=False)
        self.gate_e, self.gate_o = nn.Linear(c0, heads), nn.Linear(co, heads, bias=False)
        self.mult_cat = {"0e": 3 * c0, "0o": 3 * co}
        self.gate = nn.Linear(c0, sum(self.mult_cat.values()))
        self.out = TypedLinear(self.mult_cat, mult_z, bias=False, zero_init=zero_init)

    def forward(self, z: Feats, mask: torch.Tensor) -> Feats:
        H, N = self.heads, mask.shape[0]
        valid = mask > 0
        z = mask_pairs(z, valid)
        if not self.starting:
            z = {k: v.transpose(0, 1) for k, v in z.items()}
        x = mask_pairs(self.norm(z), valid)
        q, k = self.q(x), self.k(x)
        heads = lambda t: t.reshape(N, N, H, -1)
        q_cat = torch.cat([heads(q["0e"]), heads(q["0o"])], -1)                                 # [i, j, H, D]
        k_cat = torch.cat([heads(k["0e"]), heads(k["0o"])], -1)                                 # [i, k, H, D]
        logit = (torch.einsum("ijhd,ikhd->hijk", q_cat, k_cat) * self.dim ** -0.5
                 + self.bias_e(x["0e"]).permute(2, 0, 1)[:, None]                                # the bias of the edge jk
                 + torch.einsum("ijhd,jkhd->hijk", heads(q["0o"]), self.key_o(x["0o"]).reshape(N, N, H, self.d_o)))
        ok = (valid[:, None, None] & valid[None, :, None] & valid[None, None, :])[None]
        alpha = torch.where(ok, torch.softmax(logit.masked_fill(~ok, MASKED), -1), torch.zeros_like(logit))
        beta_e = alpha * torch.sigmoid(self.gate_e(x["0e"])).permute(2, 0, 1)[:, None]
        beta_o = alpha * torch.tanh(self.gate_o(x["0o"])).permute(2, 0, 1)[:, None]
        v0, ve, vo = self.v0(x), self.ve(x), self.vo(x)
        agg = lambda beta, v: torch.einsum("hijk,ikhd->ijhd", beta, heads(v)).reshape(N, N, -1)
        flip = {"0e": "0o", "0o": "0e"}
        cat = {kind: torch.cat([agg(alpha, v0[kind]), agg(beta_e, ve[kind]), agg(beta_o, vo[flip[kind]])], -1) for kind in ("0e", "0o")}
        gates = torch.sigmoid(self.gate(x["0e"])).split(list(self.mult_cat.values()), dim=-1)
        y = self.out({kind: G.scale(cat[kind], kind, g, "0e")[0] for (kind, g) in zip(self.mult_cat, gates)})
        if not self.starting:
            y = {kind: v.transpose(0, 1) for kind, v in y.items()}
        return mask_pairs(y, valid)


class PairBlock(nn.Module):
    """One block of the pair representation (the order of PhysDock's current structure): z += single -> pair, z += triangle multiplication (outgoing, incoming), z += triangle
    attention (starting, ending node), z += transition."""

    def __init__(self, mult_s: Mult, mult_z: Mult, heads: int, channels: int = 16, length_scale: float = 8.0, expansion: float = 8 / 3, zero_init: bool = False) -> None:
        super().__init__()
        self.outer = PairFromSingle(mult_s, mult_z, channels, length_scale, zero_init)
        self.mul_out = TypedTriangleMultiplication(mult_s, mult_z, True, channels, length_scale, zero_init)
        self.mul_in = TypedTriangleMultiplication(mult_s, mult_z, False, channels, length_scale, zero_init)
        self.att_start = TypedTriangleAttention(mult_z, heads, True, zero_init)
        self.att_end = TypedTriangleAttention(mult_z, heads, False, zero_init)
        self.transition = TypedTransition(mult_z, expansion, zero_init=zero_init)

    def forward(self, s: Feats, z: Feats, centers: torch.Tensor, mask: torch.Tensor) -> Feats:
        valid = mask > 0
        z = mask_pairs(z, valid)
        z = add(z, self.outer(s, centers, mask))
        z = add(z, self.mul_out(s, z, centers, mask))
        z = add(z, self.mul_in(s, z, centers, mask))
        z = add(z, self.att_start(z, mask))
        z = add(z, self.att_end(z, mask))
        return mask_pairs(add(z, self.transition(z)), valid)
