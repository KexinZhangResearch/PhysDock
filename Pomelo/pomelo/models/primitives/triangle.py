"""The layers of the pair representation z = {"0e": [N, N, C0e], "0o": [N, N, C0o]} of the trunk and its layers. Torch only, parity-equivariant by construction.

Only the even and the odd scalars are stored. Vectors appear inside the layers, built from the structure:  w_ij = c_j - c_i  (the centres of the units, a polar vector,
squashed to the bounded direction  w_hat = w / sqrt(l^2 + |w|^2))  and from the vectors u_k of the single representation. They are products of kinds by the rules of
geometry.py: an even coefficient times w_hat is polar (1o), an odd coefficient (a tanh of a pseudoscalar) times w_hat is axial (1e), w_jk x u_k is axial.

Note that the three points i, j, k of a triangle are coplanar, so no product of the w's of a triangle is a signed volume: the out-of-plane direction u_k of the single
representation is what the triangle layers use to see the handedness.
"""
from typing import Dict, List

import torch
import torch.nn as nn

from pomelo.models.primitives import geometry as G
from pomelo.models.primitives.irreps import Feats, Mult, TypedLinear, TypedNorm, mask_pairs, mask_rows, pair_valid

def directions(centers: torch.Tensor, valid: torch.Tensor, length_scale: float, same: torch.Tensor = None) -> torch.Tensor:
    """w_hat[i, j] = (c_j - c_i) / sqrt(l^2 + |c_j - c_i|^2): [N, N, 3], a polar vector, bounded by 1, smooth at 0; zero for padded units, and for the pairs that are not in the same
    frame (`same` [N, N] bool, None: all): the centres of two conformers are in two unrelated frames and have no direction between them"""
    c = torch.where(valid[:, None], centers, torch.zeros_like(centers))
    w = c[None, :, :] - c[:, None, :]
    w_hat = w / torch.sqrt(length_scale ** 2 + w.square().sum(-1, keepdim=True))
    keep = pair_valid(valid) if same is None else pair_valid(valid) & same
    return torch.where(keep[..., None], w_hat, torch.zeros_like(w_hat))


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
        self.norm = TypedNorm(mult_s)
        proj = {k: channels for k in mult_s}
        self.proj_a, self.proj_b = TypedLinear(mult_s, proj), TypedLinear(mult_s, proj)
        self.out = TypedLinear({"0e": 6 * channels, "0o": 7 * channels}, mult_z, zero_init=zero_init)

    def forward(self, s: Feats, centers: torch.Tensor, mask: torch.Tensor, same: torch.Tensor = None) -> Feats:
        valid = mask > 0
        x = mask_rows(self.norm(mask_rows(s, valid)), valid)
        a, b = self.proj_a(x), self.proj_b(x)
        wh = directions(centers, valid, self.length_scale, same)[:, :, None, :]              # [N, N, 1, 3]
        sm = (torch.ones(1, 1, 1, dtype=s["0e"].dtype, device=s["0e"].device) if same is None else same.to(s["0e"].dtype)[..., None])    # vector products only in one frame
        ai = {k: v[:, None] for k, v in a.items()}                                              # i side [N, 1, P(, 3)]
        bj = {k: v[None, :] for k, v in b.items()}                                              # j side [1, N, P(, 3)]
        out: Dict[str, List[torch.Tensor]] = {"0e": [], "0o": []}

        def put(r, m=1.0):
            out[r[1]].append(r[0] * m)
        put(G.scale(ai["0e"], "0e", bj["0e"], "0e"))                                           # 0e * 0e
        put(G.scale(ai["0o"], "0o", bj["0o"], "0o"))                                           # 0o * 0o = 0e
        put(G.dot(ai["1o"], "1o", bj["1o"], "1o"), sm)
        put(G.dot(ai["1e"], "1e", bj["1e"], "1e"), sm)
        put(G.dot(ai["1o"], "1o", wh, "1o"))
        put(G.dot(bj["1o"], "1o", wh, "1o"))
        put(G.scale(ai["0e"], "0e", bj["0o"], "0o"))                                           # 0e * 0o = 0o
        put(G.scale(ai["0o"], "0o", bj["0e"], "0e"))
        put(G.dot(ai["1o"], "1o", bj["1e"], "1e"), sm)                                          # 1o . 1e = 0o
        put(G.dot(ai["1e"], "1e", bj["1o"], "1o"), sm)
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
        self.norm_z, self.norm_u = TypedNorm(mult_z), TypedNorm({"1o": mult_s["1o"]})          # of s only the polar vectors are read
        self.a_scalar, self.b_scalar = TypedLinear(mult_z, sc, bias=False), TypedLinear(mult_z, sc, bias=False)
        self.a_gate, self.b_gate = nn.Linear(mult_z["0e"], c), nn.Linear(mult_z["0e"], c)
        self.a_ce, self.b_ce = nn.Linear(mult_z["0e"], c, bias=False), nn.Linear(mult_z["0e"], c, bias=False)
        self.a_co = nn.Linear(mult_z["0o"], c, bias=False)
        self.b_cx = nn.Linear(mult_z["0e"], c, bias=False)
        self.proj_u = TypedLinear({"1o": mult_s["1o"]}, {"1o": c}, bias=False)
        cat = {"0e": 4 * c, "0o": 4 * c}
        self.norm_out = TypedNorm(cat)
        self.lin_out = TypedLinear(cat, mult_z, bias=False, zero_init=zero_init)
        self.gate_out = nn.Linear(mult_z["0e"], sum(mult_z.values()))
        self.split = list(mult_z.values())

    def forward(self, s: Feats, z: Feats, centers: torch.Tensor, mask: torch.Tensor, same: torch.Tensor = None) -> Feats:
        valid = mask > 0
        pair = pair_valid(valid)
        zn = mask_pairs(self.norm_z(mask_pairs(z, valid)), valid)
        z_out = zn["0e"]                                                                       # the gate of the output reads z_ij itself
        wh = directions(centers, valid, self.length_scale, same)                                # [N, N, 3]: zero between two frames, so a triangle has vector terms only inside one
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
