"""The transformer blocks of the trunk, built from the typed layers: TypedTransition, SingleBlock (geometric products, attention with the pair state, transition), PairBlock (single ->
pair, triangle multiplication out / in, triangle attention start / end, transition) and TrunkBlock (pair block, then single block: z learns from s, then s from z). Torch only,
parity-equivariant by construction (see irreps.py for the rules)."""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pomelo.models.primitives import geometry as G
from pomelo.models.primitives.attentions import TypedAttention, TypedTriangleAttention
from pomelo.models.primitives.irreps import (Feats, GeometricProduct, Mult, TypedLinear, TypedNorm, add, mask_pairs, mask_rows)
from pomelo.models.primitives.triangle import PairFromSingle, TypedTriangleMultiplication


class TypedTransition(nn.Module):
    """SwiGLU feed-forward of a feature set. "0e": w2(silu(w1 x) * w3 x), the usual one, from the "0e" channels alone. The other kinds: w2_k((w1_k x_k) * silu(g_k(x_0e))),
    a hidden layer of the same kind scaled by a gate that the even channels compute (the kind is kept). forward(x) is the update."""

    def __init__(self, mult: Mult, expansion: float = 8 / 3, zero_init: bool = False) -> None:
        super().__init__()
        self.hidden = {k: max(1, int(round(expansion * c))) for k, c in mult.items() if c}
        self.norm = TypedNorm(mult)
        self.w1 = TypedLinear(mult, self.hidden, bias=False)
        self.w3 = nn.Linear(mult["0e"], sum(self.hidden.values()), bias=False)            # the 0e half of the SwiGLU and the gates of the other kinds
        self.w2 = TypedLinear(self.hidden, mult, bias=False, zero_init=zero_init)

    def forward(self, x: Feats, normed: bool = False) -> Feats:
        """normed: x is normalised already (by the AdaLN of a DiT block)"""
        x = x if normed else self.norm(x)
        u = self.w1(x)
        b = self.w3(x["0e"]).split(list(self.hidden.values()), dim=-1)
        h = {}
        for (k, _), bk in zip(self.hidden.items(), b):
            h[k] = F.silu(u[k]) * bk if k == "0e" else G.scale(u[k], k, F.silu(bk), "0e")[0]
        return self.w2(h)


class SingleBlock(nn.Module):
    """One block of the single representation: s += geometric products, s += attention with the pair state, s += transition. The padded units come out as zeros.

    The sublayers keep the rules separately: the attention and the transition cannot make an odd feature out of even ones (the parity selection rule), the products
    can (a signed volume of three vectors is a pseudoscalar): that is where the handedness of the geometry enters the features."""

    def __init__(self, mult_s: Mult, mult_z: Mult, heads: int, product_channels: int = 32, expansion: float = 8 / 3, zero_init: bool = False) -> None:
        super().__init__()
        self.product = GeometricProduct(mult_s, mult_s, product_channels, zero_init=zero_init)
        self.attention = TypedAttention(mult_s, mult_z, heads, zero_init=zero_init)
        self.transition = TypedTransition(mult_s, expansion, zero_init=zero_init)

    def forward(self, s: Feats, z: Feats, centers: torch.Tensor, mask: torch.Tensor, same: torch.Tensor = None, attn_mask: torch.Tensor = None) -> Feats:
        valid = mask > 0
        s = mask_rows(s, valid)
        s = mask_rows(add(s, self.product(s)), valid)
        s = add(s, self.attention(s, z, centers, mask, same, attn_mask))
        return mask_rows(add(s, self.transition(s)), valid)


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

    def forward(self, s: Feats, z: Feats, centers: torch.Tensor, mask: torch.Tensor, same: torch.Tensor = None) -> Feats:
        valid = mask > 0
        z = mask_pairs(z, valid)
        z = add(z, self.outer(s, centers, mask, same))
        z = add(z, self.mul_out(s, z, centers, mask, same))
        z = add(z, self.mul_in(s, z, centers, mask, same))
        z = add(z, self.att_start(z, mask))
        z = add(z, self.att_end(z, mask))
        return mask_pairs(add(z, self.transition(z)), valid)


class TrunkBlock(nn.Module):
    def __init__(self, mult_s: Mult, mult_z: Mult, heads_s: int, heads_z: int, channels: int = 16, product_channels: int = 32, length_scale: float = 8.0,
                 zero_init: bool = False) -> None:
        super().__init__()
        self.pair = PairBlock(mult_s, mult_z, heads_z, channels, length_scale, zero_init=zero_init)
        self.single = SingleBlock(mult_s, mult_z, heads_s, product_channels, zero_init=zero_init)

    def forward(self, s: Feats, z: Feats, centers: torch.Tensor, mask: torch.Tensor, same: torch.Tensor = None) -> Tuple[Feats, Feats]:
        """same [N, N] (bool): the pairs of units in the same frame (the same conformer); None: all in one frame"""
        z = self.pair(s, z, centers, mask, same)
        return self.single(s, z, centers, mask, same), z


class TemplateBlock(nn.Module):
    """One block of the template stack (PhysDock's TriangleBlock): z += triangle multiplication (outgoing, incoming; the out-of-plane directions u_k are the template's), z +=
    triangle attention (starting, ending node), z += transition, all on the edges of pair_mask. As in PhysDock the edges outside the mask keep their value and get no gradient
    (a row with no valid key would amplify the gradient that reaches it in every block)."""

    def __init__(self, mult_z: Mult, mult_u: Mult, heads: int, channels: int = 16, length_scale: float = 8.0, expansion: float = 8 / 3) -> None:
        super().__init__()
        self.mul_out = TypedTriangleMultiplication(mult_u, mult_z, True, channels, length_scale)
        self.mul_in = TypedTriangleMultiplication(mult_u, mult_z, False, channels, length_scale)
        self.att_start = TypedTriangleAttention(mult_z, heads, True)
        self.att_end = TypedTriangleAttention(mult_z, heads, False)
        self.transition = TypedTransition(mult_z, expansion)

    def forward(self, u: Feats, z: Feats, centers: torch.Tensor, mask: torch.Tensor, same: torch.Tensor, pair_mask: torch.Tensor) -> Feats:
        y = add(z, self.mul_out(u, z, centers, mask, same, pair_mask))
        y = add(y, self.mul_in(u, y, centers, mask, same, pair_mask))
        y = add(y, self.att_start(y, mask, pair_mask))
        y = add(y, self.att_end(y, mask, pair_mask))
        y = add(y, self.transition(y))
        keep = pair_mask[..., None]
        return {k: torch.where(keep, v, z[k].detach()) for k, v in y.items()}
