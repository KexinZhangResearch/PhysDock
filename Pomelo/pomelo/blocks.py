"""The blocks of the trunk, built from the typed layers. Torch only, parity-equivariant by construction (see irreps.py for the rules)."""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from pomelo import geometry as G
from pomelo.attention import TypedAttention
from pomelo.irreps import Feats, GeometricProduct, Mult, TypedLinear, TypedRMSNorm


def add(a: Feats, b: Feats) -> Feats:
    """residual connection of two feature sets of the same kinds"""
    return {k: v + b[k] for k, v in a.items()}


def mask_rows(x: Feats, valid: torch.Tensor) -> Feats:
    """the rows of padded units replaced by zeros (replaced, not multiplied: whatever they hold, nan included, is gone)"""
    return {k: torch.where(valid.reshape(-1, *([1] * (v.dim() - 1))), v, torch.zeros_like(v)) for k, v in x.items()}


class TypedTransition(nn.Module):
    """SwiGLU feed-forward of a feature set. "0e": w2(silu(w1 x) * w3 x), the usual one, from the "0e" channels alone. The other kinds: w2_k((w1_k x_k) * silu(g_k(x_0e))),
    a hidden layer of the same kind scaled by a gate that the even channels compute (the kind is kept). forward(x) is the update."""

    def __init__(self, mult: Mult, expansion: float = 8 / 3, zero_init: bool = False) -> None:
        super().__init__()
        self.hidden = {k: max(1, int(round(expansion * c))) for k, c in mult.items() if c}
        self.norm = TypedRMSNorm(mult)
        self.w1 = TypedLinear(mult, self.hidden, bias=False)
        self.w3 = nn.Linear(mult["0e"], sum(self.hidden.values()), bias=False)            # the 0e half of the SwiGLU and the gates of the other kinds
        self.w2 = TypedLinear(self.hidden, mult, bias=False, zero_init=zero_init)

    def forward(self, x: Feats) -> Feats:
        x = self.norm(x)
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

    def forward(self, s: Feats, z: Feats, centers: torch.Tensor, mask: torch.Tensor) -> Feats:
        valid = mask > 0
        s = mask_rows(s, valid)
        s = mask_rows(add(s, self.product(s)), valid)
        s = add(s, self.attention(s, z, centers, mask))
        return mask_rows(add(s, self.transition(s)), valid)
