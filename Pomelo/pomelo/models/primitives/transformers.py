"""The feed-forward layer of the typed streams: TypedTransition. Torch only, parity-equivariant by construction (see irreps.py for the rules)."""
import torch.nn as nn
import torch.nn.functional as F

from pomelo.models.primitives import geometry as G
from pomelo.models.primitives.irreps import Feats, Mult, TypedLinear, TypedNorm


class TypedTransition(nn.Module):
    """SwiGLU feed-forward of a feature set. "0e": w2(silu(w1 x) * w3 x), the usual one, from the "0e" channels alone. The other kinds: w2_k((w1_k x_k) * silu(g_k(x_0e))),
    a hidden layer of the same kind scaled by a gate that the even channels compute (the kind is kept): the scalars set the size of every vector channel. forward(x) is the
    update."""

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
