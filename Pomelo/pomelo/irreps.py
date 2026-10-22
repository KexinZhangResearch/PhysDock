"""Typed feature sets and the layers that keep them O(3)-equivariant (rotations, translations, reflections), torch only.

A feature set is a dict {kind: tensor} with the kinds of geometry.py ("0e", "0o", "1o", "1e"); the tensors are [..., C_kind] for the scalars and
[..., C_kind, 3] for the vectors, so the same code holds a single representation [N, ...], a pair representation [N, N, ...] or an atom representation.
`mult` (a dict {kind: C_kind}) is the number of channels of every kind. The rules that make a layer parity-equivariant, all enforced here:

* a linear map mixes the channels of one kind only (0e with 0e, 0o with 0o, ...; the three components of a vector share the weights);
* only the "0e" channels have a bias, and only they, together with the function of the length of a vector, go through a non-linearity or a softmax;
* the other kinds are scaled by a "0e" gate (the kind is kept), or multiplied with each other by the products of geometry.py (the kinds combine by
  the parity rule), so that information moves between kinds only through products;
* the scales of the normalisation are "0e" numbers (the RMS of a pseudoscalar or of the lengths of vectors is even).
"""
import math
from typing import Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from pomelo import geometry as G

Feats = Dict[str, torch.Tensor]
Mult = Mapping[str, int]


def zeros(mult: Mult, shape, dtype=torch.float32, device=None) -> Feats:
    return {k: torch.zeros(*shape, c, *((3,) if k in G.VECTOR_KINDS else ()), dtype=dtype, device=device) for k, c in mult.items() if c}


def randn(mult: Mult, shape, generator: torch.Generator = None, dtype=torch.float64) -> Feats:
    return {k: G.random_feature(k, shape, c, generator, dtype) for k, c in mult.items() if c}


def transform(x: Feats, q: torch.Tensor) -> Feats:
    """the feature set transformed by the orthogonal matrix q: every kind by its own rule"""
    return {k: G.transform(v, k, q) for k, v in x.items()}


def mult_of(x: Feats) -> Dict[str, int]:
    return {k: v.shape[-1] if k in G.SCALAR_KINDS else v.shape[-2] for k, v in x.items()}


class TypedLinear(nn.Module):
    """Linear map within every kind: out[k] = W_k in[k] (+ b for "0e" only). A kind of `mult_out` needs the same kind in `mult_in`.

    A bias on "0o" would break the mirror symmetry (a constant is even), one on "1o" / "1e" the rotation symmetry, so there are none."""

    def __init__(self, mult_in: Mult, mult_out: Mult, bias: bool = True, zero_init: bool = False) -> None:
        super().__init__()
        self.mult_in, self.mult_out = dict(mult_in), dict(mult_out)
        self.weight = nn.ParameterDict()
        for k, c_out in self.mult_out.items():
            if not c_out:
                continue
            c_in = self.mult_in.get(k, 0)
            if not c_in:
                raise ValueError(f"kind {k!r} of the output has no input channels of that kind (a linear map cannot change the kind)")
            w = torch.zeros(c_out, c_in) if zero_init else torch.randn(c_out, c_in) / math.sqrt(c_in)
            self.weight[k] = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(self.mult_out["0e"])) if bias and self.mult_out.get("0e") else None

    def forward(self, x: Feats) -> Feats:
        out = {}
        for k, w in self.weight.items():
            v = x[k]
            out[k] = F.linear(v, w) if k in G.SCALAR_KINDS else torch.einsum("oi,...ic->...oc", w, v)
        if self.bias is not None:
            out["0e"] = out["0e"] + self.bias
        return out


class TypedRMSNorm(nn.Module):
    """RMS normalisation of every kind with a learned per-channel scale (an even number): "0e" by the RMS of the channels, "0o" the same, the vectors by the
    root of the mean over the channels of the squared lengths. No mean is subtracted and there is no shift. `eps_odd` (for 0o, 1o, 1e) is large on purpose: the vectors
    are all zero before the first structure is available, and the gradient of x / sqrt(mean(x^2) + eps) at 0 is 1 / sqrt(eps)."""

    def __init__(self, mult: Mult, eps_0e: float = 1e-5, eps_odd: float = 1e-3) -> None:
        super().__init__()
        self.eps_0e, self.eps_odd = eps_0e, eps_odd
        self.weight = nn.ParameterDict({k: nn.Parameter(torch.ones(c)) for k, c in mult.items() if c})

    def forward(self, x: Feats) -> Feats:
        out = {}
        for k, v in x.items():
            dt = torch.promote_types(v.dtype, torch.float32)
            u = v.to(dt)
            eps = self.eps_0e if k == "0e" else self.eps_odd
            if k in G.SCALAR_KINDS:
                y = u * torch.rsqrt(u.square().mean(-1, keepdim=True) + eps) * self.weight[k]
            else:
                y = u * torch.rsqrt(u.square().sum(-1).mean(-1, keepdim=True) + eps)[..., None] * self.weight[k][:, None]
            out[k] = y.to(v.dtype)
        return out


class Gate(nn.Module):
    """The only non-linearity of a feature set: SiLU on "0e", and every channel of the other kinds multiplied by sigmoid of a linear map of the "0e" channels (the
    gate is even, so the kind is kept)."""

    def __init__(self, mult: Mult) -> None:
        super().__init__()
        self.others = {k: c for k, c in mult.items() if k != "0e" and c}
        self.gate = nn.Linear(mult["0e"], sum(self.others.values())) if self.others else None

    def forward(self, x: Feats) -> Feats:
        out = {"0e": F.silu(x["0e"])}
        if self.gate is not None:
            g = torch.sigmoid(self.gate(x["0e"])).split(list(self.others.values()), dim=-1)
            for (k, _), gk in zip(self.others.items(), g):
                out[k], kind = G.scale(x[k], k, gk, "0e")
                assert kind == k
        return out


# The allowed products of the projected features a, b (and a third polar vector c for the signed volume), as (kind out, kinds read of a, kinds read of b, reads c, function).
# Every function is a rule of geometry.py: the kind of its result is the one of the first column (checked by the tests).
TERMS = [
    ("0e", ("0e",), ("0e",), False, lambda a, b, c: G.scale(a["0e"], "0e", b["0e"], "0e")),                          # 0e * 0e
    ("0e", ("0o",), ("0o",), False, lambda a, b, c: G.scale(a["0o"], "0o", b["0o"], "0o")),                          # 0o * 0o = 0e
    ("0e", ("1o",), ("1o",), False, lambda a, b, c: G.dot(a["1o"], "1o", b["1o"], "1o")),                            # 1o . 1o = 0e
    ("0e", ("1e",), ("1e",), False, lambda a, b, c: G.dot(a["1e"], "1e", b["1e"], "1e")),                            # 1e . 1e = 0e
    ("0o", ("0o",), ("0e",), False, lambda a, b, c: G.scale(a["0o"], "0o", b["0e"], "0e")),                          # 0o * 0e = 0o
    ("0o", ("1o",), ("1e",), False, lambda a, b, c: G.dot(a["1o"], "1o", b["1e"], "1e")),                            # 1o . 1e = 0o
    ("0o", ("1o",), ("1o",), True, lambda a, b, c: G.triple(a["1o"], "1o", b["1o"], "1o", c, "1o")),                 # the signed volume = 0o
    ("1o", ("0e",), ("1o",), False, lambda a, b, c: G.scale(b["1o"], "1o", a["0e"], "0e")),                          # 0e * 1o = 1o
    ("1o", ("0o",), ("1e",), False, lambda a, b, c: G.scale(b["1e"], "1e", a["0o"], "0o")),                          # 0o * 1e = 1o
    ("1o", ("1o",), ("1e",), False, lambda a, b, c: G.cross(a["1o"], "1o", b["1e"], "1e")),                          # 1o x 1e = 1o
    ("1e", ("0e",), ("1e",), False, lambda a, b, c: G.scale(b["1e"], "1e", a["0e"], "0e")),                          # 0e * 1e = 1e
    ("1e", ("0o",), ("1o",), False, lambda a, b, c: G.scale(b["1o"], "1o", a["0o"], "0o")),                          # 0o * 1o = 1e
    ("1e", ("1o",), ("1o",), False, lambda a, b, c: G.cross(a["1o"], "1o", b["1o"], "1o")),                          # 1o x 1o = 1e
    ("1e", ("1e",), ("1e",), False, lambda a, b, c: G.cross(a["1e"], "1e", b["1e"], "1e")),                          # 1e x 1e = 1e
]
# the order of the 0e-by-vector terms of the first version: the sum over TERMS is what GeometricProduct computes


def select_terms(present, want):
    """the terms whose inputs exist among the kinds `present` and whose output is one of the kinds `want`"""
    return [t for t in TERMS if t[0] in want and set(t[1]) <= set(present) and set(t[2]) <= set(present) and (not t[3] or "1o" in present)]


def producible(present) -> set:
    """the kinds of the products of features of the kinds `present`"""
    return {t[0] for t in select_terms(present, G.KINDS)}


class GeometricProduct(nn.Module):
    """The layer that moves information between the kinds: the products of two (and, for the signed volume, three) projections of the input, `channels` channels each, collected per
    kind and mixed by a TypedLinear into `mult_out`. forward(x) is the update (the caller adds it to the stream).

    Only the products that end in a kind of `mult_out` are built, and only the projections they read (an input kind that no such product reads gets no parameter). The input is
    normalised first (a product of unnormalised features is quadratic in their scale). The kinds that can come out are those of `producible(mult_in)`: with "1o" in the input all four,
    without any vector only "0e" and "0o"."""

    def __init__(self, mult_in: Mult, mult_out: Mult, channels: int = 32, zero_init: bool = False) -> None:
        super().__init__()
        self.mult_in = {k: c for k, c in mult_in.items() if c}
        want = {k for k, c in mult_out.items() if c}
        if not want <= producible(self.mult_in):
            raise ValueError(f"the kinds {sorted(want - producible(self.mult_in))} cannot be made by products of the kinds {sorted(self.mult_in)}")
        self.terms = select_terms(self.mult_in, want)
        used_a = {k for t in self.terms for k in t[1]}
        used_b = {k for t in self.terms for k in t[2]}
        self.norm = TypedRMSNorm(self.mult_in)
        self.proj_a = TypedLinear(self.mult_in, {k: channels for k in used_a})
        self.proj_b = TypedLinear(self.mult_in, {k: channels for k in used_b})
        self.proj_c = TypedLinear(self.mult_in, {"1o": channels}, bias=False) if any(t[3] for t in self.terms) else None
        mult_cat: Dict[str, int] = {}
        for t in self.terms:
            mult_cat[t[0]] = mult_cat.get(t[0], 0) + channels
        self.mult_cat = mult_cat
        self.out = TypedLinear(mult_cat, {k: c for k, c in mult_out.items() if c}, zero_init=zero_init)

    def forward(self, x: Feats) -> Feats:
        x = self.norm(x)
        a, b = self.proj_a(x), self.proj_b(x)
        c = self.proj_c(x)["1o"] if self.proj_c is not None else None
        cat: Dict[str, list] = {}
        for t in self.terms:
            value, kind = t[4](a, b, c)
            assert kind == t[0], (kind, t[0])
            cat.setdefault(kind, []).append(value)
        return self.out({k: torch.cat(v, dim=-1 if k in G.SCALAR_KINDS else -2) for k, v in cat.items()})
