"""Typed geometric features and the products between them.

Under an orthogonal matrix Q (a rotation, det Q = +1, or a reflection / improper rotation, det Q = -1) the four kinds of feature transform as

    kind   name           transforms as            shape
    "0e"   scalar         x                        [..., C]
    "0o"   pseudoscalar   det(Q) * x               [..., C]
    "1o"   (polar) vector Q x                      [..., C, 3]
    "1e"   axial vector   det(Q) * Q x             [..., C, 3]

(the names are those of e3nn: the digit is the order of the representation, 0 or 1, and "e" / "o" its parity, even or odd under the inversion x -> -x).
Positions, displacements and forces are "1o"; the cross product of two "1o" is a "1e"; the triple product of three "1o" is a "0o"; distances and dot
products of two "1o" are "0e". The kinds say how a quantity transforms and not where it lives: single, pair and atom representations can hold any of them.
A layer built only from the operations below is equivariant under the whole of O(3), so mirroring the input mirrors the output, by construction. A
network that is allowed to mix kinds (for instance a linear layer over the channels of "0e" and "0o" together) is only SO(3)-invariant and has to learn the
parity. `check_equivariance` is the test every new operation has to pass.
"""
import itertools
from typing import Callable, Sequence

import torch

KINDS = ("0e", "0o", "1o", "1e")
SCALAR_KINDS = ("0e", "0o")
VECTOR_KINDS = ("1o", "1e")


def transform(x: torch.Tensor, kind: str, q: torch.Tensor) -> torch.Tensor:
    """x transformed by the orthogonal matrix q [3, 3] (x of kind `kind`, channels in dim -1 for scalars, (-2, -1) for vectors)"""
    det = torch.det(q).round()
    if kind == "0e":
        return x
    if kind == "0o":
        return det * x
    if kind == "1o":
        return x @ q.T
    if kind == "1e":
        return det * (x @ q.T)
    raise ValueError(f"unknown kind {kind!r}, expected one of {KINDS}")


def _sign(kind_a: str, kind_b: str) -> str:
    """kind of a scalar-like product: parity multiplies (0e * 0e = 0e, 0o * 0o = 0e, 0e * 0o = 0o)"""
    return "0o" if (kind_a == "0o") != (kind_b == "0o") else "0e"


# ---------------------------------------------------------------------------------------------------------- products
def dot(u: torch.Tensor, ku: str, w: torch.Tensor, kw: str):
    """channelwise dot product of two vectors [..., C, 3] -> [..., C]; the result is a scalar if the two are of the same kind (1o.1o, 1e.1e), a pseudoscalar otherwise (1o.1e)"""
    _require(ku in VECTOR_KINDS and kw in VECTOR_KINDS, "dot", ku, kw)
    return (u * w).sum(-1), ("0e" if ku == kw else "0o")


def cross(u: torch.Tensor, ku: str, w: torch.Tensor, kw: str):
    """channelwise cross product [..., C, 3] x [..., C, 3] -> [..., C, 3]. Under Q, Qu x Qw = det(Q) Q (u x w), which gives the kind of the result:
    1o x 1o and 1e x 1e are axial vectors (1e), 1o x 1e is a polar vector (1o)"""
    _require(ku in VECTOR_KINDS and kw in VECTOR_KINDS, "cross", ku, kw)
    out = "1e" if ku == kw else "1o"
    return torch.linalg.cross(u, w, dim=-1), out


def triple(u: torch.Tensor, ku: str, v: torch.Tensor, kv: str, w: torch.Tensor, kw: str):
    """u . (v x w), channelwise: the signed volume of three polar vectors is a pseudoscalar (0o)"""
    c, kc = cross(v, kv, w, kw)
    return dot(u, ku, c, kc)


def scale(x: torch.Tensor, kx: str, g: torch.Tensor, kg: str):
    """a feature times a scalar-like feature of the same channels: g [..., C] (kind "0e" or "0o"), x [..., C] or [..., C, 3]; the kinds multiply
    (0o * 1o = 1e, 0o * 1e = 1o, 0o * 0o = 0e, ...). With g a gate computed from "0e" features only (a sigmoid, an MLP) the kind of x is unchanged"""
    _require(kg in SCALAR_KINDS, "scale", kx, kg)
    g = g[..., None] if x.dim() == g.dim() + 1 else g
    if kx in SCALAR_KINDS:
        return x * g, _sign(kx, kg)
    flip = (kg == "0o")
    return x * g, (({"1o": "1e", "1e": "1o"})[kx] if flip else kx)


def norm(u: torch.Tensor, ku: str, eps: float = 1e-12):
    """length of a vector [..., C, 3] -> [..., C] (a "0e" scalar for either vector kind; the eps keeps the gradient finite at 0)"""
    _require(ku in VECTOR_KINDS, "norm", ku, ku)
    return (u.square().sum(-1) + eps).sqrt(), "0e"


def _require(cond: bool, op: str, ka: str, kb: str) -> None:
    if not cond:
        raise TypeError(f"{op} is not defined for the kinds {ka!r} and {kb!r}")


# ---------------------------------------------------------------------------------------------------------- the test of an operation
def random_orthogonal(det: int, generator: torch.Generator = None, dtype=torch.float64) -> torch.Tensor:
    """a random orthogonal [3, 3] matrix with determinant det (+1: rotation, -1: reflection / improper rotation)"""
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=dtype))
    q = q * torch.sign(torch.diagonal(r))
    if round(float(torch.det(q))) != det:
        q[:, 0] = -q[:, 0]
    return q


def random_feature(kind: str, shape: Sequence[int], channels: int, generator: torch.Generator = None, dtype=torch.float64) -> torch.Tensor:
    size = (*shape, channels) if kind in SCALAR_KINDS else (*shape, channels, 3)
    return torch.randn(*size, generator=generator, dtype=dtype)


def check_equivariance(fn: Callable, in_kinds: Sequence[str], shape=(4, 5), channels: int = 3, trials: int = 8, atol: float = 1e-10, seed: int = 0):
    """The O(3) test of an operation fn(*features) -> (feature, out_kind) or a list of such pairs: for random rotations and reflections Q and random inputs,
    fn(Q x_1, ..., Q x_n) == Q fn(x_1, ..., x_n) with every input and the output transformed by the rule of its kind. Returns the output kinds; raises
    AssertionError with the worst deviation otherwise. A reflection that is treated as if it were a rotation is caught by the "0o" and "1e" rules."""
    g = torch.Generator().manual_seed(seed)
    kinds_out = None
    for trial in range(trials):
        q = random_orthogonal(1 if trial % 2 == 0 else -1, g)
        xs = [random_feature(k, shape, channels, g) for k in in_kinds]
        out = fn(*xs)
        outs = [out] if isinstance(out, tuple) and isinstance(out[1], str) else list(out)
        out_q = fn(*[transform(x, k, q) for x, k in zip(xs, in_kinds)])
        outs_q = [out_q] if isinstance(out_q, tuple) and isinstance(out_q[1], str) else list(out_q)
        kinds_out = [k for _, k in outs]
        for (y, k), (yq, kq) in zip(outs, outs_q):
            assert k == kq, (k, kq)
            err = float((transform(y, k, q) - yq).abs().max())
            assert err < atol, f"kind {k}, det {round(float(torch.det(q)))}: deviation {err:.3e}"
    return kinds_out


def all_kind_pairs(vector_only: bool = True):
    kinds = VECTOR_KINDS if vector_only else KINDS
    return list(itertools.product(kinds, kinds))
