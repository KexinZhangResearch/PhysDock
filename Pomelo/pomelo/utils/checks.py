"""The O(3) checks every layer of Pomelo has to pass (used by the tests, and usable on any new layer)."""
import torch

from pomelo.models.primitives import geometry as G
from pomelo.models.primitives import irreps as I


def check_layer(layer, mult_in, shape=(4, 5), trials=8, dtype=torch.float64, atol=1e-9, seed=0):
    """layer(x) is equivariant: layer(Q x) == Q layer(x) for random rotations (even trials) and reflections (odd trials), the output of every kind transformed by its rule"""
    layer = layer.to(dtype)
    g = torch.Generator().manual_seed(seed)
    for trial in range(trials):
        q = G.random_orthogonal(1 if trial % 2 == 0 else -1, g, dtype)
        x = I.randn(mult_in, shape, g, dtype)
        y, yq = layer(x), layer(I.transform(x, q))
        assert set(y) == set(yq)
        for k in y:
            err = float((G.transform(y[k], k, q) - yq[k]).abs().max())
            assert err < atol, f"kind {k}, det {round(float(torch.det(q)))}: deviation {err:.2e}"
    return layer


def transform_state(s, z, centers, q, t):
    """the inputs of a trunk layer after the orthogonal transformation q and the translation t: the single representation by the rules of its kinds, the stored pair
    representation (0e, 0o) by its kinds, the centres of the units as points (Q c + t)"""
    return I.transform(s, q), {"0e": z["0e"], "0o": G.transform(z["0o"], "0o", q)}, centers @ q.T + t
