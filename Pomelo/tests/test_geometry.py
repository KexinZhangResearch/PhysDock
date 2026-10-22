"""Step 1 of Pomelo: the typed features and their products are equivariant under the whole of O(3) (rotations and reflections), and the test itself can fail."""
import itertools

import pytest
import torch

from pomelo import geometry as G


@pytest.mark.parametrize("ku,kw", G.all_kind_pairs())
def test_dot_and_cross_of_every_pair_of_vector_kinds(ku, kw):
    G.check_equivariance(lambda u, w: G.dot(u, ku, w, kw), [ku, kw])
    G.check_equivariance(lambda u, w: G.cross(u, ku, w, kw), [ku, kw])


@pytest.mark.parametrize("kinds", list(itertools.product(G.VECTOR_KINDS, repeat=3)))
def test_triple_product(kinds):
    out = G.check_equivariance(lambda u, v, w: G.triple(u, kinds[0], v, kinds[1], w, kinds[2]), list(kinds))
    assert len(out) == 1


@pytest.mark.parametrize("kx", G.KINDS)
@pytest.mark.parametrize("kg", ["0e", "0o"])
def test_scale_by_a_scalar_like_feature(kx, kg):
    G.check_equivariance(lambda x, g: G.scale(x, kx, g, kg), [kx, kg])


@pytest.mark.parametrize("ku", G.VECTOR_KINDS)
def test_norm(ku):
    assert G.check_equivariance(lambda u: G.norm(u, ku), [ku]) == ["0e"]


def test_the_known_kinds():
    """the textbook facts the rules encode"""
    u = torch.randn(2, 3, 3, dtype=torch.float64)
    w = torch.randn(2, 3, 3, dtype=torch.float64)
    assert G.dot(u, "1o", w, "1o")[1] == "0e"            # distance, angle
    assert G.cross(u, "1o", w, "1o")[1] == "1e"          # angular momentum, a surface normal
    assert G.triple(u, "1o", w, "1o", u + w, "1o")[1] == "0o"          # the signed volume of a chirality centre
    assert G.cross(u, "1o", w, "1e")[1] == "1o"
    assert G.dot(u, "1o", w, "1e")[1] == "0o"


def test_the_signed_volume_flips_under_a_mirror_and_not_under_a_rotation():
    g = torch.Generator().manual_seed(1)
    u, v, w = (G.random_feature("1o", (6,), 1, g) for _ in range(3))
    vol = G.triple(u, "1o", v, "1o", w, "1o")[0]
    rot, mirror = G.random_orthogonal(1, g), G.random_orthogonal(-1, g)
    t = lambda x, q: G.transform(x, "1o", q)
    assert torch.allclose(G.triple(t(u, rot), "1o", t(v, rot), "1o", t(w, rot), "1o")[0], vol)
    assert torch.allclose(G.triple(t(u, mirror), "1o", t(v, mirror), "1o", t(w, mirror), "1o")[0], -vol)


def test_the_check_catches_a_wrong_kind():
    """an operation that is declared with the wrong kind (the cross product of two polar vectors called polar) must fail under a reflection"""
    wrong = lambda u, w: (torch.linalg.cross(u, w, dim=-1), "1o")
    with pytest.raises(AssertionError):
        G.check_equivariance(wrong, ["1o", "1o"])
    # and a rotation alone would not see it
    g = torch.Generator().manual_seed(0)
    q = G.random_orthogonal(1, g)
    u, w = G.random_feature("1o", (4,), 3, g), G.random_feature("1o", (4,), 3, g)
    y = wrong(u, w)[0]
    assert torch.allclose(G.transform(y, "1o", q), wrong(G.transform(u, "1o", q), G.transform(w, "1o", q))[0])


def test_a_layer_that_mixes_kinds_is_not_o3_equivariant():
    """why the types are kept apart: a linear map over the channels of a scalar and a pseudoscalar together is invariant under rotations but not under mirrors"""
    g = torch.Generator().manual_seed(2)
    w = torch.randn(2, 2, generator=g, dtype=torch.float64)
    s, p = G.random_feature("0e", (5,), 1, g), G.random_feature("0o", (5,), 1, g)
    mix = lambda s, p: torch.cat([s, p], -1) @ w
    mirror = G.random_orthogonal(-1, g)
    before = mix(s, p)
    after = mix(G.transform(s, "0e", mirror), G.transform(p, "0o", mirror))
    assert not torch.allclose(before, after)
