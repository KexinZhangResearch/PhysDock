"""Step 2 of Pomelo: the feature sets and their layers are equivariant under the whole of O(3) (rotations AND reflections), kinds are never mixed by a linear map,
and the checks can fail."""
import pytest
import torch

from pomelo import geometry as G
from pomelo import irreps as I
from pomelo.checks import check_layer

MULT = {"0e": 6, "0o": 4, "1o": 5, "1e": 3}


def test_typed_linear_is_equivariant_and_never_mixes_kinds():
    layer = check_layer(I.TypedLinear(MULT, {"0e": 7, "0o": 2, "1o": 4, "1e": 6}), MULT)
    assert set(layer.weight) == {"0e", "0o", "1o", "1e"}
    assert layer.bias.shape == (7,)                                   # a bias on the "0e" channels only
    assert not any("bias" in n for n, _ in layer.named_parameters() if n != "bias")
    # the output of a kind does not depend on the input of the other kinds
    g = torch.Generator().manual_seed(1)
    x, x2 = I.randn(MULT, (3,), g), I.randn(MULT, (3,), g)
    for kind in MULT:
        mixed = {k: (x[k] if k == kind else x2[k]) for k in MULT}
        a, b = layer(x), layer(mixed)
        for k in MULT:               # the output of kind k changes exactly when the input of kind k does, i.e. when k is not the kept kind
            assert (k != kind) == (not torch.allclose(a[k], b[k])), (k, kind)


def test_a_kind_cannot_be_created_by_a_linear_map():
    with pytest.raises(ValueError):
        I.TypedLinear({"0e": 4}, {"0e": 4, "1o": 2})


def test_the_equivariance_check_catches_a_bias_on_a_vector_and_a_mix_of_kinds():
    layer = I.TypedLinear(MULT, MULT).double()
    layer.vec_bias = torch.nn.Parameter(torch.randn(5, 3, dtype=torch.float64))
    forward = layer.forward
    layer.forward = lambda x: {**forward(x), "1o": forward(x)["1o"] + layer.vec_bias}
    with pytest.raises(AssertionError):
        check_layer(layer, MULT)
    # a "0o" channel added to a "0e" one is invariant under rotations and not under a mirror
    class Mix(torch.nn.Module):
        def forward(self, x):
            return {"0e": x["0e"] + x["0o"]}
    with pytest.raises(AssertionError):
        check_layer(Mix(), {"0e": 4, "0o": 4})


def test_typed_rms_norm_is_equivariant_and_finite_at_zero():
    layer = check_layer(I.TypedRMSNorm(MULT), MULT)
    x = I.zeros(MULT, (3,), dtype=torch.float64)
    for v in x.values():
        v.requires_grad_()
    y = layer(x)
    sum(v.sum() for v in y.values()).backward()
    assert all(torch.isfinite(v).all() for v in y.values()) and all(torch.isfinite(v.grad).all() for v in x.values())
    assert all(float(v.abs().max()) == 0 for k, v in y.items() if k != "0e")           # zero vectors and pseudoscalars stay zero
    assert all(float(v.grad.abs().max()) < 1e3 / 1e-3 ** 0.5 for v in x.values())


def test_gate_is_equivariant_and_only_0e_has_a_non_linearity():
    check_layer(I.Gate(MULT), MULT)
    g = torch.Generator().manual_seed(3)
    x = I.randn(MULT, (4,), g, torch.float32)
    y = I.Gate(MULT)(x)
    for k in ("0o", "1o", "1e"):                                   # the gate is a number in (0, 1) per channel: the direction (and the sign) never change
        ratio = (y[k] / x[k])
        assert float(ratio.min()) > 0 and float(ratio.max()) < 1


@pytest.mark.parametrize("mult_in", [MULT, {"0e": 6, "1o": 5}, {"0e": 6, "0o": 4}, {"1o": 5}, {"0e": 6, "1o": 5, "1e": 3}])
def test_geometric_product_is_equivariant(mult_in):
    out = {k: c for k, c in {"0e": 4, "0o": 3, "1o": 4, "1e": 2}.items() if k in I.producible(mult_in)}          # the kinds that can come out of these inputs
    check_layer(I.GeometricProduct(mult_in, out, channels=5), mult_in)


def test_geometric_product_sees_the_handedness():
    """from three vectors only, the layer can output a pseudoscalar (the signed volume) that changes sign under the inversion and under a mirror, and a rotation does not change"""
    layer = I.GeometricProduct({"1o": 4}, {"0o": 3}, channels=4).double()
    g = torch.Generator().manual_seed(4)
    x = I.randn({"1o": 4}, (6,), g)
    y = layer(x)["0o"]
    assert float(y.abs().max()) > 1e-3
    inv = -torch.eye(3, dtype=torch.float64)
    assert torch.allclose(layer(I.transform(x, inv))["0o"], -y)
    assert torch.allclose(layer(I.transform(x, G.random_orthogonal(1, g)))["0o"], y)
    # without a third vector channel kind there is no way to build a pseudoscalar from 0e and 1o alone except the signed volume: with no vectors none at all
    assert I.producible({"0e": 4, "0o": 2}) == {"0e", "0o"} and I.producible({"1o": 3}) == {"0e", "0o", "1e"}


def test_geometric_product_is_finite_before_the_first_structure():
    """round 0 of the trunk: the vectors are zero; outputs and gradients (also to the weights) are finite"""
    layer = I.GeometricProduct(MULT, {"0e": 5, "0o": 3, "1o": 4, "1e": 2}, channels=6)
    x = I.zeros(MULT, (7,), dtype=torch.float32)
    x["0e"] = torch.randn(7, 6)
    x["0o"] = torch.randn(7, 4)
    y = layer(x)
    sum(v.sum() for v in y.values()).backward()
    assert all(torch.isfinite(v).all() for v in y.values())
    assert all(torch.isfinite(p.grad).all() for p in layer.parameters() if p.grad is not None)


def test_float32_layers_stay_equivariant():
    layer = I.GeometricProduct(MULT, {"0e": 5, "0o": 3, "1o": 4, "1e": 2}, channels=6)
    check_layer(layer, MULT, dtype=torch.float32, atol=2e-4)
