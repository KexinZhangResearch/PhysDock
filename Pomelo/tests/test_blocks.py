"""Step 4 of Pomelo: the typed transition and the single block are O(3)-equivariant, keep the parity rules sublayer by sublayer, do not see the padding."""
import pytest
import torch

from pomelo import geometry as G
from pomelo import irreps as I
from pomelo.blocks import SingleBlock, TypedTransition
from pomelo.checks import check_layer, transform_state

S = {"0e": 16, "0o": 8, "1o": 8, "1e": 4}
Z = {"0e": 6, "0o": 4}
H, N = 4, 7


def state(n=N, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return I.randn(S, (n,), g, dtype), I.randn(Z, (n, n), g, dtype), torch.randn(n, 3, generator=g, dtype=dtype) * 5, torch.ones(n, dtype=dtype)


def block(**kw):
    torch.manual_seed(0)
    return SingleBlock(S, Z, H, product_channels=6, **kw).double()


def test_transition_is_equivariant_and_the_even_path_is_closed():
    layer = check_layer(TypedTransition(S), S)
    g = torch.Generator().manual_seed(1)
    x, x2 = I.randn(S, (3,), g), I.randn(S, (3,), g)
    mixed = {"0e": x["0e"], "0o": x2["0o"], "1o": x2["1o"], "1e": x2["1e"]}
    a, b = layer(x), layer(mixed)
    assert torch.equal(a["0e"], b["0e"])                                      # the 0e output is the SwiGLU of the 0e channels: nothing else enters
    assert all(not torch.allclose(a[k], b[k]) for k in ("0o", "1o", "1e"))


def test_transition_parity_selection_rule():
    layer = TypedTransition(S).double()
    g = torch.Generator().manual_seed(2)
    x = I.randn(S, (4,), g)
    x["0o"], x["1e"] = torch.zeros_like(x["0o"]), torch.zeros_like(x["1e"])
    y = layer(x)
    assert float(y["0o"].abs().max()) == 0 and float(y["1e"].abs().max()) == 0
    assert float(y["0e"].abs().max()) > 0 and float(y["1o"].abs().max()) > 0


@pytest.mark.parametrize("trial", range(6))
def test_block_is_equivariant(trial):
    layer = block()
    g = torch.Generator().manual_seed(20 + trial)
    q = G.random_orthogonal(1 if trial % 2 == 0 else -1, g)
    t = torch.randn(3, generator=g, dtype=torch.float64) * 20
    s, z, c, m = state(seed=trial)
    out = layer(s, z, c, m)
    out2 = layer(*transform_state(s, z, c, q, t), m)
    for kind in S:
        err = float((G.transform(out[kind], kind, q) - out2[kind]).abs().max())
        assert err < 1e-9, (kind, trial, err)


def test_three_blocks_in_a_row_stay_equivariant():
    layers = [block() for _ in range(3)]
    for i, l in enumerate(layers):
        torch.manual_seed(100 + i)
        for p in l.parameters():
            p.data.normal_(0, 0.3) if p.dim() > 1 else None
    g = torch.Generator().manual_seed(3)
    q, t = G.random_orthogonal(-1, g), torch.randn(3, generator=g, dtype=torch.float64)
    s, z, c, m = state()
    s2, z2, c2 = transform_state(s, z, c, q, t)
    for l in layers:
        s, s2 = l(s, z, c, m), l(s2, z2, c2, m)
    for kind in S:
        assert float((G.transform(s[kind], kind, q) - s2[kind]).abs().max()) < 1e-7


def test_the_geometric_products_are_where_the_geometry_creates_odd_features():
    """with only even features, and no odd pair state, the attention and the transition keep it that way; the product layer alone makes a pseudoscalar out of three vectors"""
    layer = block()
    s, z, c, m = state()
    for k in ("0o", "1e"):
        s[k] = torch.zeros_like(s[k])
    z["0o"] = torch.zeros_like(z["0o"])
    assert all(float(v.abs().max()) == 0 for k, v in layer.attention(s, z, c, m).items() if k in ("0o", "1e"))
    assert all(float(v.abs().max()) == 0 for k, v in layer.transition(s).items() if k in ("0o", "1e"))
    assert float(layer.product(s)["0o"].abs().max()) > 0
    # ... and without any vector the products cannot, whatever the pair state
    s_scalar = {k: (v if k in ("0e", "0o") else torch.zeros_like(v)) for k, v in s.items()}
    s_scalar["0o"] = torch.zeros_like(s_scalar["0o"])
    assert float(layer.product(s_scalar)["0o"].abs().max()) == 0
    assert float(layer.product(s_scalar)["1e"].abs().max()) == 0


def test_joint_mirror():
    """mirror the structure and flip the odd inputs -> the mirror image of the output"""
    layer = block()
    s, z, c, m = state()
    mirror = torch.diag(torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64))
    out = layer(s, z, c, m)
    out2 = layer(*transform_state(s, z, c, mirror, torch.zeros(3, dtype=torch.float64)), m)
    for kind in S:
        assert torch.allclose(G.transform(out[kind], kind, mirror), out2[kind], atol=1e-9)


def test_padding_is_invisible():
    n_real, n_pad = 5, 8
    layer = block()
    s, z, c, _ = state(n_pad)
    mask = torch.cat([torch.ones(n_real), torch.zeros(n_pad - n_real)]).double()
    clean_s = {k: v.clone() for k, v in s.items()}
    for v in clean_s.values():
        v[n_real:] = 0
    clean_z = {k: v.clone() for k, v in z.items()}
    for v in clean_z.values():
        v[n_real:, :], v[:, n_real:] = 0, 0
    clean_c = c.clone()
    clean_c[n_real:] = 0
    dirty_s = {k: v.clone() for k, v in s.items()}
    for v in dirty_s.values():
        v[n_real:] = torch.randn_like(v[n_real:]) * 40
    dirty_z = {k: v.clone() for k, v in z.items()}
    for v in dirty_z.values():
        v[n_real:, :], v[:, n_real:] = torch.randn_like(v[n_real:, :]) * 40, torch.randn_like(v[:, n_real:]) * 40
    dirty_c = c.clone()
    dirty_c[n_real:] = 1e3
    ref = layer({k: v[:n_real] for k, v in clean_s.items()}, {k: v[:n_real, :n_real] for k, v in clean_z.items()}, clean_c[:n_real], mask[:n_real])
    grads = {}
    for name, args in (("clean", (clean_s, clean_z, clean_c)), ("dirty", (dirty_s, dirty_z, dirty_c))):
        layer.zero_grad()
        out = layer(*args, mask)
        sum(v.square().sum() for v in out.values()).backward()
        grads[name] = {n: p.grad.clone() for n, p in layer.named_parameters()}
        for k in S:
            assert torch.allclose(out[k][:n_real], ref[k], atol=1e-9) and float(out[k][n_real:].abs().max()) == 0
    for n in grads["clean"]:
        assert torch.allclose(grads["dirty"][n], grads["clean"][n], atol=1e-8), n


def test_every_parameter_gets_a_gradient_and_the_first_round_is_finite():
    layer = block()
    s, z, c, m = state()
    sum(v.square().sum() for v in layer(s, z, c, m).values()).backward()
    assert not [n for n, p in layer.named_parameters() if p.grad is None or float(p.grad.abs().max()) == 0]
    layer = block().float()
    s, z, c, m = state(dtype=torch.float32)
    s["1o"], s["1e"], c = torch.zeros_like(s["1o"]), torch.zeros_like(s["1e"]), torch.zeros_like(c)
    out = layer(s, z, c, m)
    sum(v.square().sum() for v in out.values()).backward()
    assert all(torch.isfinite(v).all() for v in out.values())
    assert all(torch.isfinite(p.grad).all() for p in layer.parameters() if p.grad is not None)
