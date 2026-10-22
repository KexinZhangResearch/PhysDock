"""Step 3 of Pomelo: the typed attention of the single representation is O(3)-equivariant (rotations, reflections, translations), obeys the parity selection rule,
does not see the padding, and its direction message is the sum of the relative positions."""
import pytest
import torch

from pomelo import geometry as G
from pomelo import irreps as I
from pomelo.attention import TypedAttention, direction_message
from pomelo.checks import transform_state as transformed

S = {"0e": 16, "0o": 8, "1o": 8, "1e": 4}
Z = {"0e": 6, "0o": 4}
H, N = 4, 7


def inputs(n=N, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    s = I.randn(S, (n,), g, dtype)
    z = I.randn(Z, (n, n), g, dtype)
    c = torch.randn(n, 3, generator=g, dtype=dtype) * 5
    return s, z, c, torch.ones(n, dtype=dtype)


def layer(zero_init=False, seed=0):
    torch.manual_seed(seed)
    return TypedAttention(S, Z, H, zero_init=zero_init).double()


@pytest.mark.parametrize("trial", range(6))
def test_equivariant_under_rotations_reflections_and_translations(trial):
    att = layer()
    g = torch.Generator().manual_seed(10 + trial)
    q = G.random_orthogonal(1 if trial % 2 == 0 else -1, g)
    t = torch.randn(3, generator=g, dtype=torch.float64) * 20
    s, z, c, m = inputs(seed=trial)
    out = att(s, z, c, m)
    s2, z2, c2 = transformed(s, z, c, q, t)
    out2 = att(s2, z2, c2, m)
    for kind in S:
        err = float((G.transform(out[kind], kind, q) - out2[kind]).abs().max())
        assert err < 1e-9, (kind, trial, err)
    assert all(float(v.abs().max()) > 1e-3 for v in out.values()), "every kind of the update is non-trivial"


def test_parity_selection_rule():
    """with no odd input (s_0o, s_1e, z_0o all zero) the odd outputs (0o, 1e) are exactly zero; with odd inputs they are not"""
    att = layer()
    s, z, c, m = inputs()
    even = dict(s), dict(z)
    even[0]["0o"], even[0]["1e"], even[1]["0o"] = torch.zeros_like(s["0o"]), torch.zeros_like(s["1e"]), torch.zeros_like(z["0o"])
    out = att(even[0], even[1], c, m)
    assert float(out["0o"].abs().max()) == 0 and float(out["1e"].abs().max()) == 0
    assert float(out["0e"].abs().max()) > 0 and float(out["1o"].abs().max()) > 0
    out_odd = att(s, z, c, m)
    assert float(out_odd["0o"].abs().max()) > 0 and float(out_odd["1e"].abs().max()) > 0
    # the converse for a chirality-free structure: a pseudoscalar pair state alone creates odd output (the way a stereo label reaches the vectors)
    only_z = dict(even[0]), {"0e": z["0e"], "0o": z["0o"]}
    out_z = att(only_z[0], only_z[1], c, m)
    assert float(out_z["1e"].abs().max()) > 0 and float(out_z["0o"].abs().max()) > 0


def test_a_mirror_image_with_flipped_odd_inputs_gives_the_mirrored_output():
    """the joint mirror test: mirror the structure and flip the sign of the odd inputs (the stereo labels) -> the output is the mirror image; mirroring the structure ALONE is not"""
    att = layer()
    s, z, c, m = inputs()
    mirror = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
    s2, z2, c2 = transformed(s, z, c, mirror, torch.zeros(3, dtype=torch.float64))
    out, out2 = att(s, z, c, m), att(s2, z2, c2, m)
    for kind in S:
        assert torch.allclose(G.transform(out[kind], kind, mirror), out2[kind], atol=1e-10)
    # a mirror image with the odd inputs NOT flipped is a different problem (the other enantiomer is asked for)
    s3 = {"0e": s["0e"], "0o": s["0o"], "1o": s["1o"] @ mirror.T, "1e": s["1e"] @ mirror.T}          # 1e transformed as if it were polar
    out3 = att(s3, {"0e": z["0e"], "0o": z["0o"]}, c @ mirror.T, m)
    assert not torch.allclose(out2["0o"], out3["0o"], atol=1e-6)


def test_direction_message_is_the_weighted_sum_of_the_relative_positions():
    g = torch.Generator().manual_seed(1)
    beta = torch.rand(H, N, N, generator=g, dtype=torch.float64)
    c = torch.randn(N, 3, generator=g, dtype=torch.float64) * 7
    explicit = torch.einsum("hij,ijc->ihc", beta, c[None, :, :] - c[:, None, :])
    assert torch.allclose(direction_message(beta, c), explicit, atol=1e-10)


def test_padding_does_not_change_the_real_units_nor_the_gradients():
    n_real, n_pad = 6, 9
    att = layer()
    s, z, c, _ = inputs(n_pad)
    mask = torch.cat([torch.ones(n_real), torch.zeros(n_pad - n_real)]).double()
    garbage = {k: v.clone() for k, v in s.items()}
    zg, cg = {k: v.clone() for k, v in z.items()}, c.clone()
    for v in garbage.values():
        v[n_real:] = torch.randn_like(v[n_real:]) * 50
    for v in zg.values():
        v[n_real:, :], v[:, n_real:] = torch.randn_like(v[n_real:, :]) * 50, torch.randn_like(v[:, n_real:]) * 50
    cg[n_real:] = 1e3
    clean = ({k: v.clone() for k, v in s.items()}, {k: v.clone() for k, v in z.items()}, c.clone())
    for v in clean[0].values():
        v[n_real:] = 0
    for v in clean[1].values():
        v[n_real:, :], v[:, n_real:] = 0, 0
    clean[2][n_real:] = 0
    unpadded = att({k: v[:n_real] for k, v in clean[0].items()}, {k: v[:n_real, :n_real] for k, v in clean[1].items()}, clean[2][:n_real], mask[:n_real])
    res = {}
    for name, args in (("clean", clean), ("garbage", (garbage, zg, cg))):
        att.zero_grad()
        out = att(*args, mask)
        sum(v.square().sum() for v in out.values()).backward()
        res[name] = (out, {n: p.grad.clone() for n, p in att.named_parameters()})
    for kind in S:
        assert torch.allclose(res["clean"][0][kind][:n_real], unpadded[kind], atol=1e-10)
        assert torch.allclose(res["garbage"][0][kind][:n_real], unpadded[kind], atol=1e-10)
        assert float(res["garbage"][0][kind][n_real:].abs().max()) == 0
    for n in res["clean"][1]:
        assert torch.allclose(res["garbage"][1][n], res["clean"][1][n], atol=1e-9), n


def test_every_parameter_gets_a_gradient():
    att = layer()
    s, z, c, m = inputs()
    sum(v.square().sum() for v in att(s, z, c, m).values()).backward()
    dead = [n for n, p in att.named_parameters() if p.grad is None or float(p.grad.abs().max()) == 0]
    assert not dead, dead


def test_first_round_without_a_structure_is_finite():
    """no coordinates yet: vectors and centres are zero"""
    att = layer().float()
    s, z, c, m = inputs(dtype=torch.float32)
    s["1o"], s["1e"], c = torch.zeros_like(s["1o"]), torch.zeros_like(s["1e"]), torch.zeros_like(c)
    out = att(s, z, c, m)
    sum(v.square().sum() for v in out.values()).backward()
    assert all(torch.isfinite(v).all() for v in out.values())
    assert all(torch.isfinite(p.grad).all() for p in att.parameters() if p.grad is not None)
    assert float(out["1o"].abs().max()) == 0 and float(out["1e"].abs().max()) == 0           # nothing to point along yet
