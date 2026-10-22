"""Step 5 of Pomelo: the pair layers and the trunk block are O(3)-equivariant (rotations, reflections, translations), obey the parity rules, do not see the padding, and the
handedness of a triangle reaches the odd pair channel."""
import pytest
import torch

from pomelo import geometry as G
from pomelo import irreps as I
from pomelo.checks import transform_state
from pomelo.pair import PairBlock, PairFromSingle, TypedTriangleAttention, TypedTriangleMultiplication, directions
from pomelo.trunk import TrunkBlock

S = {"0e": 16, "0o": 8, "1o": 8, "1e": 4}
Z = {"0e": 16, "0o": 8}
N = 6


def state(n=N, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return I.randn(S, (n,), g, dtype), I.randn(Z, (n, n), g, dtype), torch.randn(n, 3, generator=g, dtype=dtype) * 5, torch.ones(n, dtype=dtype)


def make(kind, seed=0):
    torch.manual_seed(seed)
    return {"outer": lambda: PairFromSingle(S, Z, channels=6), "mul_out": lambda: TypedTriangleMultiplication(S, Z, True, channels=6),
            "mul_in": lambda: TypedTriangleMultiplication(S, Z, False, channels=6), "att_start": lambda: TypedTriangleAttention(Z, 4, True),
            "att_end": lambda: TypedTriangleAttention(Z, 4, False), "pair": lambda: PairBlock(S, Z, 4, channels=6),
            "trunk": lambda: TrunkBlock(S, Z, 4, 4, channels=6, product_channels=6)}[kind]().double()


def apply(kind, layer, s, z, c, m):
    """every layer called the same way; returns the pair update (and the single output for the trunk)"""
    if kind == "outer":
        return layer(s, c, m)
    if kind.startswith("mul"):
        return layer(s, z, c, m)
    if kind.startswith("att"):
        return layer(z, m)
    if kind == "pair":
        return layer(s, z, c, m)
    return layer(s, z, c, m)


def kinds_of(out):
    """(single outputs or {}, pair outputs) of a layer"""
    if isinstance(out, tuple):
        return out
    return {}, out


def check(kind, trials=6):
    layer = make(kind)
    for trial in range(trials):
        g = torch.Generator().manual_seed(30 + trial)
        q = G.random_orthogonal(1 if trial % 2 == 0 else -1, g)
        t = torch.randn(3, generator=g, dtype=torch.float64) * 20
        s, z, c, m = state(seed=trial)
        so, zo = kinds_of(apply(kind, layer, s, z, c, m))
        so2, zo2 = kinds_of(apply(kind, layer, *transform_state(s, z, c, q, t), m))
        for k, v in so.items():
            assert float((G.transform(v, k, q) - so2[k]).abs().max()) < 1e-8, (kind, "single", k, trial)
        for k, v in zo.items():
            assert float((G.transform(v, k, q) - zo2[k]).abs().max()) < 1e-8, (kind, "pair", k, trial)
        assert all(float(v.abs().max()) > 1e-4 for v in zo.values()), (kind, "the update is not trivial")


@pytest.mark.parametrize("kind", ["outer", "mul_out", "mul_in", "att_start", "att_end", "pair", "trunk"])
def test_equivariant_under_rotations_reflections_and_translations(kind):
    check(kind)


def test_directions_are_translation_invariant_and_bounded():
    g = torch.Generator().manual_seed(0)
    c = torch.randn(N, 3, generator=g, dtype=torch.float64) * 30
    valid = torch.ones(N, dtype=torch.bool)
    w = directions(c, valid, 8.0)
    assert torch.allclose(w, directions(c + 100.0, valid, 8.0), atol=1e-12)
    assert float(w.norm(dim=-1).max()) < 1 and torch.allclose(w, -w.transpose(0, 1))        # antisymmetric, the diagonal is 0
    assert float(w.diagonal(dim1=0, dim2=1).abs().max()) == 0


def test_triangle_handedness_reaches_the_odd_channel():
    """no odd input at all (s_0o, s_1e, z_0o zero) and still a pseudoscalar comes out of the triangle: the signed volume of w_ik, w_jk and the out-of-plane u_k. Without u (no vectors in
    s) the three directions of a triangle are coplanar and nothing odd can come out."""
    s, z, c, m = state()
    for k in ("0o", "1e"):
        s[k] = torch.zeros_like(s[k])
    z["0o"] = torch.zeros_like(z["0o"])
    for outgoing in (True, False):
        torch.manual_seed(0)
        layer = TypedTriangleMultiplication(S, Z, outgoing, channels=6).double()
        assert float(layer(s, z, c, m)["0o"].abs().max()) > 1e-6
        s_flat = dict(s)
        s_flat["1o"] = torch.zeros_like(s["1o"])
        assert float(layer(s_flat, z, c, m)["0o"].abs().max()) == 0
    # and a planar arrangement of the centres (all in one plane) with u in that plane has no signed volume either
    c_plane = c.clone()
    c_plane[:, 2] = 0
    s_plane = dict(s)
    s_plane["1o"] = s["1o"].clone()
    s_plane["1o"][..., 2] = 0
    assert float(layer(s_plane, z, c_plane, m)["0o"].abs().max()) < 1e-10


def test_mirror_image_with_flipped_odd_inputs_is_mirrored_for_the_whole_trunk_block():
    layer = make("trunk")
    s, z, c, m = state()
    mirror = torch.diag(torch.tensor([1.0, 1.0, -1.0], dtype=torch.float64))
    so, zo = layer(s, z, c, m)
    s2, z2, c2 = transform_state(s, z, c, mirror, torch.zeros(3, dtype=torch.float64))
    so2, zo2 = layer(s2, z2, c2, m)
    for k in S:
        assert torch.allclose(G.transform(so[k], k, mirror), so2[k], atol=1e-9)
    for k in Z:
        assert torch.allclose(G.transform(zo[k], k, mirror), zo2[k], atol=1e-9)
    # the structure mirrored ALONE (the stereo labels unchanged) does not give the mirror image: the layer knows the hand of the geometry
    s3 = {"0e": s["0e"], "0o": s["0o"], "1o": s["1o"] @ mirror.T, "1e": G.transform(s["1e"], "1e", torch.eye(3, dtype=torch.float64))}
    s3["1e"] = s["1e"] @ mirror.T * (-1)
    so3, zo3 = layer(s3, {"0e": z["0e"], "0o": z["0o"]}, c @ mirror.T, m)
    assert not torch.allclose(zo2["0o"], zo3["0o"], atol=1e-6)


@pytest.mark.parametrize("kind", ["outer", "mul_out", "att_start", "pair", "trunk"])
def test_padding_is_invisible(kind):
    n_real, n_pad = 4, 7
    layer = make(kind)
    s, z, c, _ = state(n_pad)
    mask = torch.cat([torch.ones(n_real), torch.zeros(n_pad - n_real)]).double()

    def padded(fill):
        sp, zp, cp = {k: v.clone() for k, v in s.items()}, {k: v.clone() for k, v in z.items()}, c.clone()
        for v in sp.values():
            v[n_real:] = fill(v[n_real:])
        for v in zp.values():
            v[n_real:, :], v[:, n_real:] = fill(v[n_real:, :]), fill(v[:, n_real:])
        cp[n_real:] = fill(cp[n_real:])
        return sp, zp, cp
    clean, dirty = padded(torch.zeros_like), padded(lambda t: torch.randn_like(t) * 40)
    ref = kinds_of(apply(kind, layer, {k: v[:n_real] for k, v in clean[0].items()}, {k: v[:n_real, :n_real] for k, v in clean[1].items()}, clean[2][:n_real], mask[:n_real]))
    grads = {}
    for name, args in (("clean", clean), ("dirty", dirty)):
        layer.zero_grad()
        so, zo = kinds_of(apply(kind, layer, *args, mask))
        sum(v.square().sum() for v in (*so.values(), *zo.values())).backward()
        grads[name] = {n: p.grad.clone() for n, p in layer.named_parameters() if p.grad is not None}
        for k, v in so.items():
            assert torch.allclose(v[:n_real], ref[0][k], atol=1e-8) and float(v[n_real:].abs().max()) == 0
        for k, v in zo.items():
            assert torch.allclose(v[:n_real, :n_real], ref[1][k], atol=1e-8)
            assert float(v[n_real:].abs().max()) == 0 and float(v[:, n_real:].abs().max()) == 0
    for n in grads["clean"]:
        assert torch.allclose(grads["dirty"][n], grads["clean"][n], atol=1e-7), (kind, n)


@pytest.mark.parametrize("kind", ["outer", "mul_out", "mul_in", "att_start", "att_end", "pair", "trunk"])
def test_every_parameter_gets_a_gradient(kind):
    layer = make(kind)
    s, z, c, m = state()
    so, zo = kinds_of(apply(kind, layer, s, z, c, m))
    sum(v.square().sum() for v in (*so.values(), *zo.values())).backward()
    dead = [n for n, p in layer.named_parameters() if p.grad is None or float(p.grad.abs().max()) == 0]
    assert not dead, dead


def test_first_round_without_a_structure_is_finite():
    """no coordinates: the centres and the vectors are zero (the pair layers see only the scalars)"""
    layer = make("trunk").float()
    s, z, c, m = state(dtype=torch.float32)
    s["1o"], s["1e"], c = torch.zeros_like(s["1o"]), torch.zeros_like(s["1e"]), torch.zeros_like(c)
    so, zo = layer(s, z, c, m)
    sum(v.square().sum() for v in (*so.values(), *zo.values())).backward()
    assert all(torch.isfinite(v).all() for v in (*so.values(), *zo.values()))
    assert all(torch.isfinite(p.grad).all() for p in layer.parameters() if p.grad is not None)


def test_three_trunk_blocks_stay_equivariant():
    blocks = [make("trunk", seed=i) for i in range(3)]
    g = torch.Generator().manual_seed(7)
    q, t = G.random_orthogonal(-1, g), torch.randn(3, generator=g, dtype=torch.float64)
    s, z, c, m = state()
    s2, z2, c2 = transform_state(s, z, c, q, t)
    for b in blocks:
        s, z = b(s, z, c, m)
        s2, z2 = b(s2, z2, c2, m)
    for k in S:
        assert float((G.transform(s[k], k, q) - s2[k]).abs().max()) < 1e-6
    for k in Z:
        assert float((G.transform(z[k], k, q) - z2[k]).abs().max()) < 1e-6


def reference_triangle(layer, s, z, c, outgoing):
    """the triangle multiplication written with loops over i, j, k from the definitions (the edge a_ik / b_jk of the outgoing, a_ki / b_kj of the incoming update, the
    direction of an edge from its row to its column node, u at the shared node k), to check the indexing and the transposition of the vectorised layer"""
    n = c.shape[0]
    valid = torch.ones(n, dtype=torch.bool)
    zn = layer.norm_z(z)
    wh = directions(c, valid, layer.length_scale)
    u = layer.proj_u(layer.norm_u({"1o": s["1o"]}))["1o"]

    def side(r, t, scalar, gate, ce, co=None, cx=None, node=None):
        zr = {k: v[r, t][None] for k, v in zn.items()}
        g, sc = torch.sigmoid(gate(zr["0e"]))[0], {k: v[0] for k, v in scalar(zr).items()}
        v1o = wh[r, t][None, :] * ce(zr["0e"])[0][:, None]
        v1e = (wh[r, t][None, :] * torch.tanh(co(zr["0o"]))[0][:, None]) if co is not None else torch.linalg.cross(wh[r, t][None, :].expand_as(u[node]), u[node]) * cx(zr["0e"])[0][:, None]
        return sc["0e"] * g, sc["0o"] * g, v1o, v1e
    cat0e, cat0o = torch.zeros(n, n, 4 * layer.a_ce.out_features, dtype=z["0e"].dtype), torch.zeros(n, n, 4 * layer.a_ce.out_features, dtype=z["0e"].dtype)
    for i in range(n):
        for j in range(n):
            t0e = t0o = 0
            for k in range(n):
                ra, rb = ((i, k), (j, k)) if outgoing else ((k, i), (k, j))
                a0e, a0o, a1o, a1e = side(*ra, layer.a_scalar, layer.a_gate, layer.a_ce, co=layer.a_co)
                b0e, b0o, b1o, b1e = side(*rb, layer.b_scalar, layer.b_gate, layer.b_ce, cx=layer.b_cx, node=k)
                dot = lambda x, y: (x * y).sum(-1)
                t0e = t0e + torch.cat([a0e * b0e, a0o * b0o, dot(a1o, b1o), dot(a1e, b1e)])
                t0o = t0o + torch.cat([a0e * b0o, a0o * b0e, dot(a1o, b1e), dot(a1e, b1o)])
            cat0e[i, j], cat0o[i, j] = t0e, t0o
    y = layer.lin_out(layer.norm_out({"0e": cat0e, "0o": cat0o}))
    gates = torch.sigmoid(layer.gate_out(zn["0e"])).split(layer.split, dim=-1)
    return {k: G.scale(v, k, g, "0e")[0] for (k, v), g in zip(y.items(), gates)}


@pytest.mark.parametrize("outgoing", [True, False])
def test_triangle_multiplication_matches_the_loop_reference(outgoing):
    torch.manual_seed(0)
    layer = TypedTriangleMultiplication(S, Z, outgoing, channels=3).double()
    s, z, c, m = state(n=5, seed=11)
    out = layer(s, z, c, m)
    ref = reference_triangle(layer, s, z, c, outgoing)
    for k in Z:
        assert torch.allclose(out[k], ref[k], atol=1e-10), (outgoing, k, float((out[k] - ref[k]).abs().max()))


def test_ending_node_attention_is_the_starting_node_on_the_transposed_pair():
    torch.manual_seed(0)
    start, end = TypedTriangleAttention(Z, 4, True).double(), TypedTriangleAttention(Z, 4, False).double()
    end.load_state_dict(start.state_dict())
    _, z, _, m = state()
    a = end(z, m)
    b = start({k: v.transpose(0, 1) for k, v in z.items()}, m)
    assert all(torch.allclose(a[k], b[k].transpose(0, 1), atol=1e-12) for k in Z)
    assert not torch.allclose(start(z, m)["0e"], a["0e"], atol=1e-6)                 # and they are different functions


def test_triangle_attention_only_reads_the_row_and_the_column_of_an_entry():
    """perturbing z[a, b] changes the outputs of row a (queries, keys, values) and of column a (the bias of the edge a b), nothing else"""
    torch.manual_seed(0)
    layer = TypedTriangleAttention(Z, 4, True).double()
    _, z, _, m = state()
    base = layer(z, m)
    z2 = {k: v.clone() for k, v in z.items()}
    z2["0e"][1, 3] += 1.0
    z2["0o"][1, 3] += 1.0
    out = layer(z2, m)
    changed = (out["0e"] - base["0e"]).abs().amax(-1) > 1e-12
    expected = torch.zeros(N, N, dtype=torch.bool)
    expected[1, :], expected[:, 1] = True, True
    assert torch.equal(changed, expected)
