"""Step 6 of Pomelo: the embedder gives the six features, and the multi-scale DiT (atoms -> units -> atoms) is equivariant under rotations, reflections and translations with no
augmentation, the stereo labels reach the output, and the batch of noisy samples is the same as the loop."""
import pytest
import torch

from pomelo import data as D
from pomelo import geometry as G
from pomelo.dit import Config, Denoiser


@pytest.fixture(scope="module")
def mols():
    return D.make_dataset(["Tyr-Gly-Ile", "L-Trp", "D-Trp", "estradiol"])


def model(seed=0):
    torch.manual_seed(seed)
    return Denoiser(Config()).double()


def to_double(mol):
    return {k: (v.double() if v.is_floating_point() else v) for k, v in mol.items()}


def noisy(mol, b=3, seed=0, sigma=(0.2, 0.7, 2.0)):
    g = torch.Generator().manual_seed(seed)
    x0 = to_double(mol)["coords"] / 16
    e = torch.randn(b, *x0.shape, generator=g, dtype=torch.float64)
    return x0[None] + torch.tensor(sigma[:b], dtype=torch.float64)[:, None, None] * (e - e.mean(1, keepdim=True)), torch.tensor(sigma[:b], dtype=torch.float64)


def test_the_embedder_gives_the_six_features(mols):
    den = model()
    st = den.embedder(to_double(mols["Tyr-Gly-Ile"]))
    nu = st["n_units"]
    assert nu == 4 and st["n_atoms"] == 25
    assert set(st["s"]) == {"0e", "0o", "1o", "1e"} and set(st["z"]) == {"0e", "0o"}
    assert st["s"]["0e"].shape == (nu, 128) and st["s"]["0o"].shape == (nu, 32) and st["s"]["1o"].shape == (nu, 32, 3) and st["s"]["1e"].shape == (nu, 16, 3)
    assert st["z"]["0e"].shape == (nu, nu, 32) and st["z"]["0o"].shape == (nu, nu, 8)
    assert float(st["s"]["1o"].abs().max()) == 0 and float(st["s"]["1e"].abs().max()) == 0             # no structure in the static features
    assert float(st["s"]["0o"].abs().max()) > 0 and float(st["z"]["0o"].abs().max()) > 0              # the labels are in the odd features


def test_the_odd_static_features_flip_with_the_labels(mols):
    den = model()
    m = to_double(mols["L-Trp"])
    st, st_m = den.embedder(m), den.embedder(D.mirror(m))
    for k in ("0o",):
        assert torch.allclose(st_m["s"][k], -st["s"][k]) and torch.allclose(st_m["z"][k], -st["z"][k])
    assert torch.allclose(st_m["s"]["0e"], st["s"]["0e"]) and torch.allclose(st_m["z"]["0e"], st["z"]["0e"])           # the even ones do not


@pytest.mark.parametrize("name", ["Tyr-Gly-Ile", "estradiol"])
@pytest.mark.parametrize("det", [1, -1])
def test_denoiser_is_equivariant_with_no_augmentation(mols, name, det):
    """rotate / reflect / translate the noisy structure: the denoised structure is rotated / reflected / translated the same way. A reflection also flips the stereo labels (the mirror
    image is the other enantiomer): the labels are pseudoscalars of the input"""
    den = model()
    mol = to_double(mols[name])
    g = torch.Generator().manual_seed(5)
    q = G.random_orthogonal(det, g)
    t = torch.randn(3, generator=g, dtype=torch.float64) * 0.3
    x, sigma = noisy(mol)
    out = den(mol, x, sigma)
    mol_q = dict(mol)
    if det < 0:
        mol_q["stereo_sign"] = -mol["stereo_sign"]
    out_q = den(mol_q, x @ q.T + t, sigma)
    assert float((out @ q.T + t - out_q).abs().max()) < 1e-9, float((out @ q.T + t - out_q).abs().max())
    assert float(out.abs().max()) > 1e-3


def test_the_stereo_labels_reach_the_output(mols):
    """the same noisy coordinates with the labels of the other enantiomer: a different denoised structure (the network knows which hand is asked for)"""
    den = model()
    a = to_double(mols["L-Trp"])
    b = dict(a)
    b["stereo_sign"] = -a["stereo_sign"]
    x, sigma = noisy(a, sigma=(0.05, 0.1, 0.2))
    assert float((den(a, x, sigma) - den(b, x, sigma)).abs().max()) > 1e-4
    # and mirroring the structure ALONE (labels unchanged) is not the mirror image of the output
    mir = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
    out, out_mir_alone = den(a, x, sigma), den(a, x @ mir.T, sigma)
    assert float((out @ mir.T - out_mir_alone).abs().max()) > 1e-4


def test_the_batch_is_the_loop(mols):
    den = model()
    mol = to_double(mols["Tyr-Gly-Ile"])
    x, sigma = noisy(mol)
    batch = den(mol, x, sigma)
    loop = torch.cat([den(mol, x[i:i + 1], sigma[i:i + 1]) for i in range(3)])
    assert torch.allclose(batch, loop, atol=1e-10)


def test_every_parameter_gets_a_gradient(mols):
    den = model()
    mol = to_double(mols["Tyr-Gly-Ile"])
    x, sigma = noisy(mol)
    den(mol, x, sigma).square().sum().backward()
    dead = [n for n, p in den.named_parameters() if p.grad is None or float(p.grad.abs().max()) == 0]
    assert not dead, dead
