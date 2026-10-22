"""The synthetic data: stereo labels, units, and the mirror image."""
import pytest
import torch

from pomelo import data as D


@pytest.fixture(scope="module")
def dataset():
    return D.make_dataset()


def test_every_molecule_has_stereo_labels_and_a_few_units(dataset):
    for name, m in dataset.items():
        n = m["z"].shape[0]
        assert 10 <= n <= 40 and m["coords"].shape == (n, 3) and abs(float(m["coords"].mean())) < 1e-4
        assert (m["stereo_sign"].abs() == 1).all(), name                      # no label that is zero: the volumes are not flat
        assert 1 <= int(m["unit"].max()) + 1 <= n


def test_a_stereocentre_and_its_neighbours_are_in_one_unit_and_rings_are_not_cut(dataset):
    for name, m in dataset.items():
        for c, nb in zip(m["stereo_center"], m["stereo_nbrs"]):
            atoms = [int(c)] + [int(a) for a in nb if a >= 0]
            assert len({int(m["unit"][a]) for a in atoms}) == 1, (name, atoms)
        # two atoms joined by a bond that is in a ring share a unit
        from rdkit import Chem
        mol = Chem.RemoveHs(Chem.AddHs(Chem.MolFromSmiles(D.SMILES[name])))
        for b in mol.GetBonds():
            if b.IsInRing():
                assert m["unit"][b.GetBeginAtomIdx()] == m["unit"][b.GetEndAtomIdx()], (name, b.GetIdx())


def test_the_label_flips_for_the_mirror_image_and_not_for_a_rotation(dataset):
    m = dataset["estradiol"]
    v = D.signed_volume(m["coords"], m["stereo_center"], m["stereo_nbrs"])
    assert torch.equal(torch.sign(v), m["stereo_sign"])
    mir = D.mirror(m)
    v_mir = D.signed_volume(mir["coords"], mir["stereo_center"], mir["stereo_nbrs"])
    assert torch.allclose(v_mir, -v, atol=1e-6) and torch.equal(torch.sign(v_mir), mir["stereo_sign"])
    q = torch.linalg.qr(torch.randn(3, 3))[0]
    q = q * torch.sign(torch.det(q))
    assert torch.allclose(D.signed_volume(m["coords"] @ q.T + 7.0, m["stereo_center"], m["stereo_nbrs"]), v, atol=1e-5)


def test_the_two_tryptophans_have_the_same_graph_and_opposite_labels(dataset):
    a, b = dataset["L-Trp"], dataset["D-Trp"]
    assert torch.equal(a["z"], b["z"]) and torch.equal(a["bonds"], b["bonds"]) and torch.equal(a["unit"], b["unit"])
    assert torch.equal(a["stereo_sign"], -b["stereo_sign"])
