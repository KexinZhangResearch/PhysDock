"""Synthetic data for the first tests of the model: small molecules with stereochemistry (RDKit conformers), cut into units (fragments), with the stereo table.

A molecule is a dict of tensors (atoms are the heavy atoms):
    z, charge, aromatic, hyb, degree, n_h, in_ring   [Na]   element / charge / aromatic / hybridisation / heavy degree / hydrogens / ring flags
    bonds [Nb, 3] (i, j, type: 1 single, 2 double, 3 triple, 4 aromatic), unit [Na] (the unit of every atom)
    coords [Na, 3] (Angstrom, centred), unit_of_atom is `unit`
    stereo_center [Ns], stereo_nbrs [Ns, 4] (the heavy neighbours sorted by atom index, -1 pads a missing fourth: the implicit hydrogen), stereo_sign [Ns] (+1 / -1)

The stereo label is the sign of the signed volume of the ordered neighbours (signed_volume below), measured on the true structure: a pseudoscalar, so the label of the mirror image
is the opposite one. Nothing refers to R / S or to a canonical numbering: the label belongs to the tuple, in the order of the atom numbering of this molecule.

The units follow the rules of the design: rings are never cut, no bond of a stereocentre is cut (the centre and all its neighbours are in one unit), only single bonds between two atoms of
degree > 1 outside rings are cut, pieces of one or two atoms are merged into their largest neighbour.
"""
from typing import Dict, List

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

# L- and D-tryptophan have the same graph: only the stereo labels (and the structure) differ
SMILES = {
    "Ala-Phe": "C[C@H](N)C(=O)N[C@@H](Cc1ccccc1)C(=O)O",
    "Leu-Ala": "CC(C)C[C@H](NC(=O)[C@@H](N)C)C(=O)O",
    "Lys-Pro": "O=C(O)[C@@H]1CCCN1C(=O)[C@H](N)CCCCN",
    "Tyr-Gly-Ile": "CC[C@H](C)[C@H](NC(=O)CNC(=O)[C@@H](N)Cc1ccc(O)cc1)C(=O)O",
    "crotonamide": "C/C=C/C(=O)N[C@H](C)c1ccccc1",
    "estradiol": "C[C@]12CC[C@H]3[C@@H](CCc4cc(O)ccc43)[C@@H]1CC[C@@H]2O",
    "NAC": "CC(=O)N[C@@H](CS)C(=O)O",
    "L-Trp": "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O",
    "D-Trp": "N[C@H](Cc1c[nH]c2ccccc12)C(=O)O",
    "Gly-Phe-Val": "CC(C)[C@H](NC(=O)[C@@H](NC(=O)CN)Cc1ccccc1)C(=O)O",
    "cyclo-Ala-Phe": "O=C1N[C@@H](C)C(=O)N[C@@H]1Cc1ccccc1",
}
ELEMENTS = {6: 0, 7: 1, 8: 2, 16: 3, 9: 4, 17: 5, 35: 6, 15: 7, 53: 8}            # others: 9
N_ELEMENTS = 10
HYBRIDISATION = {Chem.HybridizationType.SP: 0, Chem.HybridizationType.SP2: 1, Chem.HybridizationType.SP3: 2}
BOND_TYPES = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 4}


def signed_volume(x: torch.Tensor, center: torch.Tensor, nbrs: torch.Tensor) -> torch.Tensor:
    """signed volume of the ordered neighbours of the stereocentres, of the three unit vectors (so in [-1, 1], the size of the molecule does not matter):
    four neighbours: det[x_2 - x_1, x_3 - x_1, x_4 - x_1]; three neighbours and an implicit hydrogen: det[x_1 - x_c, x_2 - x_c, x_3 - x_c].
    x [..., Na, 3], center [Ns], nbrs [Ns, 4] (-1: missing) -> [..., Ns]. A pseudoscalar: it changes sign under a reflection and not under a rotation or a translation."""
    four = (nbrs[:, 3] >= 0)
    n = nbrs.clamp(min=0)
    p = [x[..., n[:, k], :] for k in range(4)]
    c = x[..., center, :]
    origin = torch.where(four[:, None], p[0], c)
    vecs = [torch.where(four[:, None], p[1], p[0]) - origin, torch.where(four[:, None], p[2], p[1]) - origin, torch.where(four[:, None], p[3], p[2]) - origin]
    vecs = [v / (v.square().sum(-1, keepdim=True) + 1e-12).sqrt() for v in vecs]
    return (vecs[0] * torch.linalg.cross(vecs[1], vecs[2], dim=-1)).sum(-1)


FAR = 9                      # graph distances 0 .. 8 and "far"


def graph_distance(n: int, bonds: torch.Tensor) -> torch.Tensor:
    """[n, n] shortest path length in bonds, FAR (9) beyond 8 and between disconnected atoms"""
    adj = torch.zeros(n, n)
    if len(bonds):
        adj[bonds[:, 0], bonds[:, 1]] = 1
        adj[bonds[:, 1], bonds[:, 0]] = 1
    spd = torch.full((n, n), FAR, dtype=torch.long)
    reach = torch.eye(n)
    spd[reach > 0] = 0
    for k in range(1, 9):
        reach = (reach @ adj).clamp(max=1)
        spd[(reach > 0) & (spd == FAR)] = k
    return spd


def unit_distance(spd: torch.Tensor, unit: torch.Tensor) -> torch.Tensor:
    """[Nu, Nu] the smallest graph distance between an atom of one unit and an atom of the other (FAR beyond 8), by two scatter-min reductions (no [Nu, Nu, Na, Na] tensor)"""
    nu = int(unit.max()) + 1
    rows = torch.full((nu, spd.shape[1]), FAR, dtype=spd.dtype).scatter_reduce(0, unit[:, None].expand_as(spd), spd, "amin", include_self=True)
    return torch.full((nu, nu), FAR, dtype=spd.dtype).scatter_reduce(1, unit[None, :].expand(nu, -1), rows, "amin", include_self=True)


def add_graph_features(mol: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """the static graph features of a molecule, computed once: spd [Na, Na] (graph distance), bond [Na, Na] (bond type, 0 none), unit_spd [Nu, Nu]"""
    n = mol["z"].shape[0]
    spd = graph_distance(n, mol["bonds"])
    bond = torch.zeros(n, n, dtype=torch.long)
    if len(mol["bonds"]):
        bond[mol["bonds"][:, 0], mol["bonds"][:, 1]] = mol["bonds"][:, 2]
        bond[mol["bonds"][:, 1], mol["bonds"][:, 0]] = mol["bonds"][:, 2]
    mol.update({"spd": spd, "bond": bond, "unit_spd": unit_distance(spd, mol["unit"])})
    return mol


def fragment(mol: Chem.Mol, stereo_atoms: set) -> np.ndarray:
    """unit index of every atom: union-find over the bonds that are not cut"""
    n = mol.GetNumAtoms()
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        cut = (b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing() and mol.GetAtomWithIdx(i).GetDegree() > 1 and mol.GetAtomWithIdx(j).GetDegree() > 1
               and i not in stereo_atoms and j not in stereo_atoms)
        if not cut:
            parent[find(i)] = find(j)
    for _ in range(n):                                                                 # merge the pieces of at most two atoms into their largest neighbour
        comp: Dict[int, List[int]] = {}
        for a in range(n):
            comp.setdefault(find(a), []).append(a)
        small = [atoms for atoms in sorted(comp.values(), key=len) if len(atoms) <= 2 and len(comp) > 1]
        if not small:
            break
        atoms = small[0]
        nbr = {find(x.GetIdx()) for a in atoms for x in mol.GetAtomWithIdx(a).GetNeighbors()} - {find(atoms[0])}
        if not nbr:
            break
        parent[find(atoms[0])] = max(nbr, key=lambda r: len(comp[r]))
    roots = sorted({find(a) for a in range(n)}, key=lambda r: min(a for a in range(n) if find(a) == r))
    return np.array([roots.index(find(a)) for a in range(n)], dtype=np.int64)


def random_frame(generator: torch.Generator, scale: float = 20.0):
    """a random proper rotation [3, 3] and a translation [3]: the frame of one conformer"""
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    q = q * torch.sign(torch.diagonal(r))
    q = q * torch.sign(torch.det(q))
    return q, torch.randn(3, generator=generator) * scale


def add_reference(mol: Dict[str, torch.Tensor], ref_pos: torch.Tensor, conformer: torch.Tensor, seed: int = 0) -> Dict[str, torch.Tensor]:
    """the reference conformers of a molecule, as AF3 / PhysDock give them: ref_pos [Na, 3] (Angstrom, in any frame) and conformer [Na] (the conformer of every atom: the atoms of
    a conformer share one frame). Every conformer is put into a frame of its own (a random rotation and translation), because nothing relates the frames of two conformers.
    Adds ref_pos, conformer, and unit_conformer [Nu]."""
    g = torch.Generator().manual_seed(1000 + seed)
    ref = ref_pos.clone()
    for c in conformer.unique():
        sel = conformer == c
        q, t = random_frame(g)
        ref[sel] = (ref_pos[sel] - ref_pos[sel].mean(0)) @ q.T + t
    nu = int(mol["unit"].max()) + 1
    unit_conformer = torch.zeros(nu, dtype=torch.long).scatter_(0, mol["unit"], conformer)
    assert torch.equal(unit_conformer[mol["unit"]], conformer), "the atoms of a unit must be in one conformer"
    mol.update({"ref_pos": ref, "conformer": conformer, "unit_conformer": unit_conformer})
    return mol


def reference_conformer(smiles: str, seed: int) -> torch.Tensor:
    """the heavy-atom positions of another RDKit conformer of the molecule (other seed, MMFF), the reference conformer: the stereochemistry is that of the SMILES, the coordinates are not
    those of the structure"""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(mol, randomSeed=seed + 101) != 0:
        raise RuntimeError(f"no reference conformer for {smiles}")
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    return torch.from_numpy(Chem.RemoveHs(mol).GetConformer().GetPositions().astype(np.float32))


def combine(mols: List[Dict[str, torch.Tensor]], spacing: float = 12.0) -> Dict[str, torch.Tensor]:
    """several molecules as one system: the atoms, bonds, units and stereo tables concatenated, every molecule one conformer (its own reference frame), the structures side by side
    along x (the structure itself is one frame: the coordinates of the denoiser)"""
    off_a = off_u = 0
    out = {k: [] for k in ("z", "charge", "aromatic", "hyb", "degree", "n_h", "in_ring", "unit", "coords", "stereo_center", "stereo_nbrs", "stereo_sign", "ref_pos", "conformer")}
    bonds = []
    for i, m in enumerate(mols):
        for k in ("z", "charge", "aromatic", "hyb", "degree", "n_h", "in_ring", "stereo_sign"):
            out[k].append(m[k])
        out["unit"].append(m["unit"] + off_u)
        out["coords"].append(m["coords"] + torch.tensor([i * spacing, 0.0, 0.0]))
        out["stereo_center"].append(m["stereo_center"] + off_a)
        out["stereo_nbrs"].append(torch.where(m["stereo_nbrs"] >= 0, m["stereo_nbrs"] + off_a, m["stereo_nbrs"]))
        out["ref_pos"].append(m["ref_pos"])
        out["conformer"].append(m["conformer"] + (out["conformer"][-1].max() + 1 if out["conformer"] else 0))
        bonds.append(m["bonds"] + torch.tensor([off_a, off_a, 0]))
        off_a += m["z"].shape[0]
        off_u += int(m["unit"].max()) + 1
    mol = {k: torch.cat(v) for k, v in out.items()}
    mol["bonds"] = torch.cat(bonds)
    nu = int(mol["unit"].max()) + 1
    mol["unit_conformer"] = torch.zeros(nu, dtype=torch.long).scatter_(0, mol["unit"], mol["conformer"])
    return add_graph_features(mol)


def build(smiles: str, seed: int = 0) -> Dict[str, torch.Tensor]:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        raise RuntimeError(f"no conformer for {smiles}")
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    heavy = Chem.RemoveHs(mol)
    coords = heavy.GetConformer().GetPositions().astype(np.float32)
    coords -= coords.mean(0)
    stereo = [i.centeredOn for i in Chem.FindPotentialStereo(heavy) if i.type == Chem.StereoType.Atom_Tetrahedral and i.specified == Chem.StereoSpecified.Specified]
    n = heavy.GetNumAtoms()
    atoms = [heavy.GetAtomWithIdx(i) for i in range(n)]
    ring = np.array([a.IsInRing() for a in atoms])
    nbrs = np.full((len(stereo), 4), -1, dtype=np.int64)
    for k, c in enumerate(stereo):
        nb = sorted(x.GetIdx() for x in heavy.GetAtomWithIdx(c).GetNeighbors())
        nbrs[k, :len(nb)] = nb
    center = torch.tensor(stereo, dtype=torch.int64)
    nbrs_t = torch.from_numpy(nbrs)
    x = torch.from_numpy(coords)
    sign = torch.sign(signed_volume(x, center, nbrs_t)).to(torch.float32) if stereo else torch.zeros(0)
    unit = fragment(heavy, set(stereo))
    return _with_reference(add_graph_features({
        "z": torch.tensor([ELEMENTS.get(a.GetAtomicNum(), 9) for a in atoms]), "charge": torch.tensor([a.GetFormalCharge() for a in atoms]),
        "aromatic": torch.tensor([int(a.GetIsAromatic()) for a in atoms]), "hyb": torch.tensor([HYBRIDISATION.get(a.GetHybridization(), 3) for a in atoms]),
        "degree": torch.tensor([a.GetDegree() for a in atoms]), "n_h": torch.tensor([a.GetTotalNumHs() for a in atoms]), "in_ring": torch.from_numpy(ring).long(),
        "bonds": torch.tensor([[b.GetBeginAtomIdx(), b.GetEndAtomIdx(), BOND_TYPES.get(b.GetBondType(), 1)] for b in heavy.GetBonds()]),
        "unit": torch.from_numpy(unit), "coords": x, "stereo_center": center, "stereo_nbrs": nbrs_t, "stereo_sign": sign,
    }), reference_conformer(smiles, seed))


def _with_reference(mol, ref_pos):
    return add_reference(mol, ref_pos, torch.zeros(mol["z"].shape[0], dtype=torch.long))


def make_dataset(names=None, seed: int = 0) -> Dict[str, Dict[str, torch.Tensor]]:
    return {name: build(SMILES[name], seed) for name in (names or SMILES)}


def mirror(mol: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """the mirror image: the structure reflected, every stereo label of the opposite sign (the graph is the same)"""
    out = dict(mol)
    out["coords"] = mol["coords"] * torch.tensor([-1.0, 1.0, 1.0])
    out["stereo_sign"] = -mol["stereo_sign"]
    if "ref_pos" in mol:                                                                  # the reference conformer of the enantiomer is the mirror image of it
        out["ref_pos"] = mol["ref_pos"] * torch.tensor([-1.0, 1.0, 1.0])
    return out
