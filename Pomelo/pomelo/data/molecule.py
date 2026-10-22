"""Synthetic data for the first tests of the model: small molecules with stereochemistry (RDKit conformers), as PhysDock sees a ligand: every heavy atom is a token, the molecule is
one conformer (one reference frame).

A system is a dict of tensors, with the names of PhysDock's features (pomelo.data.protein builds the same dict for proteins):

  atoms [Na]   ref_element, ref_charge, ref_is_aromatic, ref_hybridization, ref_n_h, ref_in_ring   the features of the atom in its reference conformer (the conformer features)
               ref_pos [Na, 3]       the reference conformer (Angstrom; every conformer in a random frame of its own: nothing relates the frames of two conformers)
               atom_to_conformer     the conformer of every atom (AF3's ref_space_uid: a residue, a ligand)
               atom_to_token         the token of every atom (a standard residue is one token, every atom of a ligand is one token)
               x_gt [Na, 3]          the structure (Angstrom, centred): the target, never an input
  tokens [Nt]  restype (0 .. 19 the standard amino acids, 20 a ligand), residue_index, asym_id (the chain / molecule)
  bonds [Nb, 3]  (i, j, type: 1 single, 2 double, 3 triple, 4 aromatic), all covalent bonds of the system

and, computed once by add_features: ref_degree [Na], ref_spd / ref_bond [Na, Na] (graph distance and bond type inside a conformer, FAR / 0 between conformers, as PhysDock's bond
features exist only inside a conformer), token_spd [Nt, Nt] (the smallest ref_spd over the atoms of two tokens), token_to_conformer [Nt].

The stereo table (stereo_center [Ns], stereo_nbrs [Ns, 4]: the heavy neighbours sorted by atom index, -1 pads a missing fourth, stereo_sign [Ns]: the sign of the signed volume of the
ordered neighbours in the true structure) is for the evaluation only (does a sampled structure have the hand of the truth?): no model reads it. The hand of a molecule is in its reference
conformer.
"""
from typing import Dict, List

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

# L- and D-tryptophan have the same graph: only the reference conformer (and the structure) differ
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
LIGAND = 20                  # restype of a ligand token (0 .. 19: the standard amino acids of pomelo.data.protein.STANDARD)
N_RESTYPE = 21
ATOM_KEYS = ("ref_element", "ref_charge", "ref_is_aromatic", "ref_hybridization", "ref_n_h", "ref_in_ring")
TOKEN_KEYS = ("restype", "residue_index", "asym_id")


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


FAR = 9                      # graph distances 0 .. 8 and "far" (also: another conformer)


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


def token_distance(spd: torch.Tensor, atom_to_token: torch.Tensor) -> torch.Tensor:
    """[Nt, Nt] the smallest graph distance between an atom of one token and an atom of the other (FAR beyond 8), by two scatter-min reductions (no [Nt, Nt, Na, Na] tensor)"""
    nt = int(atom_to_token.max()) + 1
    rows = torch.full((nt, spd.shape[1]), FAR, dtype=spd.dtype).scatter_reduce(0, atom_to_token[:, None].expand_as(spd), spd, "amin", include_self=True)
    return torch.full((nt, nt), FAR, dtype=spd.dtype).scatter_reduce(1, atom_to_token[None, :].expand(nt, -1), rows, "amin", include_self=True)


def add_features(mol: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """the static features that follow from the conformers, computed once: the bonds inside a conformer (ref_bond, ref_spd, ref_degree; a bond between two conformers, the peptide
    bond, is not a feature, as in PhysDock), token_spd, token_to_conformer"""
    n, conf = mol["ref_element"].shape[0], mol["atom_to_conformer"]
    b = mol["bonds"]
    inner = b[conf[b[:, 0]] == conf[b[:, 1]]] if len(b) else b
    bond = torch.zeros(n, n, dtype=torch.long)
    if len(inner):
        bond[inner[:, 0], inner[:, 1]] = inner[:, 2]
        bond[inner[:, 1], inner[:, 0]] = inner[:, 2]
    spd = graph_distance(n, inner)
    a2t = mol["atom_to_token"]
    nt = int(a2t.max()) + 1
    t2c = torch.zeros(nt, dtype=torch.long).scatter_(0, a2t, conf)
    assert torch.equal(t2c[a2t], conf), "the atoms of a token must be in one conformer"
    mol.update({"ref_bond": bond, "ref_spd": spd, "ref_degree": (bond > 0).sum(1), "token_spd": token_distance(spd, a2t), "token_to_conformer": t2c})
    return mol


def random_frame(generator: torch.Generator, scale: float = 20.0):
    """a random proper rotation [3, 3] and a translation [3]: the frame of one conformer"""
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    q = q * torch.sign(torch.diagonal(r))
    q = q * torch.sign(torch.det(q))
    return q, torch.randn(3, generator=generator) * scale


def random_frames(ref_pos: torch.Tensor, conformer: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """every conformer of ref_pos [Na, 3] centred and put into a frame of its own (a random rotation and translation), because nothing relates the frames of two conformers"""
    g = torch.Generator().manual_seed(1000 + seed)
    ref = ref_pos.clone()
    for c in conformer.unique():
        sel = conformer == c
        q, t = random_frame(g)
        ref[sel] = (ref_pos[sel] - ref_pos[sel].mean(0)) @ q.T + t
    return ref


def reference_conformer(smiles: str, seed: int) -> torch.Tensor:
    """the heavy-atom positions of another RDKit conformer of the molecule (other seed, MMFF), the reference conformer: the stereochemistry is that of the SMILES, the coordinates are not
    those of the structure"""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(mol, randomSeed=seed + 101) != 0:
        raise RuntimeError(f"no reference conformer for {smiles}")
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    return torch.from_numpy(Chem.RemoveHs(mol).GetConformer().GetPositions().astype(np.float32))


def combine(mols: List[Dict[str, torch.Tensor]], spacing: float = 12.0) -> Dict[str, torch.Tensor]:
    """several systems as one: the atoms, tokens, bonds, conformers and stereo tables concatenated, every system its own chain (asym_id), the structures side by side along x (the
    structure is one frame: the coordinates of the denoiser)"""
    keys = ATOM_KEYS + TOKEN_KEYS + ("ref_pos", "x_gt", "atom_to_conformer", "atom_to_token", "stereo_center", "stereo_nbrs", "stereo_sign", "bonds")
    out = {k: [] for k in keys}
    off_a = off_t = off_c = off_chain = 0
    for i, m in enumerate(mols):
        for k in ATOM_KEYS + ("restype", "residue_index", "ref_pos", "stereo_sign"):
            out[k].append(m[k])
        out["asym_id"].append(m["asym_id"] + off_chain)
        out["x_gt"].append(m["x_gt"] + torch.tensor([i * spacing, 0.0, 0.0]))
        out["atom_to_conformer"].append(m["atom_to_conformer"] + off_c)
        out["atom_to_token"].append(m["atom_to_token"] + off_t)
        out["stereo_center"].append(m["stereo_center"] + off_a)
        out["stereo_nbrs"].append(torch.where(m["stereo_nbrs"] >= 0, m["stereo_nbrs"] + off_a, m["stereo_nbrs"]))
        out["bonds"].append(m["bonds"] + torch.tensor([off_a, off_a, 0]))
        off_a += m["ref_element"].shape[0]
        off_t += int(m["atom_to_token"].max()) + 1
        off_c += int(m["atom_to_conformer"].max()) + 1
        off_chain += int(m["asym_id"].max()) + 1
    return add_features({k: torch.cat(v) for k, v in out.items()})


def build(smiles: str, seed: int = 0) -> Dict[str, torch.Tensor]:
    """a molecule as a ligand: one token per heavy atom, one conformer, the structure from an RDKit conformer (seed), the reference conformer from another one (seed + 101)"""
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
    nbrs = np.full((len(stereo), 4), -1, dtype=np.int64)
    for k, c in enumerate(stereo):
        nb = sorted(x.GetIdx() for x in heavy.GetAtomWithIdx(c).GetNeighbors())
        nbrs[k, :len(nb)] = nb
    center = torch.tensor(stereo, dtype=torch.int64)
    nbrs_t = torch.from_numpy(nbrs)
    x = torch.from_numpy(coords)
    sign = torch.sign(signed_volume(x, center, nbrs_t)).to(torch.float32) if stereo else torch.zeros(0)
    conformer = torch.zeros(n, dtype=torch.long)
    return add_features({
        "ref_element": torch.tensor([ELEMENTS.get(a.GetAtomicNum(), 9) for a in atoms]), "ref_charge": torch.tensor([a.GetFormalCharge() for a in atoms]),
        "ref_is_aromatic": torch.tensor([int(a.GetIsAromatic()) for a in atoms]), "ref_hybridization": torch.tensor([HYBRIDISATION.get(a.GetHybridization(), 3) for a in atoms]),
        "ref_n_h": torch.tensor([a.GetTotalNumHs() for a in atoms]), "ref_in_ring": torch.tensor([int(a.IsInRing()) for a in atoms]),
        "ref_pos": random_frames(reference_conformer(smiles, seed), conformer, seed), "atom_to_conformer": conformer, "atom_to_token": torch.arange(n),
        "restype": torch.full((n,), LIGAND), "residue_index": torch.zeros(n, dtype=torch.long), "asym_id": torch.zeros(n, dtype=torch.long),
        "bonds": torch.tensor([[b.GetBeginAtomIdx(), b.GetEndAtomIdx(), BOND_TYPES.get(b.GetBondType(), 1)] for b in heavy.GetBonds()]),
        "x_gt": x, "stereo_center": center, "stereo_nbrs": nbrs_t, "stereo_sign": sign,
    })


def make_dataset(names=None, seed: int = 0) -> Dict[str, Dict[str, torch.Tensor]]:
    return {name: build(SMILES[name], seed) for name in (names or SMILES)}


def mirror(mol: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """the mirror image: the structure and the reference conformers reflected (the reference of the enantiomer is the mirror image of it), the stereo labels of the evaluation of the
    opposite sign (the graph is the same)"""
    out = dict(mol)
    m = torch.tensor([-1.0, 1.0, 1.0])
    out["x_gt"], out["ref_pos"] = mol["x_gt"] * m.to(mol["x_gt"]), mol["ref_pos"] * m.to(mol["ref_pos"])
    out["stereo_sign"] = -mol["stereo_sign"]
    return out
