"""Synthetic data for the first tests of the model: small molecules with stereochemistry (RDKit conformers), as PhysDock sees a ligand: every heavy atom is a token, the molecule is
one conformer (one reference frame). A system is a dict of the features of pomelo.data.features (PhysDock's), plus the stereo table of the evaluation:

    stereo_center [Ns], stereo_nbrs [Ns, 4] (the heavy neighbours sorted by atom index, -1 pads a missing fourth), stereo_sign [Ns] (the sign of the signed volume of the ordered
    neighbours in the true structure)

No model reads the stereo table: the hand of a molecule is in its reference conformer.
"""
from typing import Dict, List

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from pomelo.data import features as Fe

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
    """several systems as one: the atoms, tokens, conformers and stereo tables concatenated, the pair features block-diagonal, every system its own chain and entity, the
    structures side by side along x (the structure is one frame), the template made again from the joined structure"""
    out: Dict[str, list] = {}
    off_a = off_t = off_c = off_chain = 0
    for i, m in enumerate(mols):
        na, nt = m["ref_feat"].shape[0], m["restype"].shape[0]
        shifted = {"x_gt": m["x_gt"] + torch.tensor([i * spacing, 0.0, 0.0]), "atom_to_conformer": m["atom_to_conformer"] + off_c, "atom_to_token": m["atom_to_token"] + off_t,
                   "asym_id": m["asym_id"] + off_chain, "entity_id": m["entity_id"] + off_chain, "stereo_center": m["stereo_center"] + off_a,
                   "stereo_nbrs": torch.where(m["stereo_nbrs"] >= 0, m["stereo_nbrs"] + off_a, m["stereo_nbrs"])}
        for k in ("ref_feat", "ref_pos", "stereo_sign") + Fe.TOKEN_KEYS:
            if k != "templ_pos":
                shifted.setdefault(k, m[k])
        for k, v in shifted.items():
            out.setdefault(k, []).append(v)
        for k in ("token_bonds", "ligand_pair_feat"):
            out.setdefault(k, []).append((m[k], off_t, nt))
        off_a, off_t = off_a + na, off_t + nt
        off_c += int(m["atom_to_conformer"].max()) + 1
        off_chain += int(m["asym_id"].max()) + 1
    sysd = {k: torch.cat(v) for k, v in out.items() if k not in ("token_bonds", "ligand_pair_feat")}
    for k in ("token_bonds", "ligand_pair_feat"):
        full = torch.zeros(off_t, off_t, *out[k][0][0].shape[2:])
        for v, o, nt in out[k]:
            full[o:o + nt, o:o + nt] = v
        sysd[k] = full
    return with_template(sysd)


def with_template(m: Dict[str, torch.Tensor], seed: int = 0) -> Dict[str, torch.Tensor]:
    """the self-template from x_gt: the position of the first atom of every token (a ligand atom, or the residue's own choice: pomelo.data.protein passes templ_source)"""
    first = m.get("templ_source")
    if first is None:
        first = torch.zeros(m["restype"].shape[0], dtype=torch.long).scatter_reduce(0, m["atom_to_token"], torch.arange(m["atom_to_token"].shape[0]), "amin", include_self=False)
    pos, feat = Fe.template(m["x_gt"][first], m["is_protein"], torch.Generator().manual_seed(2000 + seed))
    m = {k: v for k, v in m.items() if k != "templ_source"}
    return {**m, "templ_pos": pos, "templ_feat": feat, "templ_mask": torch.tensor(1.0)}


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
    nbrs = np.full((len(stereo), 4), -1, dtype=np.int64)
    for k, c in enumerate(stereo):
        nb = sorted(x.GetIdx() for x in heavy.GetAtomWithIdx(c).GetNeighbors())
        nbrs[k, :len(nb)] = nb
    center = torch.tensor(stereo, dtype=torch.int64)
    nbrs_t = torch.from_numpy(nbrs)
    x = torch.from_numpy(coords)
    sign = torch.sign(signed_volume(x, center, nbrs_t)).to(torch.float32) if stereo else torch.zeros(0)
    conformer = torch.zeros(n, dtype=torch.long)
    entry = Fe.entry_from_mol(heavy, reference_conformer(smiles, seed).numpy())
    ref_pos = random_frames(torch.from_numpy(entry["ref_pos"]), conformer, seed)
    feat = Fe.ref_feat(entry, np.arange(n))
    feat[:, Fe.POS] = ref_pos
    zeros = torch.zeros(n, dtype=torch.long)
    return with_template({
        "ref_feat": feat, "ref_pos": ref_pos, "atom_to_conformer": conformer, "atom_to_token": torch.arange(n), "x_gt": x,
        **Fe.token_features(torch.full((n,), Fe.LIGAND), zeros, zeros, zeros),
        "token_bonds": torch.from_numpy(entry["token_bonds"]).float(), "ligand_pair_feat": Fe.ligand_pair_feat(entry, np.arange(n)),
        "stereo_center": center, "stereo_nbrs": nbrs_t, "stereo_sign": sign,
    }, seed)


def make_dataset(names=None, seed: int = 0) -> Dict[str, Dict[str, torch.Tensor]]:
    return {name: build(SMILES[name], seed) for name in (names or SMILES)}


def mirror(mol: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """the mirror image: the structure, the reference conformers and the template reflected (the reference of the enantiomer is the mirror image of it), the stereo labels of the
    evaluation of the opposite sign (the graph is the same)"""
    out = dict(mol)
    m = torch.tensor([-1.0, 1.0, 1.0])
    for k in ("x_gt", "ref_pos", "templ_pos"):
        out[k] = mol[k] * m.to(mol[k])
    out["ref_feat"] = mol["ref_feat"].clone()
    out["ref_feat"][:, Fe.POS] = out["ref_pos"]
    out["stereo_sign"] = -mol["stereo_sign"]
    return out
