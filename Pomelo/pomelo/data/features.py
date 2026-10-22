"""The input features of Pomelo: those of PhysDock (PhysDock/data/features.py, the same names, shapes and layouts), plus the coordinates of the template.

  atoms   ref_feat [Na, 167] ([ref_pos 3, charge 1, element 128, aromatic 1, degree 9, hybridisation 7, implicit valence 9, chirality 3, ring size 3..8 6]),
          ref_pos [Na, 3] (every conformer in a random frame of its own), atom_to_token, atom_to_conformer [Na], x_gt [Na, 3] (the target)
  tokens  restype, residue_index, asym_id, entity_id, sym_id [Nt], is_protein, is_ligand, is_pocket [Nt], key_res_feat [Nt, 7], target_feat [Nt, 65]
  pairs   token_bonds [Nt, Nt], ligand_pair_feat [Nt, Nt, 42] (inside non-standard components), templ_feat [Nt, Nt, 40] (self-template: distogram of the template
          positions of the protein tokens, 39 bins, and the mask), templ_mask [] (1: the template is used)
  Pomelo  templ_pos [Nt, 3]: the template positions themselves, in a random frame of their own (an equivariant template needs the coordinates: a distogram has no hand)

Pomelo does not read the position columns of ref_feat (they are not invariant) nor the chirality columns (an RDKit tag, a label of the atom order: the hand of a molecule is the
hand of its reference conformer). The bonds between atoms are not a feature of PhysDock; the layers that need the bonded neighbours (relative vectors) take the pairs of a conformer
closer than BOND_CUTOFF in the reference conformer.

The stereo table (stereo_center, stereo_nbrs, stereo_sign) is for the evaluation only.
"""
from typing import Dict

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType, ChiralType, HybridizationType

REF_DIM = 167
POS, CHIRALITY = slice(0, 3), slice(158, 161)
BOND_CUTOFF = 1.9                        # Angstrom: a covalent bond in a reference conformer
N_RESTYPE = 21
LIGAND = 20
TEMPL_BINS = (3.25, 50.75, 39)          # PhysDock.utils.tensor_utils.dgram_from_positions
HYBRIDISATION = {HybridizationType.S: 0, HybridizationType.SP: 1, HybridizationType.SP2: 2, HybridizationType.SP3: 3, HybridizationType.SP3D: 4, HybridizationType.SP3D2: 5}
CHIRAL_TAG = {ChiralType.CHI_TETRAHEDRAL_CW: 0, ChiralType.CHI_TETRAHEDRAL_CCW: 1}
BOND_TYPE = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2, BondType.AROMATIC: 3}
TOKEN_KEYS = ("restype", "residue_index", "asym_id", "entity_id", "sym_id", "is_protein", "is_ligand", "is_pocket", "key_res_feat", "target_feat", "templ_pos")
TOKEN_PAIR_KEYS = ("token_bonds", "ligand_pair_feat", "templ_feat")


def entry_from_mol(mol: Chem.Mol, ref_pos: np.ndarray) -> Dict[str, np.ndarray]:
    """the conformer features of a molecule (heavy atoms, the RDKit graph) as an entry of PhysDock's CCD library (data/ccd_id_meta_data.pkl.gz; the fields of
    PhysDock.data.tools.rdkit.get_features_from_ref_mol), with ref_pos [n, 3] as the reference conformer"""
    n = mol.GetNumAtoms()
    ring = mol.GetRingInfo()
    atoms = [mol.GetAtomWithIdx(i) for i in range(n)]
    e = {"ref_pos": np.asarray(ref_pos, dtype=np.float32), "ref_charge": np.array([a.GetFormalCharge() for a in atoms], dtype=np.float32),
         "ref_element": np.array([a.GetAtomicNum() - 1 for a in atoms]), "ref_is_aromatic": np.array([int(a.GetIsAromatic()) for a in atoms]),
         "ref_degree": np.array([min(a.GetDegree(), 8) for a in atoms]), "ref_hybridization": np.array([HYBRIDISATION.get(a.GetHybridization(), 6) for a in atoms]),
         "ref_implicit_valence": np.array([min(a.GetImplicitValence(), 8) for a in atoms]), "ref_chirality": np.array([CHIRAL_TAG.get(a.GetChiralTag(), 2) for a in atoms])}
    for r in range(3, 9):
        e[f"ref_in_ring_of_{r}"] = np.array([int(ring.IsAtomInRingOfSize(i, r)) for i in range(n)])
    d = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            p = Chem.rdmolops.GetShortestPath(mol, i, j)
            d[i, j] = d[j, i] = min(30, len(p) - 1) if p else 30
    keys = ("token_bonds", "bond_type", "bond_as_double", "bond_in_ring", "bond_is_conjugated", "bond_is_aromatic")
    pair = {k: np.zeros((n, n), dtype=np.float32 if k == "bond_as_double" else np.int64) for k in keys}
    for b in mol.GetBonds():
        for i, j in ((b.GetBeginAtomIdx(), b.GetEndAtomIdx()), (b.GetEndAtomIdx(), b.GetBeginAtomIdx())):
            pair["token_bonds"][i, j] = 1
            pair["bond_type"][i, j] = BOND_TYPE.get(b.GetBondType(), 4)
            pair["bond_as_double"][i, j] = b.GetBondTypeAsDouble()
            pair["bond_in_ring"][i, j] = int(b.IsInRing())
            pair["bond_is_conjugated"][i, j] = int(b.GetIsConjugated())
            pair["bond_is_aromatic"][i, j] = int(b.GetIsAromatic())
    return {**e, "d_token": d, **pair}


def ref_feat(entry, idx) -> torch.Tensor:
    """[len(idx), 167] the rows of the atoms idx of a library entry, PhysDock's layout (the position columns hold ref_pos, centred)"""
    idx = np.asarray(idx)
    eye = lambda k, v: np.eye(k, dtype=np.float32)[np.asarray(v)[idx]]
    pos = np.asarray(entry["ref_pos"], dtype=np.float32)
    cols = [pos[idx] - pos[idx].mean(0), np.asarray(entry["ref_charge"], dtype=np.float32)[idx, None], eye(128, entry["ref_element"]),
            np.asarray(entry["ref_is_aromatic"], dtype=np.float32)[idx, None], eye(9, entry["ref_degree"]), eye(7, entry["ref_hybridization"]),
            eye(9, entry["ref_implicit_valence"]), eye(3, entry["ref_chirality"])]
    cols += [np.asarray(entry[f"ref_in_ring_of_{r}"], dtype=np.float32)[idx, None] for r in range(3, 9)]
    return torch.from_numpy(np.concatenate(cols, -1).astype(np.float32))


def ligand_pair_feat(entry, idx) -> torch.Tensor:
    """[n, n, 42] the pair features of the atoms idx of a non-standard component (PhysDock's ligand_pair_feat: topological distance 32, bond type 5, bonded, double, ring,
    conjugated, aromatic)"""
    ix = np.ix_(np.asarray(idx), np.asarray(idx))
    eye = lambda k, v: np.eye(k, dtype=np.float32)[np.asarray(v)[ix]]
    f = [eye(32, entry["d_token"]), eye(5, entry["bond_type"])] + [np.asarray(entry[k], dtype=np.float32)[ix][..., None]
                                                                   for k in ("token_bonds", "bond_as_double", "bond_in_ring", "bond_is_conjugated", "bond_is_aromatic")]
    return torch.from_numpy(np.concatenate(f, -1))


def dgram(pos: torch.Tensor) -> torch.Tensor:
    """[N, N, 39] one-hot of the distance bins of PhysDock's self-template"""
    lo, hi, n = TEMPL_BINS
    d2 = (pos[:, None] - pos[None]).square().sum(-1, keepdim=True)
    lower = torch.linspace(lo, hi, n, dtype=pos.dtype, device=pos.device) ** 2
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])])
    return ((d2 > lower) & (d2 < upper)).to(pos.dtype)


def template(pos: torch.Tensor, is_protein: torch.Tensor, generator: torch.Generator = None):
    """the self-template of the protein tokens: pos [Nt, 3] (the template position of every token) -> templ_pos (the protein tokens in a random frame of their own, 0 elsewhere)
    and templ_feat [Nt, Nt, 40] (distogram and mask of the protein pairs)"""
    from pomelo.data.molecule import random_frame
    q, t = random_frame(generator or torch.Generator().manual_seed(0))
    p = is_protein.to(pos.dtype)
    templ_pos = ((pos - pos.mean(0)) @ q.T.to(pos) + t.to(pos)) * p[:, None]
    pp = p[:, None] * p[None]
    return templ_pos, torch.cat([dgram(templ_pos) * pp[..., None], pp[..., None]], -1)


def token_features(restype: torch.Tensor, residue_index: torch.Tensor, asym_id: torch.Tensor, entity_id: torch.Tensor) -> Dict[str, torch.Tensor]:
    """the token features that follow from the sequence (no MSA, no pocket, no key residue: zeros, as PhysDock without them)"""
    nt = restype.shape[0]
    is_protein = (restype < LIGAND).float()
    return {"restype": restype, "residue_index": residue_index, "asym_id": asym_id, "entity_id": entity_id, "sym_id": torch.zeros(nt, dtype=torch.long),
            "is_protein": is_protein, "is_ligand": 1 - is_protein, "is_pocket": torch.zeros(nt), "key_res_feat": torch.zeros(nt, 7),
            "target_feat": torch.nn.functional.pad(torch.nn.functional.one_hot(restype, 32).float(), (0, 33))}


def bonded(ref_pos: torch.Tensor, conformer: torch.Tensor) -> torch.Tensor:
    """[Na, Na] bool: the atom pairs of one conformer closer than BOND_CUTOFF in the reference conformer"""
    d = torch.cdist(ref_pos, ref_pos, compute_mode="donot_use_mm_for_euclid_dist")
    return (d < BOND_CUTOFF) & (conformer[:, None] == conformer[None, :]) & ~torch.eye(ref_pos.shape[0], dtype=torch.bool, device=ref_pos.device)
