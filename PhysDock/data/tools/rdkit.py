import os
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem.rdchem import ChiralType, BondType
import numpy as np
import copy

from PhysDock.data.constants.periodic_table import PeriodicTable
from PhysDock.utils.io_utils import load_txt


def get_ref_mol(string):
    if Chem.MolFromSmiles(string) is not None:
        mol = Chem.MolFromSmiles(string)
    elif os.path.isfile(string) and string.split(".")[-1] == "smi":
        mol = Chem.MolFromSmiles(load_txt(string).strip())
    else:
        mol = None
    if mol is not None:
        AllChem.EmbedMolecule(mol,maxAttempts=100000)
        # mol2 = Chem.MolFromPDBBlock(Chem.MolToPDBBlock(mol))
        # for atom in mol2.GetAtoms():
        #     # if atom.GetChiralTag() != ChiralType.CHI_UNSPECIFIED:
        #     print(f"Atom {atom.GetIdx()} has chiral tag: {atom.GetChiralTag()}")
        # mol = Chem.RemoveAllHs(mol2)
        mol = Chem.RemoveAllHs(mol)
    return mol


Hybridization = {
    Chem.rdchem.HybridizationType.S: 0,
    Chem.rdchem.HybridizationType.SP: 1,
    Chem.rdchem.HybridizationType.SP2: 2,
    Chem.rdchem.HybridizationType.SP3: 3,
    Chem.rdchem.HybridizationType.SP3D: 4,
    Chem.rdchem.HybridizationType.SP3D2: 5,
}

Chirality = {ChiralType.CHI_TETRAHEDRAL_CW: 0,
             ChiralType.CHI_TETRAHEDRAL_CCW: 1,
             ChiralType.CHI_UNSPECIFIED: 2,
             ChiralType.CHI_OTHER: 2}
# Add None
Bonds = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2, BondType.AROMATIC: 3}

dihedral_pattern = Chem.MolFromSmarts('[*]~[*]~[*]~[*]')


# Feats From SMI
# Feats From MOL
# Feats From SDF


def get_features_from_ref_mol(
        ref_mol,
        remove_hs=True
):
    if remove_hs:
        ref_mol = Chem.RemoveAllHs(ref_mol)
    # print(ref_mol)
    # if ref_mol.GetNumConformers()==0:
    #     AllChem.EmbedMolecule(ref_mol,useExpTorsionAnglePrefs=True, useBasicKnowledge=True,maxAttempts=100000)
    ref_conf = ref_mol.GetConformer()
    x_gt = []
    for atom_id, atom in enumerate(ref_mol.GetAtoms()):
        atom_pos = ref_conf.GetAtomPosition(atom_id)
        x_gt.append(np.array([atom_pos.x, atom_pos.y, atom_pos.z]))
    x_gt = np.stack(x_gt, axis=0).astype(np.float32)
    x_exists = np.ones_like(x_gt[:, 0]).astype(np.int64)
    a_mask = np.ones_like(x_gt[:, 0]).astype(np.int64)

    # Ref Mol
    AllChem.EmbedMolecule(ref_mol,maxAttempts=100000)
    AllChem.MMFFOptimizeMolecule(ref_mol)
    num_atoms = ref_mol.GetNumAtoms()
    conf = ref_mol.GetConformer()
    ring = ref_mol.GetRingInfo()

    # Filtering Conditions
    # if ref_mol.GetNumAtoms() < 4:
    #     return None
    # if ref_mol.GetNumBonds() < 4:
    #     return None
    #
    # k = 0
    # for conf in [conf]:
    #     # skip mols with atoms with more than 4 neighbors for now
    #     n_neighbors = [len(a.GetNeighbors()) for a in ref_mol.GetAtoms()]
    #     if np.max(n_neighbors) > 4:
    #         continue
    #     try:
    #         conf_canonical_smi = Chem.MolToSmiles(Chem.RemoveHs(ref_mol))
    #     except Exception as e:
    #         continue
    #     k += 1
    # if k == 0:
    #     return None

    ref_pos = []
    ref_charge = []
    ref_element = []
    ref_is_aromatic = []
    ref_degree = []
    ref_hybridization = []
    ref_implicit_valence = []
    ref_chirality = []
    ref_in_ring_of_3 = []
    ref_in_ring_of_4 = []
    ref_in_ring_of_5 = []
    ref_in_ring_of_6 = []
    ref_in_ring_of_7 = []
    ref_in_ring_of_8 = []
    for atom_id, atom in enumerate(ref_mol.GetAtoms()):
        atom_pos = conf.GetAtomPosition(atom_id)
        ref_pos.append(np.array([atom_pos.x, atom_pos.y, atom_pos.z]))
        ref_charge.append(atom.GetFormalCharge())
        ref_element.append(atom.GetAtomicNum() - 1)
        ref_is_aromatic.append(int(atom.GetIsAromatic()))
        ref_degree.append(min(atom.GetDegree(), 8))
        ref_hybridization.append(Hybridization.get(atom.GetHybridization(), 6))
        ref_implicit_valence.append(min(atom.GetImplicitValence(), 8))
        ref_chirality.append(Chirality.get(atom.GetChiralTag(), 2))
        ref_in_ring_of_3.append(int(ring.IsAtomInRingOfSize(atom_id, 3)))
        ref_in_ring_of_4.append(int(ring.IsAtomInRingOfSize(atom_id, 4)))
        ref_in_ring_of_5.append(int(ring.IsAtomInRingOfSize(atom_id, 5)))
        ref_in_ring_of_6.append(int(ring.IsAtomInRingOfSize(atom_id, 6)))
        ref_in_ring_of_7.append(int(ring.IsAtomInRingOfSize(atom_id, 7)))
        ref_in_ring_of_8.append(int(ring.IsAtomInRingOfSize(atom_id, 8)))

    ref_pos = np.stack(ref_pos, axis=0).astype(np.float32)
    ref_charge = np.array(ref_charge).astype(np.float32)
    ref_element = np.array(ref_element).astype(np.int8)
    ref_is_aromatic = np.array(ref_is_aromatic).astype(np.int8)
    ref_degree = np.array(ref_degree).astype(np.int8)
    ref_hybridization = np.array(ref_hybridization).astype(np.int8)
    ref_implicit_valence = np.array(ref_implicit_valence).astype(np.int8)
    ref_chirality = np.array(ref_chirality).astype(np.int8)
    ref_in_ring_of_3 = np.array(ref_in_ring_of_3).astype(np.int8)
    ref_in_ring_of_4 = np.array(ref_in_ring_of_4).astype(np.int8)
    ref_in_ring_of_5 = np.array(ref_in_ring_of_5).astype(np.int8)
    ref_in_ring_of_6 = np.array(ref_in_ring_of_6).astype(np.int8)
    ref_in_ring_of_7 = np.array(ref_in_ring_of_7).astype(np.int8)
    ref_in_ring_of_8 = np.array(ref_in_ring_of_8).astype(np.int8)

    d_token = np.zeros([num_atoms, num_atoms], dtype=np.int8)
    token_bonds = np.zeros([num_atoms, num_atoms], dtype=np.int8)
    bond_type = np.zeros([num_atoms, num_atoms], dtype=np.int8)
    bond_as_double = np.zeros([num_atoms, num_atoms], dtype=np.int8)
    bond_in_ring = np.zeros([num_atoms, num_atoms], dtype=np.int8)
    bond_is_aromatic = np.zeros([num_atoms, num_atoms], dtype=np.int8)
    bond_is_conjugated = np.zeros([num_atoms, num_atoms], dtype=np.int8)
    for i in range(num_atoms - 1):
        for j in range(i + 1, num_atoms):
            dist = len(rdmolops.GetShortestPath(ref_mol, i, j)) - 1
            dist = min(30, dist)
            d_token[i, j] = dist
            d_token[j, i] = dist
    for bond_id, bond in enumerate(ref_mol.GetBonds()):
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        token_bonds[i, j] = 1
        token_bonds[j, i] = 1
        bond_type[i, j] = Bonds.get(bond.GetBondType(), 4)
        bond_type[j, i] = Bonds.get(bond.GetBondType(), 4)
        bond_as_double[i, j] = bond.GetBondTypeAsDouble()
        bond_as_double[j, i] = bond.GetBondTypeAsDouble()
        bond_in_ring[i, j] = bond.IsInRing()
        bond_in_ring[j, i] = bond.IsInRing()
        bond_is_conjugated[i, j] = bond.GetIsConjugated()
        bond_is_conjugated[j, i] = bond.GetIsConjugated()
        bond_is_aromatic[i, j] = bond.GetIsAromatic()
        bond_is_aromatic[j, i] = bond.GetIsAromatic()

    ref_atom_name_chars = [PeriodicTable[e] for e in ref_element.tolist()]
    ref_mask_in_polymer = [1] * len(ref_pos)

    # ccds, restype, residue_index, atom_id_to_conformer_atom_id, a_mask, x_gt, x_exists
    num_atoms = len(x_gt)
    label_feature = {
        "x_gt": x_gt,
        "x_exists": x_exists,
        "a_mask": a_mask,
        "restype": np.array([20]).astype(np.int64),
        "residue_index": np.arange(1).astype(np.int64),
        "atom_id_to_conformer_atom_id": np.arange(num_atoms).astype(np.int64),
        "conformer_id_to_chunk_sizes": np.array([num_atoms]).astype(np.int64)
    }
    conf_feature = {
        "ref_pos": ref_pos,
        "ref_charge": ref_charge,
        "ref_element": ref_element,
        "ref_is_aromatic": ref_is_aromatic,
        "ref_degree": ref_degree,
        "ref_hybridization": ref_hybridization,
        "ref_implicit_valence": ref_implicit_valence,
        "ref_chirality": ref_chirality,
        "ref_in_ring_of_3": ref_in_ring_of_3,
        "ref_in_ring_of_4": ref_in_ring_of_4,
        "ref_in_ring_of_5": ref_in_ring_of_5,
        "ref_in_ring_of_6": ref_in_ring_of_6,
        "ref_in_ring_of_7": ref_in_ring_of_7,
        "ref_in_ring_of_8": ref_in_ring_of_8,
        "d_token": d_token,
        "token_bonds": token_bonds,
        "bond_type": bond_type,
        "bond_as_double": bond_as_double,
        "bond_in_ring": bond_in_ring,
        "bond_is_conjugated": bond_is_conjugated,
        "bond_is_aromatic": bond_is_aromatic,
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_mask_in_polymer": ref_mask_in_polymer,
    }
    return label_feature, conf_feature, ref_mol


def get_features_from_smi(smi, remove_hs=True):
    ref_mol = get_ref_mol(smi)
    label_feature, conf_feature, ref_mol = get_features_from_ref_mol(ref_mol, remove_hs=remove_hs)
    return label_feature, conf_feature, ref_mol
