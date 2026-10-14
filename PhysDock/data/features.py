"""Feature schema: shape, dtype and meaning of every model input.

Shapes: N_token tokens (a standard residue is one token, every atom of any other component is a token), N_atom atoms,
N_msa sampled MSA rows (max_msa_clusters), N_resample = num_msa_samples - 1. Features are unbatched; in training the
token / atom dims are padded to the crop sizes (see token_mask / atom_mask). Pair masks are built in the model.
"""
from typing import NamedTuple, Tuple, Union

import torch


class FeatureSpec(NamedTuple):
    shape: Tuple[Union[str, int], ...]
    dtype: torch.dtype
    description: str


f32, i32, i64 = torch.float32, torch.int32, torch.int64
FEATURES = {
    # atoms
    "atom_mask": FeatureSpec(("N_atom",), f32, "1 for real atoms, 0 for padding"),
    "atom_to_token": FeatureSpec(("N_atom",), i64, "token of every atom (atoms of a token are contiguous)"),
    "atom_to_conformer": FeatureSpec(("N_atom",), i64, "conformer (residue / ligand) of every atom (AF3 ref_space_uid)"),
    "ref_pos": FeatureSpec(("N_atom", 3), f32, "reference conformer, randomly rotated / shifted per conformer"),
    "ref_feat": FeatureSpec(("N_atom", 167), f32, "[ref_pos 3, charge 1, element 128, aromatic 1, degree 9, "
                                                  "hybridisation 7, implicit valence 9, chirality 3, ring size 3..8 6]"),
    "x_gt": FeatureSpec(("N_atom", 3), f32, "ground-truth coordinates (RDKit conformer for a SMILES ligand)"),
    # tokens
    "token_mask": FeatureSpec(("N_token",), f32, "1 for real tokens, 0 for padding"),
    "split_sizes": FeatureSpec(("N_token",), i64, "number of atoms of every token"),
    "token_to_centre_atom": FeatureSpec(("N_token",), i64, "centre atom (CA / C1', the atom itself for ligands)"),
    "token_to_pseudo_beta_atom": FeatureSpec(("N_token",), i64, "pseudo-beta atom (CB, CA for GLY, C4 / C2 for nucleotides)"),
    "restype": FeatureSpec(("N_token",), i64, "index in constants.standard_ccds, 20 for non-standard components"),
    "residue_index": FeatureSpec(("N_token",), i64, "position of the residue in its input chain"),
    "asym_id": FeatureSpec(("N_token",), i32, "chain index"),
    "entity_id": FeatureSpec(("N_token",), i32, "entity index (same component sequence)"),
    "sym_id": FeatureSpec(("N_token",), i32, "index of the chain within its entity"),
    "is_protein": FeatureSpec(("N_token",), f32, "receptor tokens and peptide-like ligand chains"),
    "is_ligand": FeatureSpec(("N_token",), f32, "other ligand tokens"),
    "is_pocket": FeatureSpec(("N_token",), f32, "receptor residues near the reference ligand"),
    "key_res_feat": FeatureSpec(("N_token", 7), f32, "PLIP flags [salt bridge, pi-cation, hydrophobic, pi-stacking, "
                                                     "H-bond, metal, 0], randomly masked"),
    "target_feat": FeatureSpec(("N_token", 65), f32, "[restype one-hot 32, MSA profile 32, deletion mean 1]"),
    "msa_feat": FeatureSpec(("N_msa", "N_token", 34), f32, "MSA rows (row 0 = query): [one-hot 32, has deletion, deletion value]"),
    "msa_feat_resampled": FeatureSpec(("N_resample", "N_msa", "N_token", 34), f32, "MSA samples of redocking rounds 2.. (optional)"),
    # token pairs
    "token_bonds": FeatureSpec(("N_token", "N_token"), f32, "bonds inside non-standard components"),
    "ligand_pair_feat": FeatureSpec(("N_token", "N_token", 42), f32, "inside non-standard components: [topological distance 32, "
                                                                     "bond type 5, bonded, double, ring, conjugated, aromatic]"),
    "templ_feat": FeatureSpec(("N_token", "N_token", 40), f32, "self-template of protein tokens: [distogram 39, mask 1]"),
    "templ_mask": FeatureSpec((), f32, "1 if the template is used"),
}
OPTIONAL_FEATURES = {"msa_feat_resampled"}


def validate_features(features) -> dict:
    """Check keys, dtypes and shapes against FEATURES; returns the sizes of the symbolic dims."""
    missing = set(FEATURES) - OPTIONAL_FEATURES - set(features)
    unknown = set(features) - set(FEATURES)
    if missing or unknown:
        raise ValueError(f"feature keys do not match the schema: missing {sorted(missing)}, unknown {sorted(unknown)}")
    sizes = {}
    for key, value in features.items():
        spec = FEATURES[key]
        if value.dtype != spec.dtype or value.dim() != len(spec.shape):
            raise ValueError(f"{key}: {value.dtype} {tuple(value.shape)}, expected {spec.dtype} {spec.shape}")
        for dim, size in zip(spec.shape, value.shape):
            if (isinstance(dim, int) and size != dim) or (isinstance(dim, str) and sizes.setdefault(dim, size) != size):
                raise ValueError(f"{key}: shape {tuple(value.shape)} does not match {spec.shape} ({sizes})")
    return sizes
