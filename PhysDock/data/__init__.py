"""PhysDock data pipeline: feature schema, dataset, collator and dataloader.

Every model input is listed in :data:`FEATURES` with its shape, dtype and meaning. Shape symbols:

* ``N_token``: number of tokens. A standard residue (amino acid or nucleotide) is one token; every atom of any other
  component (ligand, ion, modified residue) is its own token.
* ``N_atom``: number of atoms (only atoms with a ground-truth position are kept).
* ``N_msa``: number of sampled MSA rows (``max_msa_clusters``, 128 by default).
* ``N_resample``: number of *additional* MSA samples (``num_msa_samples - 1``), used by the multi-round redocking.

All per-token / per-atom features are unbatched (no leading batch dimension); :class:`PhysDockCollator` adds one only
when it is asked to stack several samples. In training the token and atom dimensions are padded to the crop sizes and
``token_mask`` / ``atom_mask`` mark the padding. Pair masks (``token_mask[:, None] * token_mask[None]`` and the atom
equivalent) are not stored; the model builds them where it needs them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple, Union

import numpy as np
import torch

FeatureDict = Dict[str, Union[np.ndarray, Any]]
TensorDict = Dict[str, Union[torch.Tensor, Any]]

PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


@dataclass(frozen=True)
class FeatureSpec:
    """Shape, dtype and meaning of one model input.

    Attributes:
        shape: Dimensions; strings are the symbolic sizes of the module docstring, integers are fixed sizes.
        dtype: Torch dtype of the tensor.
        description: What the feature contains.
    """

    shape: Tuple[Union[str, int], ...]
    dtype: torch.dtype
    description: str


FEATURES: Dict[str, FeatureSpec] = {
    # ------------------------------------------------------------------ atom level
    "atom_mask": FeatureSpec(("N_atom",), torch.float32, "1 for real atoms, 0 for padding (every real atom has a ground-truth position)."),
    "atom_to_token": FeatureSpec(("N_atom",), torch.int64, "Index of the token an atom belongs to; atoms of one token are contiguous."),
    "atom_to_conformer": FeatureSpec(("N_atom",), torch.int64, "Index of the conformer (residue / ligand) an atom belongs to (AF3 ref_space_uid)."),
    "ref_pos": FeatureSpec(("N_atom", 3), torch.float32, "Reference-conformer coordinates, randomly rotated and translated per conformer."),
    "ref_feat": FeatureSpec(("N_atom", 167), torch.float32,
                            "Reference-conformer atom features: [ref_pos (3, == ref_pos), charge (1), element one-hot (128), aromatic (1), "
                            "degree (9), hybridisation (7), implicit valence (9), chirality (3), in ring of size 3..8 (6)]."),
    "x_gt": FeatureSpec(("N_atom", 3), torch.float32, "Ground-truth coordinates (Angstrom). For a SMILES ligand: its RDKit conformer."),
    # ------------------------------------------------------------------ token level
    "token_mask": FeatureSpec(("N_token",), torch.float32, "1 for real tokens, 0 for padding."),
    "split_sizes": FeatureSpec(("N_token",), torch.int64, "Number of atoms of every token (segment sizes for atom -> token pooling)."),
    "token_to_centre_atom": FeatureSpec(("N_token",), torch.int64, "Centre atom of every token (CA / C1' for standard residues, the atom itself otherwise)."),
    "token_to_pseudo_beta_atom": FeatureSpec(("N_token",), torch.int64, "Pseudo-beta atom of every token (CB, CA for GLY, C4/C2 for nucleotides, the atom itself otherwise)."),
    "restype": FeatureSpec(("N_token",), torch.int64, "Residue type index in constants.standard_ccds; 20 for every non-standard component."),
    "residue_index": FeatureSpec(("N_token",), torch.int64, "Position of the token's residue in its input chain."),
    "asym_id": FeatureSpec(("N_token",), torch.int32, "Chain index."),
    "entity_id": FeatureSpec(("N_token",), torch.int32, "Entity index (chains with the same sequence share it)."),
    "sym_id": FeatureSpec(("N_token",), torch.int32, "Index of the chain among the chains of its entity."),
    "is_protein": FeatureSpec(("N_token",), torch.float32, "1 for receptor-chain tokens and for peptide-like ligand chains (>= 2 standard residues)."),
    "is_ligand": FeatureSpec(("N_token",), torch.float32, "1 for ligand-chain tokens except peptide-like ligand chains (1 - is_protein for real tokens)."),
    "is_pocket": FeatureSpec(("N_token",), torch.float32, "1 for receptor residues near the reference ligand (pocket feature; all 0 if disabled)."),
    "key_res_feat": FeatureSpec(("N_token", 7), torch.float32,
                                "PLIP interaction flags [salt bridge, pi-cation, hydrophobic, pi-stacking, H-bond, metal, 0], randomly masked."),
    "target_feat": FeatureSpec(("N_token", 65), torch.float32, "[restype one-hot (32), MSA profile (32), deletion mean (1)]."),
    "msa_feat": FeatureSpec(("N_msa", "N_token", 34), torch.float32, "Sampled MSA rows (row 0 = query): [residue one-hot (32), has deletion, deletion value]."),
    "msa_feat_resampled": FeatureSpec(("N_resample", "N_msa", "N_token", 34), torch.float32,
                                      "Further MSA samples for the redocking rounds 2..num_msa_samples (only present if num_msa_samples > 1)."),
    # ------------------------------------------------------------------ token pairs
    "token_bonds": FeatureSpec(("N_token", "N_token"), torch.float32, "Covalent bonds between atom tokens of the same non-standard component."),
    "ligand_pair_feat": FeatureSpec(("N_token", "N_token", 42), torch.float32,
                                    "Pairs inside a non-standard component: [topological distance one-hot (32), bond type one-hot (5), "
                                    "bonded, bond order, in ring, conjugated, aromatic]; 0 elsewhere."),
    "templ_feat": FeatureSpec(("N_token", "N_token", 40), torch.float32,
                              "Self-template of the protein tokens: [pseudo-beta distogram (39), pair mask (1)]."),
    "templ_mask": FeatureSpec((), torch.float32, "1 if the template is used, 0 if it is switched off (training augmentation)."),
}
"""All model inputs. ``msa_feat_resampled`` is optional (see :data:`OPTIONAL_FEATURES`)."""

OPTIONAL_FEATURES = frozenset({"msa_feat_resampled"})


def validate_features(features: Mapping[str, torch.Tensor]) -> Dict[str, int]:
    """Check keys, dtypes and shapes of an (unbatched) feature dict against :data:`FEATURES`.

    Args:
        features: Output of :class:`PhysDockDataset`.

    Returns:
        The sizes of the symbolic dimensions, e.g. ``{"N_token": 256, "N_atom": 1791, "N_msa": 128}``.

    Raises:
        ValueError: On missing or unexpected keys, wrong dtypes or inconsistent dimensions.
    """
    missing = set(FEATURES) - OPTIONAL_FEATURES - set(features)
    unknown = set(features) - set(FEATURES)
    if missing or unknown:
        raise ValueError(f"feature keys do not match the schema: missing {sorted(missing)}, unknown {sorted(unknown)}")
    sizes: Dict[str, int] = {}
    for key, value in features.items():
        spec = FEATURES[key]
        if value.dtype != spec.dtype:
            raise ValueError(f"{key}: dtype {value.dtype}, expected {spec.dtype}")
        if value.dim() != len(spec.shape):
            raise ValueError(f"{key}: shape {tuple(value.shape)}, expected {spec.shape}")
        for dim, size in zip(spec.shape, value.shape):
            if isinstance(dim, int):
                if size != dim:
                    raise ValueError(f"{key}: shape {tuple(value.shape)}, expected {spec.shape}")
            elif sizes.setdefault(dim, size) != size:
                raise ValueError(f"{key}: dimension {dim} is {size}, but {sizes[dim]} elsewhere")
    return sizes


from PhysDock.data.dataloader import (CCDLibrary, FeatureConfig, PhysDockCollator, PhysDockDataloader,  # noqa: E402
                                      PhysDockDataset, Sample, SystemEntry, SystemFeaturizer, SystemMeta,
                                      default_ccd_path, msa_feat_for_round, seed_worker)

__all__ = [
    "FeatureDict", "TensorDict", "FeatureSpec", "FEATURES", "OPTIONAL_FEATURES", "validate_features", "PDB_CHAIN_IDS",
    "CCDLibrary", "FeatureConfig", "SystemEntry", "SystemMeta", "Sample", "SystemFeaturizer", "PhysDockDataset",
    "PhysDockCollator", "PhysDockDataloader", "default_ccd_path", "msa_feat_for_round", "seed_worker",
]
