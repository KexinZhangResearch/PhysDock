# Copyright 2021 DeepMind Technologies Limited, 2022 AlQuraishi Laboratory
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Simplified for PhysDock.
"""Merge the MSAs of all chains into one MSA of 16384 rows (AlphaFold-Multimer style pairing + padding)."""
import numpy as np

from PhysDock.data.tools.msa_pairing import pair_rows

MSA_ROWS = 16384
MSA_GAP = 31


def _pad_rows(msa: np.ndarray, deletion: np.ndarray, rows: int):
    """Pad to ``rows`` rows by drawing existing rows with replacement (np.random), or truncate."""
    if len(msa) < rows:
        extra = np.random.choice(np.arange(len(msa)), rows - len(msa), replace=True)
        return np.concatenate([msa, msa[extra]], axis=0), np.concatenate([deletion, deletion[extra]], axis=0)
    return msa[:rows], deletion[:rows]


def pair_and_merge(chains):
    """Merged MSA of all chains.

    Every protein chain gets [paired UniProt rows (<= 8192); its own MSA padded to the remaining rows], every ligand
    chain its pseudo-MSA padded to 16384 rows. Chains are ordered proteins first, then grouped by entity_id.

    Args:
        chains: featurizer.Chain objects with asym_id / sym_id / entity_id [n_res], chain_class ("protein" /
            "ligand"), msa / deletion_matrix [rows, n_res]; proteins also msa_all_seq / deletion_matrix_all_seq
            [rows, n_res] and msa_species_identifiers_all_seq [rows]

    Returns:
        dict with msa / deletion_matrix [16384, sum n_res] and asym_id / sym_id / entity_id [sum n_res]
    """
    proteins = [c for c in chains if c.chain_class == "protein"]
    ligands = [c for c in chains if c.chain_class == "ligand"]
    paired = pair_rows(proteins) if len(proteins) >= 2 else None

    merged = []
    for i, c in enumerate(proteins):
        msa_all, del_all = c.msa_all_seq, c.deletion_matrix_all_seq
        if paired is not None:              # append a gap row (index -1) and select the paired rows
            n_res = msa_all.shape[1]
            msa_all = np.concatenate([msa_all, np.full([1, n_res], MSA_GAP, msa_all.dtype)], axis=0)[paired[:, i]]
            del_all = np.concatenate([del_all, np.zeros([1, n_res], del_all.dtype)], axis=0)[paired[:, i]]
        n_all = min(len(msa_all), MSA_ROWS // 2)
        msa, deletion = _pad_rows(c.msa, c.deletion_matrix, MSA_ROWS - n_all)
        merged.append((c, np.concatenate([msa_all[:n_all], msa], axis=0),
                       np.concatenate([del_all[:n_all], deletion], axis=0)))
    for c in ligands:
        merged.append((c, *_pad_rows(c.msa, c.deletion_matrix, MSA_ROWS)))

    merged.sort(key=lambda m: int(m[0].entity_id[0]))      # stable: list order within an entity
    out = {"msa": np.concatenate([m[1] for m in merged], axis=1),
           "deletion_matrix": np.concatenate([m[2] for m in merged], axis=1)}
    for key in ("asym_id", "sym_id", "entity_id"):
        out[key] = np.concatenate([getattr(m[0], key) for m in merged], axis=0)
    return out
