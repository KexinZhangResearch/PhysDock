# Copyright 2021 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Simplified for PhysDock.
"""Pairing of UniProt MSA rows across protein chains by species (AlphaFold-Multimer)."""
import numpy as np


def _sort_desc(values: np.ndarray) -> np.ndarray:
    """Indices that sort ``values`` in descending order, exactly as pandas ``sort_values(ascending=False)``."""
    idx = np.arange(len(values))[::-1]
    return idx[values[::-1].argsort(kind="quicksort")][::-1]


def _species_rows(chain):
    """{species: (rows of that species in MSA order, similarity of those rows to the query)} of one chain."""
    msa = chain.msa_all_seq
    similarity = np.sum(msa[0][None] == msa, axis=-1) / float(len(msa[0]))
    groups = {}
    for row, species in enumerate(chain.msa_species_identifiers_all_seq):
        if species is not None:                 # pandas groupby drops missing keys
            groups.setdefault(species, []).append(row)
    return {s: (np.array(rows), similarity[rows]) for s, rows in groups.items()}


def pair_rows(chains) -> np.ndarray:
    """Paired row indices [P, n_chains]; -1 selects the gap row appended to a chain without that species.

    Row 0 pairs the queries. Species found in >= 2 chains (and with <= 600 rows in each) give min(rows) pairs, the
    rows of every chain taken by decreasing similarity to its query. Pairs are ordered by the number of chains that
    take part (more first), then by abs(product of the indices).

    Args:
        chains: protein chains with msa_all_seq [rows, n_res] and msa_species_identifiers_all_seq [rows]
    """
    n = len(chains)
    tables = [_species_rows(c) for c in chains]
    species_all = sorted(set().union(*tables))
    species_all.remove(b"")                       # query species
    by_count = {k: [] for k in range(n)}
    by_count[n] = [np.zeros(n, int)]
    for species in species_all:
        if not species:
            continue
        found = [t.get(species) for t in tables]
        present = [f for f in found if f is not None]
        if len(present) <= 1 or any(len(rows) > 600 for rows, _ in present):
            continue
        take = min(len(rows) for rows, _ in present)
        cols = []
        for f in found:
            if f is None:
                cols.append([-1] * take)            # gap row
            else:
                rows, sim = f
                cols.append(rows[_sort_desc(sim)][:take])
        by_count[len(present)].extend(np.array(cols).transpose())
    ordered = []
    for k in sorted(by_count, reverse=True):
        rows = np.array(by_count[k])
        if len(rows):
            ordered.extend(rows[np.argsort(np.abs(np.prod(rows, axis=1)))])
    return np.array(ordered)
