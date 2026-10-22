"""Data of Pomelo (the counterpart of PhysDock's data pipeline).

molecule.py   systems as dicts of tensors with PhysDock's feature names: the conformer features of the heavy atoms (ref_*), atom_to_token / atom_to_conformer, the token features,
              the bonds, the stereo table of the evaluation; a set of synthetic molecules with stereochemistry built with RDKit (ligands: one token per atom); combine() joins systems
protein.py    windows of real proteins of the PhysDock training set (one token and one conformer per residue), PDB output
proteins_10.json   the ten proteins of the overfitting experiments (always the same ones)
"""
