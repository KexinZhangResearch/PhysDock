"""Data of Pomelo (the counterpart of PhysDock's data pipeline).

molecule.py   molecules as dicts of tensors: heavy atoms, bonds, units (fragments), the stereo table (ordered neighbours and a label), the reference conformers, graph features; a set of
              synthetic molecules with stereochemistry built with RDKit; combine() joins molecules into a system of several conformers
protein.py    windows of real proteins of the PhysDock training set (one unit and one conformer per residue), PDB output
proteins_10.json   the ten proteins of the overfitting experiments (always the same ones)
"""
