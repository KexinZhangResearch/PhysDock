import numpy as np
from .PDBData import protein_letters_3to1_extended, nucleic_letters_3to1_extended

restype_1_to_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    "X": "UNK",
    "0": "A  ", "1": "G  ", "2": "C  ", "3": "U  ", "4": "N  ",
    "5": "DA ", "6": "DG ", "7": "DC ", "8": "DT ", "9": "DN ",
}
na_c_to_type = {
    "A": "A  ", "G": "G  ", "C": "C  ", "U": "U  ", "N": "N  ", "T": "T  ", "X": "N  "
}

restype_3_to_1 = {v: k for k, v in restype_1_to_3.items()}
restype_3_to_1["T  "]="8"


restypes3 = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "UNK",
    "A  ", "G  ", "C  ", "U  ", "N  ",
    "DA ", "DG ", "DC ", "DT ", "DN ",
]
restypes1 = [restype_3_to_1[ccd] for ccd in restypes3]

restype_3_to_1_extended = {}

for c3, c in protein_letters_3to1_extended.items():
    restype_3_to_1_extended[f"{c3:<3}"] = c
# TODO: How to distinguish RNA and DNA
for c3, c in nucleic_letters_3to1_extended.items():
    restype_3_to_1_extended[c3] = restype_3_to_1[na_c_to_type[c]]
restype_3_to_1_extended.update(restype_3_to_1)

############
standard_protein = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
                    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "UNK", ]
standard_rna = ["A  ", "G  ", "C  ", "U  ", "N  ", ]
standard_dna = ["DA ", "DG ", "DC ", "DT ", "DN ", ]
standard_nucleics = standard_rna + standard_dna
standard_ccds_without_gap = standard_protein + standard_nucleics
GAP = ["GAP"]  # used in msa one-hot
standard_ccds = standard_protein + standard_nucleics + GAP

standard_ccd_to_order = {ccd: id for id, ccd in enumerate(standard_ccds)}

standard_purines = ["A  ", "G  ", "DA ", "DG "]
standard_pyrimidines = ["C  ", "U  ", "DC ", "DT "]

is_standard = lambda x: x in standard_ccds
is_unk = lambda x: x in ["UNK", "N  ", "DN ", "GAP", "UNL"]
is_protein = lambda x: x in standard_protein and not is_unk(x)
is_rna = lambda x: x in standard_rna and not is_unk(x)
is_dna = lambda x: x in standard_dna and not is_unk(x)
is_nucleics = lambda x: x in standard_nucleics and not is_unk(x)
is_purines = lambda x: x in standard_purines
is_pyrimidines = lambda x: x in standard_pyrimidines


standard_ccd_to_atoms_num = {s: n for s, n in zip(standard_ccds, [
    5, 11, 8, 8, 6, 9, 9, 4, 10, 8,
    8, 9, 8, 11, 7, 6, 7, 14, 12, 7, None,
    22, 23, 20, 20, None,
    21, 22, 19, 20, None,
    None,
])}

standard_ccd_to_token_centre_atom_name = {
    **{residue: "CA" for residue in standard_protein},
    **{residue: "C1'" for residue in standard_nucleics},
}

standard_ccd_to_frame_atom_name_0 = {
    **{residue: "N" for residue in standard_protein},
    **{residue: "C1'" for residue in standard_nucleics},
}

standard_ccd_to_frame_atom_name_1 = {
    **{residue: "CA" for residue in standard_protein},
    **{residue: "C3'" for residue in standard_nucleics},
}

standard_ccd_to_frame_atom_name_2 = {
    **{residue: "C" for residue in standard_protein},
    **{residue: "C4'" for residue in standard_nucleics},
}

standard_ccd_to_token_pseudo_beta_atom_name = {
    **{residue: "CB" for residue in standard_protein},
    **{residue: "C4" for residue in standard_purines},
    **{residue: "C2" for residue in standard_pyrimidines},
}
standard_ccd_to_token_pseudo_beta_atom_name.update({"GLY": "CA"})


eye_64 = np.eye(64)
eye_128 = np.eye(128)
eye_9 = np.eye(9)
eye_7 = np.eye(7)
eye_3 = np.eye(3)
eye_32 = np.eye(32)
eye_5 = np.eye(5)
eye8 = np.eye(8)
eye5 = np.eye(5)
