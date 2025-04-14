import os.path
import numpy as np
from typing import Optional, Sequence, Dict, OrderedDict, Any, Union
from scipy.sparse import coo_matrix

from .parsers import parse_fasta, parse_hhr, parse_stockholm, parse_a3m, parse_hmmsearch_a3m, \
    parse_hmmsearch_sto, Msa, parse_stockholm_file, Msa
from . import msa_identifiers

FeatureDict = Dict[str, Union[np.ndarray, coo_matrix, None, Any]]


def load_txt(fname):
    with open(fname, "r") as f:
        data = f.read()
    return data


amino_acids = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
               "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "UNK", ]

HHBLITS_AA_TO_AA = {
    "A": "A",
    "B": "D",
    "C": "C",
    "D": "D",
    "E": "E",
    "F": "F",
    "G": "G",
    "H": "H",
    "I": "I",
    "J": "X",
    "K": "K",
    "L": "L",
    "M": "M",
    "N": "N",
    "O": "X",
    "P": "P",
    "Q": "Q",
    "R": "R",
    "S": "S",
    "T": "T",
    "U": "C",
    "V": "V",
    "W": "W",
    "X": "X",
    "Y": "Y",
    "Z": "E",
    "-": "-",
}
standard_protein = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
                    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "UNK", ]
amino_acid_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "UNK",
}

amino_acid_3to1 = {v: k for k, v in amino_acid_1to3.items()}

AA_TO_ID = {
    amino_acid_3to1[ccd]: amino_acids.index(ccd) for ccd in standard_protein
}
AA_TO_ID["-"] = 31

robon_nucleic_acids = ["A", "G", "C", "U", "N", ]

RNA_TO_ID = {ch: robon_nucleic_acids.index(ch) + 21 for ch in robon_nucleic_acids}
RNA_TO_ID["-"] = 31


# DEBUG
# RNA_TO_ID["."] = 31


def make_msa_features(msas: Sequence[Msa], is_rna=False) -> FeatureDict:
    """Constructs a feature dict of MSA features."""
    if not msas:
        raise ValueError("At least one MSA must be provided.")

    int_msa = []
    deletion_matrix = []
    species_ids = []
    seen_sequences = set()

    for msa_index, msa in enumerate(msas):
        if not msa:
            raise ValueError(f"MSA {msa_index} must contain at least one sequence.")
        for sequence_index, (sequence, msa_deletion_matrix) in enumerate(
                zip(msa.sequences, msa.deletion_matrix)):
            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            if is_rna:
                int_msa.append(
                    [RNA_TO_ID.get(res, RNA_TO_ID["N"]) for res in sequence]
                )
                # deletion_matrix.append([
                #     msa_deletion_matrix[id] for id, res in enumerate(sequence)
                # ])
            else:
                int_msa.append(
                    [AA_TO_ID[HHBLITS_AA_TO_AA[res]] for res in sequence]
                )
            deletion_matrix.append(msa_deletion_matrix)
            identifiers = msa_identifiers.get_identifiers(
                msa.descriptions[sequence_index]
            )
            species_ids.append(identifiers.species_id.encode("utf-8"))
    features = {}
    features["deletion_matrix"] = np.array(deletion_matrix, dtype=np.int8)

    features["msa"] = np.array(int_msa, dtype=np.int8)
    features["msa_species_identifiers"] = np.array(species_ids, dtype=np.object_)
    return features


def parse_alignment_dir(
        alignment_dir,
):
    # MSA Order: uniref90 bfd_uniclust30/bfd_uniref30 mgnify
    uniref90_out_path = os.path.join(alignment_dir, "uniref90_hits.sto")
    uniprot_out_path = os.path.join(alignment_dir, "uniprot_hits.sto")
    reduced_bfd_out_path = os.path.join(alignment_dir, "reduced_bfd_hits.sto")
    mgnify_out_path = os.path.join(alignment_dir, "mgnify_hits.sto")
    bfd_uniref30_out_path = os.path.join(alignment_dir, f"bfd_uniref30_hits.a3m")
    bfd_uniclust30_out_path = os.path.join(alignment_dir, f"bfd_uniclust30_hits.a3m")
    rfam_out_path = os.path.join(alignment_dir, f"rfam_hits2.sto")
    rnacentral_out_path = os.path.join(alignment_dir, f"rnacentral_hits.sto")
    nt_out_path = os.path.join(alignment_dir, f"nt_hits.sto")

    uniref90_msa = None
    bfd_uniclust30_msa = None
    bfd_uniref30_msa = None
    reduced_bfd_msa = None
    mgnify_msa = None
    uniprot_msa = None
    rfam_msa = None
    rnacentral_msa = None
    nt_msa = None

    if os.path.exists(uniref90_out_path):
        uniref90_msa = parse_stockholm(load_txt(uniref90_out_path))

    if os.path.exists(bfd_uniclust30_out_path):
        bfd_uniclust30_msa = parse_a3m(load_txt(bfd_uniclust30_out_path))
    if os.path.exists(bfd_uniref30_out_path):
        bfd_uniref30_msa = parse_a3m(load_txt(bfd_uniref30_out_path))
    if os.path.exists(reduced_bfd_out_path):
        reduced_bfd_msa = parse_stockholm(load_txt(reduced_bfd_out_path))
    if os.path.exists(mgnify_out_path):
        mgnify_msa = parse_stockholm(load_txt(mgnify_out_path))

    if os.path.exists(uniprot_out_path):
        uniprot_msa = parse_stockholm(load_txt(uniprot_out_path))

    if os.path.exists(rfam_out_path):
        # rfam_msa = parse_stockholm(load_txt(rfam_out_path))
        rfam_msa = parse_stockholm_file(rfam_out_path)

    if os.path.exists(rnacentral_out_path):
        # rnacentral_msa = parse_stockholm(load_txt(rnacentral_out_path))
        rnacentral_msa = parse_stockholm_file(rnacentral_out_path)

    if os.path.exists(nt_out_path):
        # nt_msa = parse_stockholm(load_txt(nt_out_path))
        nt_msa = parse_stockholm_file(nt_out_path)

    protein_msas = [uniref90_msa, bfd_uniclust30_msa, bfd_uniref30_msa, reduced_bfd_msa, mgnify_msa]
    uniprot_msas = [uniprot_msa]
    rna_msas = [rfam_msa, rnacentral_msa, nt_msa]
    protein_msas = [i for i in protein_msas if i is not None]
    uniprot_msas = [i for i in uniprot_msas if i is not None]
    rna_msas = [i for i in rna_msas if i is not None]
    output = dict()
    if len(uniprot_msas) > 0:
        uniprot_msa_features = make_msa_features(uniprot_msas)
        output["msa_all_seq"] = uniprot_msa_features.pop("msa")
        output["deletion_matrix_all_seq"] = uniprot_msa_features.pop("deletion_matrix")
        output["msa_species_identifiers_all_seq"] = uniprot_msa_features.pop("msa_species_identifiers")
    if len(protein_msas) > 0:
        msa_features = make_msa_features(protein_msas)
        output["msa"] = msa_features.pop("msa")
        output["deletion_matrix"] = msa_features.pop("deletion_matrix")
        output["msa_species_identifiers"] = msa_features.pop("msa_species_identifiers")

    # TODO: DEBUG parse rna sto and
    if len(rna_msas) > 0:
        assert len(protein_msas) == 0
        msa_features = make_msa_features(rna_msas, is_rna=True)
        output["msa"] = msa_features.pop("msa")
        output["deletion_matrix"] = msa_features.pop("deletion_matrix")
        output["msa_species_identifiers"] = msa_features.pop("msa_species_identifiers")

    return output


def parse_protein_alignment_dir(alignment_dir):
    # MSA Order: uniref90 bfd_uniclust30/bfd_uniref30 mgnify
    uniref90_out_path = os.path.join(alignment_dir, "uniref90_hits.sto")
    reduced_bfd_out_path = os.path.join(alignment_dir, "reduced_bfd_hits.sto")
    mgnify_out_path = os.path.join(alignment_dir, "mgnify_hits.sto")
    bfd_uniref30_out_path = os.path.join(alignment_dir, f"bfd_uniref_hits.a3m")
    bfd_uniclust30_out_path = os.path.join(alignment_dir, f"bfd_uniclust30_hits.a3m")

    uniref90_msa = None
    bfd_uniclust30_msa = None
    bfd_uniref30_msa = None
    reduced_bfd_msa = None
    mgnify_msa = None

    if os.path.exists(uniref90_out_path):
        uniref90_msa = parse_stockholm(load_txt(uniref90_out_path))

    if os.path.exists(bfd_uniclust30_out_path):
        bfd_uniclust30_msa = parse_a3m(load_txt(bfd_uniclust30_out_path))
    if os.path.exists(bfd_uniref30_out_path):
        bfd_uniref30_msa = parse_a3m(load_txt(bfd_uniref30_out_path))
    if os.path.exists(reduced_bfd_out_path):
        reduced_bfd_msa = parse_stockholm(load_txt(reduced_bfd_out_path))
    if os.path.exists(mgnify_out_path):
        mgnify_msa = parse_stockholm(load_txt(mgnify_out_path))

    protein_msas = [uniref90_msa, bfd_uniclust30_msa, bfd_uniref30_msa, reduced_bfd_msa, mgnify_msa]
    protein_msas = [i for i in protein_msas if i is not None]

    output = dict()
    if len(protein_msas) > 0:
        msa_features = make_msa_features(protein_msas)
        output["msa"] = msa_features.pop("msa")
        output["deletion_matrix"] = msa_features.pop("deletion_matrix")
        output["msa_species_identifiers"] = msa_features.pop("msa_species_identifiers")

    return output


def parse_uniprot_alignment_dir(
        alignment_dir,
):
    uniprot_out_path = os.path.join(alignment_dir, "uniprot_hits.sto")
    uniprot_msa = None
    if os.path.exists(uniprot_out_path):
        uniprot_msa = parse_stockholm(load_txt(uniprot_out_path))
    uniprot_msas = [uniprot_msa]
    uniprot_msas = [i for i in uniprot_msas if i is not None]
    output = dict()
    if len(uniprot_msas) > 0:
        uniprot_msa_features = make_msa_features(uniprot_msas)
        output["msa_all_seq"] = uniprot_msa_features.pop("msa")
        output["deletion_matrix_all_seq"] = uniprot_msa_features.pop("deletion_matrix")
        output["msa_species_identifiers_all_seq"] = uniprot_msa_features.pop("msa_species_identifiers")
    return output


def parse_rna_from_input_fasta_path(input_fasta_path):
    with open(input_fasta_path, "r") as f:
        query_sequence, dec = parse_fasta(f.read())
        deletion_matrix = [[0] * len(query_sequence[0])]

    query_msa = Msa(
        sequences=query_sequence,
        deletion_matrix=deletion_matrix,
        descriptions=dec
    )
    return query_msa


def parse_rna_single_alignment(input_fasta_path):
    query_msa = parse_rna_from_input_fasta_path(input_fasta_path)
    rna_msas = [query_msa]
    msa_features = make_msa_features(rna_msas, is_rna=True)
    output = dict()
    output["msa"] = msa_features.pop("msa")
    output["deletion_matrix"] = msa_features.pop("deletion_matrix")
    return output


def parse_rna_alignment_dir(
        alignment_dir,
        input_fasta_path,
):
    rfam_out_path = os.path.join(alignment_dir, f"rfam_hits_realigned.sto")
    rnacentral_out_path = os.path.join(alignment_dir, f"rnacentral_hits_realigned.sto")
    nt_out_path = os.path.join(alignment_dir, f"nt_hits_realigned.sto")
    rfam_msa = None
    rnacentral_msa = None
    nt_msa = None

    if os.path.exists(rfam_out_path):
        # rfam_msa = parse_stockholm(load_txt(rfam_out_path))
        rfam_msa = parse_stockholm_file(rfam_out_path)

    if os.path.exists(rnacentral_out_path):
        # rnacentral_msa = parse_stockholm(load_txt(rnacentral_out_path))
        rnacentral_msa = parse_stockholm_file(rnacentral_out_path)

    if os.path.exists(nt_out_path):
        # nt_msa = parse_stockholm(load_txt(nt_out_path))
        nt_msa = parse_stockholm_file(nt_out_path)

    query_msa = parse_rna_from_input_fasta_path(input_fasta_path)
    rna_msas = [query_msa, rfam_msa, rnacentral_msa, nt_msa]

    rna_msas = [i for i in rna_msas if i is not None and len(i) > 0]
    # rna_msas_gt0 = [i for i in rna_msas if len(i) > 0]
    output = dict()
    msa_features = make_msa_features(rna_msas, is_rna=True)
    output["msa"] = msa_features.pop("msa")
    output["deletion_matrix"] = msa_features.pop("deletion_matrix")
    return output