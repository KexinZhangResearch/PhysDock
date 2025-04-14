import os.path
from typing import Optional
from functools import partial
import numpy as np

from PhysDock.utils.io_utils import run_pool_tasks, load_json, load_txt, dump_pkl, \
    convert_md5_string
from PhysDock.data.tools.parsers import parse_fasta
from PhysDock.data.tools.parse_msas import parse_protein_alignment_dir, parse_uniprot_alignment_dir, \
    parse_rna_alignment_dir
from PhysDock.data.alignment_runner import DataProcessor
from PhysDock.data.tools.residue_constants import standard_ccds, amino_acid_3to1
from PhysDock.data.tools.PDBData import protein_letters_3to1_extended, nucleic_letters_3to1_extended


def get_protein_md5(sequence_3):
    ccds = sequence_3.split("-")

    start = 0
    end = 0
    for i, ccd in enumerate(ccds):
        if ccd not in ["UNK", "N  ", "DN ", "GAP"]:
            start = i
            break
    for i, ccd in enumerate(ccds[::-1]):
        if ccd not in ["UNK", "N  ", "DN ", "GAP"]:
            end = i
            break
    # print(start,end)
    ccds_strip_unk = ccds[start:-end] if end > 0 else ccds[start:]
    sequence_0 = "".join(
        [protein_letters_3to1_extended[ccd] if ccd in protein_letters_3to1_extended else "X" for ccd in ccds])
    sequence_1 = "".join([amino_acid_3to1[ccd] if ccd in amino_acid_3to1 else "X" for ccd in ccds])
    sequence_2 = "".join(
        [protein_letters_3to1_extended[ccd] if ccd in protein_letters_3to1_extended else "X" for ccd in ccds_strip_unk])
    sequence_3 = "".join([amino_acid_3to1[ccd] if ccd in amino_acid_3to1 else "X" for ccd in ccds_strip_unk])

    sequences = []
    for sequence in [sequence_0, sequence_1, sequence_2, sequence_3]:
        if sequence not in sequences:
            sequences.append(sequence)
    return sequences, [convert_md5_string(f"protein:{i}") for i in sequences]


def get_rna_md5(sequence_3):
    ccds = sequence_3.split("-")
    chs = [nucleic_letters_3to1_extended[ccd] if ccd not in ["UNK", "GAP", "N  ", "DN "] else "N" for ccd in ccds]
    sequence = "".join(chs)
    md5 = convert_md5_string(f"rna:{sequence}")
    return sequence, md5


class DatasetManager:
    def __init__(
            self,
            dataset_path
    ):
        self.dataset_path = dataset_path
        # Meta Data
        self.chain_id_to_meta_info = load_json(os.path.join(dataset_path, "chain_id_to_meta_info.json"))
        self.pdb_id_to_meta_info = load_json(os.path.join(dataset_path, "pdb_id_to_meta_info.json"))
        self.ccd_id_to_meta_info = load_json(os.path.join(dataset_path, "ccd_id_to_meta_info.json"))

        # filtering chains
        # self.train_polymer_chain_ids = load_json()
        # self.validation_polymer_chain_ids = load_json()

    # def check_protein_msa_features_completeness(input_fasta_path):
    #     pass
    #
    # def check_protein_uniprot_msa_features_completeness(self, chain_ids, num_workers):
    #     def _run(chain_id):
    #         if chain_id not in self.chain_id_to_meta_info:
    #             return {
    #                 "chain_id": f"Not find this chain {chain_id}"
    #             }
    #         sequence_3 = self.chain_id_to_meta_info[chain_id]["sequence_3"]
    #
    #     out = run_pool_tasks(_run, chain_ids, return_dict=True, num_workers=num_workers)
    #     return out

    @staticmethod
    def homo_search(
            input_fasta_path,
            output_dir,
            msas_type,
            convert_md5,
            alphafold3_database_path,
            jackhmmer_binary_path: Optional[str] = None,
            hhblits_binary_path: Optional[str] = None,
            nhmmer_binary_path: Optional[str] = None,
            kalign_binary_path: Optional[str] = None,
            hmmbuild_binary_path: Optional[str] = None,
            hmmalign_binary_path: Optional[str] = None,
            n_cpus: int = 8,
            n_workers: int = 1,
    ):
        data_processor = DataProcessor(
            alphafold3_database_path=alphafold3_database_path,
            jackhmmer_binary_path=jackhmmer_binary_path,
            hhblits_binary_path=hhblits_binary_path,
            nhmmer_binary_path=nhmmer_binary_path,
            kalign_binary_path=kalign_binary_path,
            hmmbuild_binary_path=hmmbuild_binary_path,
            hmmalign_binary_path=hmmalign_binary_path,
            n_cpus=n_cpus,
            n_workers=n_workers,
        )
        data_processor.process(
            input_fasta_path=input_fasta_path,
            output_dir=output_dir,
            msas_type=msas_type,
            # msas_type="bfd_uniclust30", # alphafold3 # rna
            # msas_type="uniprot", # alphafold3 # rna
            convert_md5=convert_md5
        )

    @staticmethod
    def get_unsearched_input_fasta_path(input_fasta_path, output_dir, msas_type, convert_md5, num_workers=128):
        if isinstance(input_fasta_path, list):
            input_fasta_paths = input_fasta_path
        elif os.path.isdir(input_fasta_path):
            input_fasta_paths = [os.path.join(input_fasta_path, i) for i in os.listdir(input_fasta_path)]
        elif os.path.isfile(input_fasta_path):
            input_fasta_paths = [input_fasta_path]
        else:
            input_fasta_paths = []
            Exception("Can't parse input fasta path!")

        prefix = {
            "uniref90": "protein",
            "bfd_uniclust30": "protein",
            "bfd_uniref30": "protein",
            "uniprot": "protein",
            "mgnify": "protein",
            "rfam": "rna",
            "rnacentral": "rna",
            "nt": "rna",
        }[msas_type]
        global _get_unsearched_input_fasta_path

        def _get_unsearched_input_fasta_path(input_fasta_path, convert_md5, prefix, output_dir, msas_type):
            # TODO
            # seqs, decs = parse_fasta(i)
            if convert_md5:
                # dec = convert_md5_string(f"{prefix}:{input_fasta_path}")
                dec = convert_md5_string(f"{prefix}:{seqs[0]}")
            else:
                dec = os.path.split(input_fasta_path)[1].split(".")[0]

            if os.path.exists(os.path.join(output_dir, dec, f"{msas_type}_hits.sto")) or \
                    os.path.exists(os.path.join(output_dir, dec, f"{msas_type}_hits.a3m")):
                return dict()
            else:
                return {input_fasta_path: False}

        out = run_pool_tasks(partial(
            _get_unsearched_input_fasta_path,
            convert_md5=convert_md5,
            prefix=prefix,
            output_dir=output_dir,
            msas_type=msas_type,
        ), input_fasta_paths, num_workers=num_workers, return_dict=True)
        return list(out.keys())

    @staticmethod
    def convert_msas_out_to_msa_features(
            input_fasta_path,
            output_dir,
            msa_feature_dir,
            convert_md5=True,
            num_workers=128
    ):
        if isinstance(input_fasta_path, list):
            input_fasta_paths = input_fasta_path
        elif os.path.isdir(input_fasta_path):
            input_fasta_paths = [os.path.join(input_fasta_path, i) for i in os.listdir(input_fasta_path)]
        elif os.path.isfile(input_fasta_path):
            input_fasta_paths = [input_fasta_path]
        else:
            input_fasta_paths = []
            Exception("Can't parse input fasta path!")

        global _convert_msas_out_to_msa_features

        def _convert_msas_out_to_msa_features(
                input_fasta_path,
                output_dir,
                msa_feature_dir,
                convert_md5=True,
        ):
            prefix = "protein"
            max_seq = 16384
            seqs, decs = parse_fasta(load_txt(input_fasta_path))
            md5 = convert_md5_string(f"{prefix}:{seqs[0]}")
            if convert_md5:
                # TODO: debug
                # dec = convert_md5_string(f"{prefix}:{input_fasta_path}")
                dec = md5
            else:
                dec = os.path.split(input_fasta_path)[1].split(".")[0]

            # DEBUG: whc homo search hits
            if os.path.exists(os.path.join(output_dir, dec, "msas")):
                dec = dec + "/msas"

            pkl_save_path = os.path.join(msa_feature_dir, f"{md5}.pkl.gz")

            if os.path.exists(pkl_save_path):
                return dict()
            if os.path.exists(os.path.join(output_dir, dec, "uniref90_hits.sto")) and \
                    os.path.exists(os.path.join(output_dir, dec, "bfd_uniclust30_hits.a3m")) and \
                    os.path.exists(os.path.join(output_dir, dec, "mgnify_hits.sto")):
                msa_feature = parse_protein_alignment_dir(os.path.join(output_dir, dec))

                sequence = "".join([amino_acid_3to1[standard_ccds[i]] for i in msa_feature["msa"][0]])
                md5_string = convert_md5_string(f"protein:{sequence}")
                if md5 == md5_string:
                    feature = {
                        "msa": msa_feature["msa"][:max_seq].astype(np.int8),
                        "deletion_matrix": msa_feature["deletion_matrix"][:max_seq].astype(np.int8),
                        "msa_species_identifiers": msa_feature["msa_species_identifiers"][:max_seq]
                    }
                    dump_pkl(feature, pkl_save_path)
                    return dict()
                else:
                    return {input_fasta_path: f"seqs not equal, asset [{sequence}], but found [{seqs[0]}]"}

            # DEBUG: whc
            elif os.path.exists(os.path.join(output_dir, dec, "uniref90_hits.sto")) and \
                    os.path.exists(os.path.join(output_dir, dec, "bfd_uniref_hits.a3m")) and \
                    os.path.exists(os.path.join(output_dir, dec, "mgnify_hits.sto")):
                msa_feature = parse_protein_alignment_dir(os.path.join(output_dir, dec))

                sequence = "".join([amino_acid_3to1[standard_ccds[i]] for i in msa_feature["msa"][0]])
                md5_string = convert_md5_string(f"protein:{sequence}")
                if md5 == md5_string:
                    feature = {
                        "msa": msa_feature["msa"][:max_seq].astype(np.int8),
                        "deletion_matrix": msa_feature["deletion_matrix"][:max_seq].astype(np.int8),
                        "msa_species_identifiers": msa_feature["msa_species_identifiers"][:max_seq]
                    }
                    dump_pkl(feature, pkl_save_path)
                    return dict()
                else:
                    return {input_fasta_path: f"seqs not equal, asset [{sequence}], but found [{seqs[0]}]"}

            elif os.path.exists(os.path.join(output_dir, dec, "uniref90_hits.sto")) and \
                    os.path.exists(os.path.join(output_dir, dec, "bfd_uniclust30_hits.a3m")):
                msa_feature = parse_protein_alignment_dir(os.path.join(output_dir, dec))
                if len(msa_feature["msa"]) < max_seq:
                    return {
                        input_fasta_path: f"MSA is not enough!"
                    }
                sequence = "".join([amino_acid_3to1[standard_ccds[i]] for i in msa_feature["msa"][0]])
                md5_string = convert_md5_string(f"protein:{sequence}")
                if md5 == md5_string:
                    feature = {
                        "msa": msa_feature["msa"][:max_seq].astype(np.int8),
                        "deletion_matrix": msa_feature["deletion_matrix"][:max_seq].astype(np.int8),
                        "msa_species_identifiers": msa_feature["msa_species_identifiers"][:max_seq]
                    }
                    dump_pkl(feature, pkl_save_path)
                    return dict()
                else:
                    return {input_fasta_path: f"seqs not equal, asset [{sequence}], but found [{seqs[0]}]"}
            elif os.path.exists(os.path.join(output_dir, dec, "uniref90_hits.sto")) and \
                    os.path.exists(os.path.join(output_dir, dec, "mgnify_hits.sto")):
                msa_feature = parse_protein_alignment_dir(os.path.join(output_dir, dec))

                sequence = "".join([amino_acid_3to1[standard_ccds[i]] for i in msa_feature["msa"][0]])
                md5_string = convert_md5_string(f"protein:{sequence}")
                if md5 == md5_string:
                    feature = {
                        "msa": msa_feature["msa"][:max_seq].astype(np.int8),
                        "deletion_matrix": msa_feature["deletion_matrix"][:max_seq].astype(np.int8),
                        "msa_species_identifiers": msa_feature["msa_species_identifiers"][:max_seq]
                    }
                    dump_pkl(feature, pkl_save_path)
                    return dict()
                else:
                    return {input_fasta_path: f"seqs not equal, asset [{sequence}], but found [{seqs[0]}]"}

            else:
                # msa_feature = parse_protein_alignment_dir(os.path.join(output_dir, dec))
                #
                # sequence = "".join([amino_acid_3to1[standard_ccds[i]] for i in msa_feature["msa"][0]])
                # md5_string = convert_md5_string(f"protein:{sequence}")
                # if md5 == md5_string:
                #     feature = {
                #         "msa": msa_feature["msa"][:max_seq].astype(np.int8),
                #         "deletion_matrix": msa_feature["deletion_matrix"][:max_seq].astype(np.int8),
                #         "msa_species_identifiers": msa_feature["msa_species_identifiers"][:max_seq]
                #     }
                #     dump_pkl(feature, pkl_save_path)
                #     return dict()
                # else:
                #     return {input_fasta_path: f"seqs not equal, asset [{sequence}], but found [{seqs[0]}]"}

                return {
                    input_fasta_path: f"MSA is not enough!"
                }

        out = run_pool_tasks(partial(
            _convert_msas_out_to_msa_features,
            output_dir=output_dir,
            msa_feature_dir=msa_feature_dir,
            convert_md5=convert_md5
        ), input_fasta_paths, num_workers=num_workers, return_dict=True)
        return out

    @staticmethod
    def convert_msas_out_to_uniprot_msa_features(
            input_fasta_path,
            output_dir,
            uniprot_msa_feature_dir,
            convert_md5=True,
            num_workers=128
    ):
        if isinstance(input_fasta_path, list):
            input_fasta_paths = input_fasta_path
        elif os.path.isdir(input_fasta_path):
            input_fasta_paths = [os.path.join(input_fasta_path, i) for i in os.listdir(input_fasta_path)]
        elif os.path.isfile(input_fasta_path):
            input_fasta_paths = [input_fasta_path]
        else:
            input_fasta_paths = []
            Exception("Can't parse input fasta path!")

        global _convert_msas_out_to_uniprot_msa_features

        def _convert_msas_out_to_uniprot_msa_features(
                input_fasta_path,
                output_dir,
                uniprot_msa_feature_dir,
                convert_md5=True,
        ):
            prefix = "protein"
            max_seq = 50000
            seqs, decs = parse_fasta(load_txt(input_fasta_path))
            md5 = convert_md5_string(f"{prefix}:{seqs[0]}")
            if convert_md5:
                # TODO: debug
                # dec = convert_md5_string(f"{prefix}:{input_fasta_path}")
                dec = md5
            else:
                dec = os.path.split(input_fasta_path)[1].split(".")[0]

            pkl_save_path = os.path.join(uniprot_msa_feature_dir, f"{md5}.pkl.gz")

            if os.path.exists(pkl_save_path):
                return dict()
            if os.path.exists(os.path.join(output_dir, dec, "uniprot_hits.sto")):
                msa_feature = parse_uniprot_alignment_dir(os.path.join(output_dir, dec))

                sequence = "".join([amino_acid_3to1[standard_ccds[i]] for i in msa_feature["msa_all_seq"][0]])
                md5_string = convert_md5_string(f"protein:{sequence}")
                if md5 == md5_string:
                    feature = {
                        "msa_all_seq": msa_feature["msa_all_seq"][:max_seq].astype(np.int8),
                        "deletion_matrix_all_seq": msa_feature["deletion_matrix_all_seq"][:max_seq].astype(np.int8),
                        "msa_species_identifiers_all_seq": msa_feature["msa_species_identifiers_all_seq"][:max_seq]
                    }
                    dump_pkl(feature, pkl_save_path)
                    return dict()
                else:
                    return {input_fasta_path: f"seqs not equal, asset [{sequence}], but found [{seqs[0]}]"}

            else:
                return {
                    input_fasta_path: f"MSA is not enough!"
                }

        out = run_pool_tasks(partial(
            _convert_msas_out_to_uniprot_msa_features,
            output_dir=output_dir,
            uniprot_msa_feature_dir=uniprot_msa_feature_dir,
            convert_md5=convert_md5
        ), input_fasta_paths, num_workers=num_workers, return_dict=True)
        return out

    @staticmethod
    def convert_msas_out_to_rna_msa_features(
            input_fasta_path,
            output_dir,
            rna_msa_feature_dir,
            convert_md5=True,
            num_workers=128
    ):
        import os
        os.makedirs(rna_msa_feature_dir, exist_ok=True)
        if isinstance(input_fasta_path, list):
            input_fasta_paths = input_fasta_path
        elif os.path.isdir(input_fasta_path):
            input_fasta_paths = [os.path.join(input_fasta_path, i) for i in os.listdir(input_fasta_path)]
        elif os.path.isfile(input_fasta_path):
            input_fasta_paths = [input_fasta_path]
        else:
            input_fasta_paths = []
            Exception("Can't parse input fasta path!")

        global _convert_msas_out_to_rna_msa_features

        def _convert_msas_out_to_rna_msa_features(
                input_fasta_path,
                output_dir,
                rna_msa_feature_dir,
                convert_md5=True,
        ):
            prefix = "rna"
            max_seq = 16384
            seqs, decs = parse_fasta(load_txt(input_fasta_path))
            md5 = convert_md5_string(f"{prefix}:{seqs[0]}")
            if convert_md5:
                # TODO: debug
                # dec = convert_md5_string(f"{prefix}:{input_fasta_path}")
                dec = md5
            else:
                dec = os.path.split(input_fasta_path)[1].split(".")[0]

            # DEBUG: whc homo search hits
            if os.path.exists(os.path.join(output_dir, dec, "msas")):
                dec = dec + "/msas"

            pkl_save_path = os.path.join(rna_msa_feature_dir, f"{md5}.pkl.gz")

            if os.path.exists(pkl_save_path):
                return dict()
            rna_msa_feature = parse_rna_alignment_dir(
                os.path.join(output_dir, dec),
                input_fasta_path
            )

            feature = {
                "msa": rna_msa_feature["msa"][:max_seq].astype(np.int8),
                "deletion_matrix": rna_msa_feature["deletion_matrix"][:max_seq].astype(np.int8),
                "msa_species_identifiers": None
            }
            dump_pkl(feature, pkl_save_path)

            return dict()

        out = run_pool_tasks(partial(
            _convert_msas_out_to_rna_msa_features,
            output_dir=output_dir,
            rna_msa_feature_dir=rna_msa_feature_dir,
            convert_md5=convert_md5
        ), input_fasta_paths, num_workers=num_workers, return_dict=True)
        return out

    @staticmethod
    def find_chain_ids_without_msa_features(
            polymer_filtering_out_json,
            chain_id_to_meta_info_path,
            dataset_dir,
            uniprot=False,
            num_workers=256,
    ):
        if not isinstance(polymer_filtering_out_json, list):
            polymer_filtering_out_json = [polymer_filtering_out_json]
        polymer_filtering_out = dict()
        for i in polymer_filtering_out_json:
            polymer_filtering_out.update(load_json(i))
        chain_id_to_meta_info = load_json(chain_id_to_meta_info_path)
        global find_chain_ids_without_msa_features

        def find_chain_ids_without_msa_features(chain_id):
            sequence_3 = chain_id_to_meta_info[chain_id]["sequence_3"]
            seqs, md5s = get_protein_md5(sequence_3)
            if uniprot:
                dirs = ["uniprot_msa_features", "uniprot_msa_features_zkx", "uniprot_msa_features_unifold"]
            else:
                dirs = ["msa_features", "msa_features_zkx", "msa_features_whc", "msa_features_unifold"]
            md5_dir = [[md5, dir] for dir in dirs for md5 in md5s]
            for md5, dir in md5_dir:
                if os.path.exists(os.path.join(dataset_dir, "features/", dir, f"{md5}.pkl.gz")):
                    return dict()
            return {chain_id: {"state": False, "seqs": seqs}}

        chain_ids = [k for k, v in polymer_filtering_out.items() if
                     v["state"] and chain_id_to_meta_info[k]["chain_class"] == "protein"]

        out = run_pool_tasks(
            find_chain_ids_without_msa_features, chain_ids, num_workers=num_workers, return_dict=True)
        return out

    def find_chain_ids_without_rna_msa_features(
            self,
            polymer_filtering_out_json,
    ):
        pass

    @staticmethod
    def check_msa_md5(msa_feature_dir):
        pass

    @staticmethod
    def check_uniprot_msa_md5(uniprot_msa_feature_dir):
        pass

    @staticmethod
    def check_rna_msa_md5(rna_msa_feature_dir):
        pass

    def get_training_pdbs(self):
        pass








class DataPipeline():
    def __init__(self):
        super().__init__()
        self.data_manager = DatasetManager()

# PDB:
#     polymer_chain_id:
#             weight_chain: contiguous_crop:1/3 spatial_crop: 2/3
#     Interface
#             weight_interface:
#         [chain_id1, chain_id2] 0.2 contiguous_crop
#         [chain_id1, chain_id2] 0.4 spatial_crop_interface
#         [ < 20 chains]         0.4 spatial crop
#
#
#
#
#     polymer chain contiguous crop sample weight w_chain*1/3  [chain_id]
#     polymer chain spatial crop sample weight w_chain*2/3 [chain_id]
#
#     interface contiguous crop sample weight w_interface * 0.2 [chain_id, chain_id]
#     interface spatial crop sample weight w_interface * 0.4 >[chain_id, chain_id]
#     interface spatial crop interface sample weight w_interface * 0.4 [chain_id, chain_id]
#
#
# pdb:
#     chain:
#         [chain_id]: 0.14
#         [chain_id]: 0.23


# @staticmethod
# def get_pdb_info(pdb_id):
#     all_chain_ids = pdb_id_to_meta_info[pdb_id]["chain_ids"]
#
#     chain_ids_info = {
#         "protein": [],
#         "rna": [],
#         "dna": [],
#         "ligand": []
#     }
#     for chain_id_ in all_chain_ids:
#         chain_id = f"{pdb_id}_{chain_id_}"
#         if chain_id in chain_id_to_meta_info:
#             chain_class = chain_id_to_meta_info[chain_id]["chain_class"].split("_")[0]
#             if chain_id in chain_ids and os.path.exists(os.path.join(stfold_data_path, f"{chain_id}.pkl.gz")):
#                 if chain_class == "protein":
#                     if check_protein_msa_features(chain_id, chain_id_to_meta_info)[chain_id]["state"]:
#                         chain_ids_info[chain_class].append(chain_id)
#                 elif chain_class == "rna":
#                     if check_rna_msa_features(chain_id, chain_id_to_meta_info)[chain_id]["state"]:
#                         chain_ids_info[chain_class].append(chain_id)
#
#             elif chain_class == "ligand" and os.path.exists(os.path.join(stfold_data_path, f"{chain_id}.pkl.gz")):
#                 chain_ids_info[chain_class].append(chain_id)
#     return {pdb_id: chain_ids_info}
