import logging
import os.path
from functools import partial
import tqdm
from typing import Optional, Mapping, Any, Union

from PhysDock.data.tools import jackhmmer, nhmmer, hhblits, kalign, hmmalign, parsers, hmmbuild, hhsearch, templates
from PhysDock.utils.io_utils import load_pkl, load_txt, load_json, run_pool_tasks, convert_md5_string, dump_pkl
from PhysDock.data.tools.parsers import parse_fasta

TemplateSearcher = Union[hhsearch.HHSearch]


class AlignmentRunner:
    def __init__(
            self,
            # Homo Search Tools
            jackhmmer_binary_path: Optional[str] = None,
            hhblits_binary_path: Optional[str] = None,
            nhmmer_binary_path: Optional[str] = None,
            hmmbuild_binary_path: Optional[str] = None,
            hmmalign_binary_path: Optional[str] = None,
            kalign_binary_path: Optional[str] = None,

            # Templ Search Tools
            hhsearch_binary_path: Optional[str] = None,
            template_searcher: Optional[TemplateSearcher] = None,
            template_featurizer: Optional[templates.TemplateHitFeaturizer] = None,

            # Databases
            uniref90_database_path: Optional[str] = None,
            uniprot_database_path: Optional[str] = None,
            uniclust30_database_path: Optional[str] = None,
            uniref30_database_path: Optional[str] = None,
            bfd_database_path: Optional[str] = None,
            reduced_bfd_database_path: Optional[str] = None,
            mgnify_database_path: Optional[str] = None,
            rfam_database_path: Optional[str] = None,
            rnacentral_database_path: Optional[str] = None,
            nt_database_path: Optional[str] = None,
            #
            no_cpus: int = 8,
            # Limitations
            uniref90_seq_limit: int = 100000,
            uniprot_seq_limit: int = 500000,
            reduced_bfd_seq_limit: int = 50000,
            mgnify_seq_limit: int = 50000,
            uniref90_max_hits: int = 10000,
            uniprot_max_hits: int = 50000,
            reduced_bfd_max_hits: int = 5000,
            mgnify_max_hits: int = 5000,
            rfam_max_hits: int = 10000,
            rnacentral_max_hits: int = 10000,
            nt_max_hits: int = 10000,
    ):
        self.uniref90_jackhmmer_runner = None
        self.uniprot_jackhmmer_runner = None
        self.reduced_bfd_jackhmmer_runner = None
        self.mgnify_jackhmmer_runner = None
        self.bfd_uniref30_hhblits_runner = None
        self.bfd_uniclust30_hhblits_runner = None
        self.rfam_nhmmer_runner = None
        self.rnacentral_nhmmer_runner = None
        self.nt_nhmmer_runner = None
        self.rna_realign_runner = None
        self.template_searcher = template_searcher
        self.template_featurizer = template_featurizer

        def _all_exists(*objs, hhblits_mode=False):
            if not hhblits_mode:
                for obj in objs:
                    if obj is None or not os.path.exists(obj):
                        return False
            else:
                for obj in objs:
                    if obj is None or not os.path.exists(os.path.split(obj)[0]):
                        return False
            return True

        def _run_msa_tool(
                fasta_path: str,
                msa_out_path: str,
                msa_runner,
                msa_format: str,
                max_sto_sequences: Optional[int] = None,
        ) -> Mapping[str, Any]:
            """Runs an MSA tool, checking if output already exists first."""
            if (msa_format == "sto" and max_sto_sequences is not None):
                result = msa_runner.query(fasta_path, max_sto_sequences)[0]
            else:
                result = msa_runner.query(fasta_path)[0]

            assert msa_out_path.split('.')[-1] == msa_format
            with open(msa_out_path, "w") as f:
                f.write(result[msa_format])

            return result

        def _run_rna_realign_tool(
                fasta_path: str,
                msa_in_path: str,
                msa_out_path: str,
                use_precompute=True,
        ):
            runner = hmmalign.Hmmalign(
                hmmbuild_binary_path=hmmbuild_binary_path,
                hmmalign_binary_path=hmmalign_binary_path,
            )
            if os.path.exists(msa_in_path) and os.path.getsize(msa_in_path) == 0:
                # print("MSA sto file is 0")
                with open(msa_out_path, "w") as f:
                    pass
                return
            if use_precompute:
                if os.path.exists(msa_in_path) and os.path.exists(msa_out_path):
                    if os.path.getsize(msa_in_path) > 0 and os.path.getsize(msa_out_path) == 0:
                        logging.warning(f"The msa realign file size is zero but the origin file size is over 0! "
                                        f"fasta: {fasta_path} msa_in_file: {msa_in_path}")
                        runner.realign_sto_with_fasta(fasta_path, msa_in_path, msa_out_path)
                else:
                    runner.realign_sto_with_fasta(fasta_path, msa_in_path, msa_out_path)
            else:
                runner.realign_sto_with_fasta(fasta_path, msa_in_path, msa_out_path)
            # with open(msa_out_path, "w") as f:
            #     f.write(msa_out)

        assert uniclust30_database_path is None or uniref30_database_path is None, "Only one used"

        # Jackhmmer
        if _all_exists(jackhmmer_binary_path, uniref90_database_path):
            self.uniref90_jackhmmer_runner = partial(
                _run_msa_tool,
                msa_runner=jackhmmer.Jackhmmer(
                    binary_path=jackhmmer_binary_path,
                    database_path=uniref90_database_path,
                    seq_limit=uniref90_seq_limit,
                    n_cpu=no_cpus,
                ),
                msa_format="sto",
                max_sto_sequences=uniref90_max_hits
            )

        if _all_exists(jackhmmer_binary_path, uniprot_database_path):
            self.uniprot_jackhmmer_runner = partial(
                _run_msa_tool,
                msa_runner=jackhmmer.Jackhmmer(
                    binary_path=jackhmmer_binary_path,
                    database_path=uniprot_database_path,
                    seq_limit=uniprot_seq_limit,
                    n_cpu=no_cpus,
                ),
                msa_format="sto",
                max_sto_sequences=uniprot_max_hits
            )
        if _all_exists(jackhmmer_binary_path, reduced_bfd_database_path):
            self.reduced_bfd_jackhmmer_runner = partial(
                _run_msa_tool,
                msa_runner=jackhmmer.Jackhmmer(
                    binary_path=jackhmmer_binary_path,
                    database_path=reduced_bfd_database_path,
                    seq_limit=reduced_bfd_seq_limit,
                    n_cpu=no_cpus,
                ),
                msa_format="sto",
                max_sto_sequences=reduced_bfd_max_hits
            )

        if _all_exists(jackhmmer_binary_path, mgnify_database_path):
            self.mgnify_jackhmmer_runner = partial(
                _run_msa_tool,
                msa_runner=jackhmmer.Jackhmmer(
                    binary_path=jackhmmer_binary_path,
                    database_path=mgnify_database_path,
                    seq_limit=mgnify_seq_limit,
                    n_cpu=no_cpus,
                ),
                msa_format="sto",
                max_sto_sequences=mgnify_max_hits
            )

        # HHblits
        if _all_exists(hhblits_binary_path, bfd_database_path, uniref30_database_path, hhblits_mode=True):
            self.bfd_uniref30_hhblits_runner = partial(
                _run_msa_tool,
                msa_runner=hhblits.HHBlits(
                    binary_path=hhblits_binary_path,
                    databases=[bfd_database_path, uniref30_database_path],
                    n_cpu=no_cpus,
                ),
                msa_format="a3m",
            )
        elif _all_exists(hhblits_binary_path, bfd_database_path, uniclust30_database_path, hhblits_mode=True):
            self.bfd_uniclust30_hhblits_runner = partial(
                _run_msa_tool,
                msa_runner=hhblits.HHBlits(
                    binary_path=hhblits_binary_path,
                    databases=[bfd_database_path, uniclust30_database_path],
                    n_cpu=no_cpus,
                ),
                msa_format="a3m",
            )

        # Nhmmer
        if _all_exists(nhmmer_binary_path, rfam_database_path):
            self.rfam_nhmmer_runner = partial(
                _run_msa_tool,
                msa_runner=nhmmer.Nhmmer(
                    binary_path=nhmmer_binary_path,
                    database_path=rfam_database_path,
                    n_cpu=no_cpus
                ),
                msa_format="sto",
                max_sto_sequences=rfam_max_hits
            )
        if _all_exists(nhmmer_binary_path, rnacentral_database_path):
            self.rnacentral_nhmmer_runner = partial(
                _run_msa_tool,
                msa_runner=nhmmer.Nhmmer(
                    binary_path=nhmmer_binary_path,
                    database_path=rnacentral_database_path,
                    n_cpu=no_cpus
                ),
                msa_format="sto",
                max_sto_sequences=rnacentral_max_hits
            )
        if _all_exists(nhmmer_binary_path, nt_database_path):
            self.nt_nhmmer_runner = partial(
                _run_msa_tool,
                msa_runner=nhmmer.Nhmmer(
                    binary_path=nhmmer_binary_path,
                    database_path=nt_database_path,
                    n_cpu=no_cpus
                ),
                msa_format="sto",
                max_sto_sequences=nt_max_hits
            )

        # def _run_rna_hmm(
        #         fasta_path: str,
        #         hmm_out_path: str,
        # ):
        #     runner = hmmbuild.Hmmbuild(binary_path=hmmbuild_binary_path)
        #     hmm = runner.build_rna_profile_from_fasta(fasta_path)
        #     with open(hmm_out_path, "w") as f:
        #         f.write(hmm)

        if _all_exists(hmmbuild_binary_path, hmmalign_binary_path):
            self.rna_realign_runner = _run_rna_realign_tool

    def run(self, input_fasta_path, output_msas_dir, use_precompute=True):
        os.makedirs(output_msas_dir, exist_ok=True)
        templates_out_path = os.path.join(output_msas_dir, "templates")
        uniref90_out_path = os.path.join(output_msas_dir, "uniref90_hits.sto")
        uniprot_out_path = os.path.join(output_msas_dir, "uniprot_hits.sto")
        reduced_bfd_out_path = os.path.join(output_msas_dir, "reduced_bfd_hits.sto")
        mgnify_out_path = os.path.join(output_msas_dir, "mgnify_hits.sto")
        bfd_uniref30_out_path = os.path.join(output_msas_dir, f"bfd_uniref30_hits.a3m")
        bfd_uniclust30_out_path = os.path.join(output_msas_dir, f"bfd_uniclust30_hits.a3m")

        seqs, decs = parse_fasta(load_txt(input_fasta_path))
        prefix = "protein"
        md5 = convert_md5_string(f"{prefix}:{seqs[0]}")
        output_feature = os.path.dirname(output_msas_dir)
        output_feature = os.path.dirname(output_feature)
        pkl_save_path_msa = os.path.join(output_feature, "msa_features", f"{md5}.pkl.gz")
        pkl_save_path_msa_uni = os.path.join(output_feature, "uniprot_msa_features", f"{md5}.pkl.gz")
        pkl_save_path_temp = os.path.join(output_feature, "template_features", f"{md5}.pkl.gz")

        if self.uniref90_jackhmmer_runner is not None and not os.path.exists(pkl_save_path_temp):
            if not os.path.exists(uniref90_out_path) or not use_precompute or not os.path.exists(pkl_save_path_temp):
                if not os.path.exists(uniref90_out_path):
                    print(uniref90_out_path)
                    self.uniref90_jackhmmer_runner(input_fasta_path, uniref90_out_path)

                print("begin templates")
                if templates_out_path is not None \
                        and self.template_searcher is not None and self.template_featurizer is not None:
                    try:
                        os.makedirs(templates_out_path, exist_ok=True)
                        seq, dec = parsers.parse_fasta(load_txt(input_fasta_path))
                        input_sequence = seq[0]
                        msa_for_templates = parsers.truncate_stockholm_msa(
                            uniref90_out_path, max_sequences=10000
                        )
                        msa_for_templates = parsers.deduplicate_stockholm_msa(msa_for_templates)
                        msa_for_templates = parsers.remove_empty_columns_from_stockholm_msa(
                            msa_for_templates
                        )
                        if self.template_searcher.input_format == "sto":
                            pdb_templates_result = self.template_searcher.query(msa_for_templates)
                        elif self.template_searcher.input_format == "a3m":
                            uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(msa_for_templates)
                            pdb_templates_result = self.template_searcher.query(uniref90_msa_as_a3m)
                        else:
                            raise ValueError(
                                "Unrecognized template input format: "
                                f"{self.template_searcher.input_format}"
                            )

                        pdb_hits_out_path = os.path.join(
                            templates_out_path, f"pdb_hits.{self.template_searcher.output_format}.pkl.gz"
                        )
                        with open(os.path.join(
                                templates_out_path, f"pdb_hits.{self.template_searcher.output_format}"
                        ), "w") as f:
                            f.write(pdb_templates_result)

                        pdb_template_hits = self.template_searcher.get_template_hits(
                            output_string=pdb_templates_result, input_sequence=input_sequence
                        )
                        templates_result = self.template_featurizer.get_templates(
                            query_sequence=input_sequence, hits=pdb_template_hits
                        )
                    except Exception as e:
                        logging.exception("An error in template searching")

                    dump_pkl(templates_result.features, pdb_hits_out_path, compress=True)
        if self.uniprot_jackhmmer_runner is not None and not os.path.exists(pkl_save_path_msa_uni):
            if not os.path.exists(uniprot_out_path) or not use_precompute:
                self.uniprot_jackhmmer_runner(input_fasta_path, uniprot_out_path)
        if self.reduced_bfd_jackhmmer_runner is not None and not os.path.exists(pkl_save_path_msa):
            if not os.path.exists(reduced_bfd_out_path) or not use_precompute:
                self.reduced_bfd_jackhmmer_runner(input_fasta_path, reduced_bfd_out_path)
        if self.mgnify_jackhmmer_runner is not None and not os.path.exists(pkl_save_path_msa):
            if not os.path.exists(mgnify_out_path) or not use_precompute:
                self.mgnify_jackhmmer_runner(input_fasta_path, mgnify_out_path)
        if self.bfd_uniref30_hhblits_runner is not None and not os.path.exists(pkl_save_path_msa):
            if not os.path.exists(bfd_uniref30_out_path) or not use_precompute:
                self.bfd_uniref30_hhblits_runner(input_fasta_path, bfd_uniref30_out_path)
        if self.bfd_uniclust30_hhblits_runner is not None and not os.path.exists(pkl_save_path_msa):
            if not os.path.exists(bfd_uniclust30_out_path) or not use_precompute:
                self.bfd_uniclust30_hhblits_runner(input_fasta_path, bfd_uniclust30_out_path)
        # if self.rfam_nhmmer_runner is not None:
        #     if not os.path.exists(rfam_out_path) or not use_precompute:
        #         self.rfam_nhmmer_runner(input_fasta_path, rfam_out_path)
        # # print(self.rna_realign_runner is not None, os.path.exists(rfam_out_path))
        # if self.rna_realign_runner is not None and os.path.exists(rfam_out_path):
        #     self.rna_realign_runner(input_fasta_path, rfam_out_path, rfam_out_realigned_path)
        # if self.rnacentral_nhmmer_runner is not None:
        #     if not os.path.exists(rnacentral_out_path) or not use_precompute:
        #         self.rnacentral_nhmmer_runner(input_fasta_path, rnacentral_out_path)
        # if self.rna_realign_runner is not None and os.path.exists(rnacentral_out_path):
        #     self.rna_realign_runner(input_fasta_path, rnacentral_out_path, rnacentral_out_realigned_path)
        # if self.nt_nhmmer_runner is not None:
        #     if not os.path.exists(nt_out_path) or not use_precompute:
        #         self.nt_nhmmer_runner(input_fasta_path, nt_out_path)
        # if self.rna_realign_runner is not None and os.path.exists(nt_out_path):
        #     # print("realign",nt_out_path,nt_out_realigned_path)
        #     self.rna_realign_runner(input_fasta_path, nt_out_path, nt_out_realigned_path)


class DataProcessor:
    def __init__(
            self,
            alphafold3_database_path,
            jackhmmer_binary_path: Optional[str] = None,
            hhblits_binary_path: Optional[str] = None,
            nhmmer_binary_path: Optional[str] = None,
            kalign_binary_path: Optional[str] = None,
            hmmbuild_binary_path: Optional[str] = None,
            hmmalign_binary_path: Optional[str] = None,
            hhsearch_binary_path: Optional[str] = None,
            template_searcher: Optional[TemplateSearcher] = None,
            template_featurizer: Optional[templates.TemplateHitFeaturizer] = None,
            n_cpus: int = 8,
            n_workers: int = 1,
    ):
        '''
        Database Versions:
            Training:
                uniref90:   v2022_05
                bfd:
                reduces_bfd:
                uniclust30: v2018_08
                uniprot:    v2020_05
                mgnify:     v2022_05
                rfam:       v14.9
                rnacentral: v21.0
                nt:         v2023_02_23
            Inference:
                uniref90:   v2022_05
                bfd:
                reduces_bfd:
                uniclust30: v2018_08
                uniprot:    v2021_04   *
                mgnify:     v2022_05
                rfam:       v14.9
                rnacentral: v21.0
                nt:         v2023_02_23
            Inference Ligand:
                uniref90:   v2020_01   *
                bfd:
                reduces_bfd:
                uniclust30: v2018_08
                uniprot:    v2020_05
                mgnify:     v2018_12   *
                rfam:       v14.9
                rnacentral: v21.0
                nt:         v2023_02_23

        Args:
            alphafold3_database_path: Database dir that contains all alphafold3 databases
            jackhmmer_binary_path:
            hhblits_binary_path:
            nhmmer_binary_path:
            kalign_binary_path:
            hmmaligh_binary_path:
            n_cpus:
            n_workers:
        '''
        self.jackhmmer_binary_path = jackhmmer_binary_path
        self.hhblits_binary_path = hhblits_binary_path
        self.nhmmer_binary_path = nhmmer_binary_path
        self.hmmbuild_binary_path = hmmbuild_binary_path
        self.hmmalign_binary_path = hmmalign_binary_path
        self.hhsearch_binary_path = hhsearch_binary_path

        self.template_searcher = template_searcher
        self.template_featurizer = template_featurizer

        self.n_cpus = n_cpus
        self.n_workers = n_workers

        self.uniref90_database_path = os.path.join(
            alphafold3_database_path, "uniref90", "uniref90.fasta"
        )
        ################### TODO: DEBUG
        self.uniprot_database_path = os.path.join(
            alphafold3_database_path, "uniprot", "uniprot.fasta"
        )
        self.bfd_database_path = os.path.join(
            alphafold3_database_path, "bfd", "bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"
        )
        self.uniclust30_database_path = os.path.join(
            alphafold3_database_path, "uniclust30", "uniclust30_2018_08", "uniclust30_2018_08"
        )
        ################### TODO: check alphafold2 multimer uniref30 version
        self.uniref_30_database_path = os.path.join(
            alphafold3_database_path, "uniref30", "v2020_06"
        )
        # self.reduced_bfd_database_path = os.path.join(
        #     alphafold3_database_path,"reduced_bfd"
        # )

        self.mgnify_database_path = os.path.join(
            alphafold3_database_path, "mgnify", "mgnify", "mgy_clusters.fa"
        )
        self.rfam_database_path = os.path.join(
            alphafold3_database_path, "rfam", "v14.9", "Rfam_af3_clustered_rep_seq.fasta"
        )
        self.rnacentral_database_path = os.path.join(
            alphafold3_database_path, "rnacentral", "v21.0", "rnacentral_db_rep_seq.fasta"
        )

        self.nt_database_path = os.path.join(
            # alphafold3_database_path, "nt", "v2023_02_23", "nt_af3_clustered_rep_seq.fasta" # DEBUG
            alphafold3_database_path, "nt", "v2023_02_23", "nt.fasta"
        )

        self.runner_args_map = {
            "uniref90": {
                "jackhmmer_binary_path": self.jackhmmer_binary_path,
                "uniref90_database_path": self.uniref90_database_path,
            },
            "bfd_uniclust30": {
                "hhblits_binary_path": self.hhblits_binary_path,
                "bfd_database_path": self.bfd_database_path,
                "uniclust30_database_path": self.uniclust30_database_path
            },
            "bfd_uniref30": {
                "hhblits_binary_path": self.hhblits_binary_path,
                "bfd_database_path": self.bfd_database_path,
                "uniref_30_database_path": self.uniref_30_database_path
            },

            "mgnify": {
                "jackhmmer_binary_path": self.jackhmmer_binary_path,
                "mgnify_database_path": self.mgnify_database_path,
            },
            "uniprot": {
                "jackhmmer_binary_path": self.jackhmmer_binary_path,
                "uniprot_database_path": self.uniprot_database_path,
            },
            ###################### RNA ########################
            "rfam": {
                "nhmmer_binary_path": self.nhmmer_binary_path,
                "rfam_database_path": self.rfam_database_path,
                "hmmbuild_binary_path": self.hmmbuild_binary_path,
                "hmmalign_binary_path": self.hmmalign_binary_path,
            },
            "rnacentral": {
                "nhmmer_binary_path": self.nhmmer_binary_path,
                "rnacentral_database_path": self.rnacentral_database_path,
                "hmmbuild_binary_path": self.hmmbuild_binary_path,
                "hmmalign_binary_path": self.hmmalign_binary_path,
            },
            "nt": {
                "nhmmer_binary_path": self.nhmmer_binary_path,
                "nt_database_path": self.nt_database_path,
                "hmmbuild_binary_path": self.hmmbuild_binary_path,
                "hmmalign_binary_path": self.hmmalign_binary_path,
            },

            ###################################################
            "alphafold2": {
                "jackhmmer_binary_path": self.jackhmmer_binary_path,
                "hhblits_binary_path": self.hhblits_binary_path,
                "uniref90_database_path": self.uniref90_database_path,
                "bfd_database_path": self.bfd_database_path,
                "uniclust30_database_path": self.uniclust30_database_path,
                "mgnify_database_path": self.mgnify_database_path,
            },
            "alphafold2_multimer": {
                "jackhmmer_binary_path": self.jackhmmer_binary_path,
                "hhblits_binary_path": self.hhblits_binary_path,
                "uniref90_database_path": self.uniref90_database_path,
                "bfd_database_path": self.bfd_database_path,
                "uniref_30_database_path": self.uniref_30_database_path,
                "mgnify_database_path": self.mgnify_database_path,
                "uniprot_database_path": self.uniprot_database_path,
            },
            "alphafold3": {
                "jackhmmer_binary_path": self.jackhmmer_binary_path,
                "hhblits_binary_path": self.hhblits_binary_path,
                "template_searcher": self.template_searcher,
                "template_featurizer": self.template_featurizer,
                "uniref90_database_path": self.uniref90_database_path,
                "bfd_database_path": self.bfd_database_path,
                "uniclust30_database_path": self.uniclust30_database_path,
                "mgnify_database_path": self.mgnify_database_path,
                "uniprot_database_path": self.uniprot_database_path,
            },

            "rna": {
                "nhmmer_binary_path": self.nhmmer_binary_path,
                "rfam_database_path": self.rfam_database_path,
                "rnacentral_database_path": self.rnacentral_database_path,
                "hmmbuild_binary_path": self.hmmbuild_binary_path,
                "hmmalign_binary_path": self.hmmalign_binary_path,
            },
        }

    def _parse_io_tuples(self, input_fasta_path, output_dir, convert_md5=True, prefix="protein"):
        os.makedirs(output_dir, exist_ok=True)
        if isinstance(input_fasta_path, list):
            input_fasta_paths = input_fasta_path
        elif os.path.isdir(input_fasta_path):
            input_fasta_paths = [os.path.join(input_fasta_path, i) for i in os.listdir(input_fasta_path)]
        elif os.path.isfile(input_fasta_path):
            input_fasta_paths = [input_fasta_path]
        else:
            input_fasta_paths = []
            Exception("Can't parse input fasta path!")
        seqs = [parse_fasta(load_txt(i))[0][0] for i in input_fasta_paths]
        # sequences = [parsers.parse_fasta(load_txt(path))[0][0] for path in input_fasta_paths]
        # TODO: debug
        if convert_md5:
            output_msas_dirs = [os.path.join(output_dir, convert_md5_string(f"{prefix}:{i}")) for i in
                                seqs]
        else:
            output_msas_dirs = [os.path.join(output_dir, os.path.split(i)[1].split(".")[0]) for i in input_fasta_paths]
        io_tuples = [(i, o) for i, o in zip(input_fasta_paths, output_msas_dirs)]
        return io_tuples

    def _process_iotuple(self, io_tuple, msas_type):
        i, o = io_tuple
        alignment_runner = AlignmentRunner(
            **self.runner_args_map[msas_type],
            no_cpus=self.n_cpus
        )
        try:
            alignment_runner.run(i, o)
        except:
            logging.warning(f"{i}:{o} task failed!")

    def process(self, input_fasta_path, output_dir, msas_type="rfam", convert_md5=True):
        prefix = "rna" if msas_type in ["rfam", "rnacentral", "nt", "rna"] else "protein"
        io_tuples = self._parse_io_tuples(input_fasta_path, output_dir, convert_md5=convert_md5, prefix=prefix)
        run_pool_tasks(partial(self._process_iotuple, msas_type=msas_type), io_tuples, num_workers=self.n_workers,
                       return_dict=False)

    def convert_output_to_md5(self, input_fasta_path, output_dir, md5_output_dir, prefix="protein"):
        io_tuples = self._parse_io_tuples(input_fasta_path, output_dir, convert_md5=False, prefix=prefix)
        io_tuples_md5 = self._parse_io_tuples(input_fasta_path, md5_output_dir, convert_md5=True, prefix=prefix)

        for io0, io1 in tqdm.tqdm(zip(io_tuples, io_tuples_md5)):
            o, o_md5 = io0[1], io1[1]
            os.system(f"cp -r {os.path.abspath(o)} {os.path.abspath(o_md5)}")
