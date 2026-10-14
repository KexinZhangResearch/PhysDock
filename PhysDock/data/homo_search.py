import logging
import os.path
import shutil
from functools import partial
import tqdm
from typing import Optional, Mapping, Any, Union

from PhysDock.data.tools import jackhmmer, nhmmer, hhblits, kalign, hmmalign, parsers, hmmbuild, hhsearch, templates
from PhysDock.utils.io_utils import load_pkl, load_txt, load_json, run_pool_tasks, convert_md5_string, dump_pkl
from PhysDock.data.tools.parsers import parse_fasta
from PhysDock.data.tools.dataset_manager import DatasetManager

TemplateSearcher = Union[hhsearch.HHSearch]


class AlignmentRunner:
    def __init__(
            self,
            # Databases
            uniref90_database_path: Optional[str] = None,
            uniprot_database_path: Optional[str] = None,
            uniclust30_database_path: Optional[str] = None,
            bfd_database_path: Optional[str] = None,
            mgnify_database_path: Optional[str] = None,

            # Homo Search Tools
            jackhmmer_binary_path: str = "/usr/bin/jackhmmer",
            hhblits_binary_path: str = "/usr/bin/hhblits",

            # Params
            no_cpus: int = 8,

            # Thresholds
            uniref90_seq_limit: int = 100000,
            uniprot_seq_limit: int = 500000,
            mgnify_seq_limit: int = 50000,
            uniref90_max_hits: int = 10000,
            uniprot_max_hits: int = 50000,
            mgnify_max_hits: int = 5000,
    ):
        self.uniref90_jackhmmer_runner = None
        self.uniprot_jackhmmer_runner = None
        self.mgnify_jackhmmer_runner = None
        self.bfd_uniref30_hhblits_runner = None
        self.bfd_uniclust30_hhblits_runner = None

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
        if _all_exists(hhblits_binary_path, bfd_database_path, uniclust30_database_path, hhblits_mode=True):
            self.bfd_uniclust30_hhblits_runner = partial(
                _run_msa_tool,
                msa_runner=hhblits.HHBlits(
                    binary_path=hhblits_binary_path,
                    databases=[bfd_database_path, uniclust30_database_path],
                    n_cpu=no_cpus,
                ),
                msa_format="a3m",
            )

    def run(self, input_fasta_path, output_msas_dir, use_precompute=True):
        os.makedirs(output_msas_dir, exist_ok=True)
        uniref90_out_path = os.path.join(output_msas_dir, "uniref90_hits.sto")
        uniprot_out_path = os.path.join(output_msas_dir, "uniprot_hits.sto")
        mgnify_out_path = os.path.join(output_msas_dir, "mgnify_hits.sto")
        bfd_uniclust30_out_path = os.path.join(output_msas_dir, f"bfd_uniclust30_hits.a3m")

        seqs, decs = parse_fasta(load_txt(input_fasta_path))
        prefix = "protein"
        md5 = convert_md5_string(f"{prefix}:{seqs[0]}")
        output_feature = os.path.dirname(output_msas_dir)
        output_feature = os.path.dirname(output_feature)

        pkl_save_path_msa = os.path.join(output_feature, "msa_features", f"{md5}.pkl.gz")
        pkl_save_path_msa_uni = os.path.join(output_feature, "uniprot_msa_features", f"{md5}.pkl.gz")

        if self.uniref90_jackhmmer_runner is not None and not os.path.exists(pkl_save_path_msa):
            if not os.path.exists(uniref90_out_path) or not use_precompute:
                self.uniref90_jackhmmer_runner(input_fasta_path, uniref90_out_path)

        if self.uniprot_jackhmmer_runner is not None and not os.path.exists(pkl_save_path_msa_uni):
            if not os.path.exists(uniprot_out_path) or not use_precompute:
                self.uniprot_jackhmmer_runner(input_fasta_path, uniprot_out_path)
        if self.mgnify_jackhmmer_runner is not None and not os.path.exists(pkl_save_path_msa):
            if not os.path.exists(mgnify_out_path) or not use_precompute:
                self.mgnify_jackhmmer_runner(input_fasta_path, mgnify_out_path)
        if self.bfd_uniclust30_hhblits_runner is not None and not os.path.exists(pkl_save_path_msa):
            if not os.path.exists(bfd_uniclust30_out_path) or not use_precompute:
                self.bfd_uniclust30_hhblits_runner(input_fasta_path, bfd_uniclust30_out_path)


class DataProcessor:
    def __init__(
            self,
            bfd_database_path,
            uniclust30_database_path,
            uniref90_database_path,
            mgnify_database_path,
            uniprot_database_path,
            jackhmmer_binary_path: Optional[str] = None,
            hhblits_binary_path: Optional[str] = None,

            n_cpus: int = 8,
            n_workers: int = 1,
    ):
        '''
        '''
        self.jackhmmer_binary_path = jackhmmer_binary_path
        self.hhblits_binary_path = hhblits_binary_path

        self.n_cpus = n_cpus
        self.n_workers = n_workers

        self.uniref90_database_path = uniref90_database_path
        self.uniprot_database_path = uniprot_database_path
        self.bfd_database_path = bfd_database_path
        self.uniclust30_database_path = uniclust30_database_path
        self.mgnify_database_path = mgnify_database_path

        # self.uniref90_database_path = os.path.join(
        #     alphafold3_database_path, "uniref90", "uniref90.fasta"
        # )
        # self.uniprot_database_path = os.path.join(
        #     alphafold3_database_path, "uniprot", "uniprot.fasta"
        # )
        # self.bfd_database_path = os.path.join(
        #     alphafold3_database_path, "bfd", "bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"
        # )
        # self.uniclust30_database_path = os.path.join(
        #     alphafold3_database_path, "uniclust30", "uniclust30_2018_08", "uniclust30_2018_08"
        # )
        #
        # self.mgnify_database_path = os.path.join(
        #     alphafold3_database_path, "mgnify", "mgnify", "mgy_clusters.fa"
        # )

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

    def _process_iotuple(self, io_tuple, use_precompute=True):
        i, o = io_tuple
        kwargs = {
            "jackhmmer_binary_path": self.jackhmmer_binary_path,
            "hhblits_binary_path": self.hhblits_binary_path,
            "uniref90_database_path": self.uniref90_database_path,
            "bfd_database_path": self.bfd_database_path,
            "uniclust30_database_path": self.uniclust30_database_path,
            "mgnify_database_path": self.mgnify_database_path,
            "uniprot_database_path": self.uniprot_database_path,
        }
        alignment_runner = AlignmentRunner(
            **kwargs,
            no_cpus=self.n_cpus
        )
        try:
            alignment_runner.run(i, o, use_precompute=use_precompute)
        except:
            logging.warning(f"{i}:{o} task failed!")

    def process(self, input_fasta_path, output_dir, convert_md5=True, use_precompute=True):
        prefix = "protein"
        io_tuples = self._parse_io_tuples(input_fasta_path, output_dir, convert_md5=convert_md5, prefix=prefix)
        run_pool_tasks(partial(self._process_iotuple, use_precompute=use_precompute), io_tuples,
                       num_workers=self.n_workers,
                       return_dict=False)

    def convert_output_to_md5(self, input_fasta_path, output_dir, md5_output_dir, prefix="protein"):
        io_tuples = self._parse_io_tuples(input_fasta_path, output_dir, convert_md5=False, prefix=prefix)
        io_tuples_md5 = self._parse_io_tuples(input_fasta_path, md5_output_dir, convert_md5=True, prefix=prefix)

        for io0, io1 in tqdm.tqdm(zip(io_tuples, io_tuples_md5)):
            o, o_md5 = io0[1], io1[1]
            os.system(f"cp -r {os.path.abspath(o)} {os.path.abspath(o_md5)}")


def run_homo_search(
        bfd_database_path,
        uniclust30_database_path,
        uniref90_database_path,
        mgnify_database_path,
        uniprot_database_path,
        jackhmmer_binary_path,
        hhblits_binary_path,

        input_fasta_path,
        out_dir,

        n_cpus=16,
        n_workers=1,
):
    # save_dir = os.path.join(out_dir,"cache")
    data_processor = DataProcessor(
        bfd_database_path,
        uniclust30_database_path,
        uniref90_database_path,
        mgnify_database_path,
        uniprot_database_path,
        jackhmmer_binary_path=jackhmmer_binary_path,
        hhblits_binary_path=hhblits_binary_path,
        n_cpus=n_cpus,
        n_workers=n_workers
    )

    output_dir = os.path.join(out_dir, "msas")
    os.makedirs(output_dir, exist_ok=True)
    if os.path.isfile(input_fasta_path):
        files = [input_fasta_path]
    else:
        files = os.listdir(input_fasta_path)
        files = [os.path.join(input_fasta_path, file) for file in files[::-1]]

    data_processor.process(
        input_fasta_path=files,
        output_dir=output_dir,
        convert_md5=True
    )
    print(f"save msa to {output_dir}")

    msa_dir = os.path.join(out_dir, "msa_features")
    os.makedirs(msa_dir, exist_ok=True)

    out = DatasetManager.convert_msas_out_to_msa_features(
        input_fasta_path=input_fasta_path,
        output_dir=output_dir,
        msa_feature_dir=msa_dir,
        convert_md5=True,
        num_workers=2
    )
    print(f"save msa feature to {msa_dir}")

    msa_dir_uni = os.path.join(out_dir, "uniprot_msa_features")
    os.makedirs(msa_dir_uni, exist_ok=True)
    out = DatasetManager.convert_msas_out_to_uniprot_msa_features(
        input_fasta_path=input_fasta_path,
        output_dir=output_dir,
        uniprot_msa_feature_dir=msa_dir_uni,
        convert_md5=True,
        num_workers=2
    )
    print(f"save uni msa feature to {msa_dir_uni}")
