import argparse
from PhysDock.data.alignment_runner_v2 import run_homo_search

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_fasta_path",
        type=str,
    )
    parser.add_argument(
        "--features_dir",
        type=str,
    )

    parser.add_argument(
        "--bfd_database_path",
        type=str,
    )
    parser.add_argument(
        "--uniclust30_database_path",
        type=str,
    )
    parser.add_argument(
        "--uniref90_database_path",
        type=str,
    )
    parser.add_argument(
        "--mgnify_database_path",
        type=str,
    )
    parser.add_argument(
        "--uniprot_database_path",
        type=str,
    )
    parser.add_argument(
        "--jackhmmer_binary_path",
        type=str,
        default="/usr/bin/jackhmmer"
    )
    parser.add_argument(
        "--hhblits_binary_path",
        type=str,
        default="/usr/bin/hhblits"
    )
    parser.add_argument(
        "--n_cpus",
        type=int,
        default=16
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1
    )

    args = parser.parse_args()

    run_homo_search(
        bfd_database_path=args.bfd_database_path,
        uniclust30_database_path=args.uniclust30_database_path,
        uniref90_database_path=args.uniref90_database_path,
        mgnify_database_path=args.mgnify_database_path,
        uniprot_database_path=args.uniprot_database_path,
        jackhmmer_binary_path=args.jackhmmer_binary_path,
        hhblits_binary_path=args.hhblits_binary_path,

        input_fasta_path=args.input_fasta_path,
        out_dir=args.features_dir,
        n_cpus=16,
        n_workers=1,
    )
