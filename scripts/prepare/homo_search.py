#!/usr/bin/env python3
"""MSA search (jackhmmer / hhblits) for the receptor chains of prepared systems.

    python scripts/prepare/homo_search.py --input_fasta systems/fastas --features_dir features \\
        --bfd ... --uniclust30 ... --uniref90 ... --mgnify ... --uniprot ...

Writes <features_dir>/msa_features/ and <features_dir>/uniprot_msa_features/. The databases come from
scripts/download/download_homo_data.py (data/homo_data).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from PhysDock.data.homo_search import run_homo_search  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_fasta", required=True, help="FASTA file or folder of FASTA files")
    ap.add_argument("--features_dir", required=True)
    for db in ("bfd", "uniclust30", "uniref90", "mgnify", "uniprot"):
        ap.add_argument(f"--{db}", required=True, help=f"{db} database")
    ap.add_argument("--jackhmmer", default="/usr/bin/jackhmmer", help="jackhmmer binary")
    ap.add_argument("--hhblits", default="/usr/bin/hhblits", help="hhblits binary")
    ap.add_argument("--n_cpus", type=int, default=16, help="CPUs per search")
    ap.add_argument("--n_workers", type=int, default=1, help="parallel searches")
    a = ap.parse_args()
    run_homo_search(bfd_database_path=a.bfd, uniclust30_database_path=a.uniclust30, uniref90_database_path=a.uniref90,
                    mgnify_database_path=a.mgnify, uniprot_database_path=a.uniprot, jackhmmer_binary_path=a.jackhmmer,
                    hhblits_binary_path=a.hhblits, input_fasta_path=a.input_fasta, out_dir=a.features_dir,
                    n_cpus=a.n_cpus, n_workers=a.n_workers)
