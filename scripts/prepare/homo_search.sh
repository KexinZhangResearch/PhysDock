#!/usr/bin/env bash
# Demo: MSA search for the demo system (after prepare_system.sh). Needs the sequence databases
# (python scripts/download/download_homo_data.py, or HOMO_DATA=<dir>) and jackhmmer / hhblits.
#   sh scripts/prepare/homo_search.sh
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${PYTHON:-python}           # e.g. PYTHON=/opt/conda/bin/python sh ...
DB=${HOMO_DATA:-$ROOT/data/homo_data}

$PYTHON $ROOT/scripts/prepare/homo_search.py \
  --input_fasta $ROOT/data/demo/system_preparation/systems/fastas \
  --features_dir $ROOT/data/demo/system_preparation/features \
  --bfd $DB/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
  --uniclust30 $DB/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
  --uniref90 $DB/uniref90.fasta \
  --mgnify $DB/mgy_clusters_2022_05.fa \
  --uniprot $DB/uniprot.fasta \
  --jackhmmer /usr/bin/jackhmmer \
  --hhblits /usr/bin/hhblits \
  "$@"
