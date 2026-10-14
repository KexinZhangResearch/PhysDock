#!/usr/bin/env bash
# Demo: build a system pickle (+ FASTA files for the MSA search) from the demo receptor PDB and ligand SDF.
#   sh scripts/prepare/prepare_system.sh
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${PYTHON:-python}           # e.g. PYTHON=/opt/conda/bin/python sh ...

$PYTHON $ROOT/scripts/prepare/prepare_system.py \
  --receptor_pdb $ROOT/data/demo/system_preparation/receptor.pdb \
  --ligand_sdf $ROOT/data/demo/system_preparation/EJQ.sdf \
  --ligand_ccd_id EJQ \
  --systems_dir $ROOT/data/demo/system_preparation/systems \
  "$@"
