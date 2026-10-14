#!/usr/bin/env bash
# Demo: full docking pipeline from a receptor PDB and a ligand SDF:
# system preparation -> MSA search (needs the sequence databases, see prepare/homo_search.sh) -> redocking.
#   sh scripts/infer/docking.sh
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${PYTHON:-python}           # e.g. PYTHON=/opt/conda/bin/python sh ...

sh $ROOT/scripts/prepare/prepare_system.sh
sh $ROOT/scripts/prepare/homo_search.sh

$PYTHON $ROOT/scripts/infer/redocking.py \
  input_dir=$ROOT/data/demo/system_preparation/systems \
  features_dir=$ROOT/data/demo/system_preparation/features \
  sampling.max_samples=40 \
  sampling.max_rounds=5 \
  sampling.num_samples_per_round=20 \
  sampling.physics_correction=true \
  features.crop_size=256 \
  features.atom_crop_size=2048 \
  features.use_pocket=true \
  features.use_key_res=true \
  ranking=true \
  "$@"
#  sidechain_relaxation=true
