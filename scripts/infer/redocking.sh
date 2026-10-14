#!/usr/bin/env bash
# Demo: redock the PoseBusters demo systems (config: configs/redocking.yaml).
# ranking=true needs scikit-learn; without it pass ranking=false.
#   sh scripts/infer/redocking.sh             # -> data/demo/redocking/Posebusters_subset_predictions
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${PYTHON:-python}           # e.g. PYTHON=/opt/conda/bin/python sh ...

$PYTHON $ROOT/scripts/infer/redocking.py \
  input_dir=$ROOT/data/demo/redocking/Posebusters_subset \
  features_dir=$ROOT/data/demo/redocking/features \
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
