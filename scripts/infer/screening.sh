#!/usr/bin/env bash
# Demo: screen the demo SMILES against 6kzd (config: configs/screening.yaml).
# ranking=true needs scikit-learn; without it pass ranking=false.
#   sh scripts/infer/screening.sh             # -> data/demo/screening/6kzd_screening
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${PYTHON:-python}           # e.g. PYTHON=/opt/conda/bin/python sh ...

$PYTHON $ROOT/scripts/infer/screening.py \
  input_pkl=$ROOT/data/demo/screening/6kzd.pkl.gz \
  features_dir=$ROOT/data/demo/screening/features \
  smiles=$ROOT/data/demo/screening/demo_db.txt \
  sampling.max_samples=40 \
  sampling.max_rounds=5 \
  sampling.num_samples_per_round=20 \
  features.crop_size=256 \
  features.atom_crop_size=2048 \
  features.use_pocket=true \
  features.use_key_res=true \
  ranking=true \
  "$@"
