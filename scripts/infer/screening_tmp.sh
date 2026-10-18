#!/usr/bin/env bash
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${PYTHON:-python}           # e.g. PYTHON=/opt/conda/bin/python sh ...

$PYTHON $ROOT/scripts/infer/screening.py \
  input_pkl=/home/zhangkexin/Downloads/test_cache/3lck_noshift/systems/Apo_3LCK_A_1.pkl.gz \
  features_dir=/home/zhangkexin/Downloads/test_cache/features \
  smiles=/home/zhangkexin/Downloads/test_cache/LHL.txt \
  sampling.max_samples=10 \
  sampling.max_rounds=5 \
  sampling.num_samples_per_round=20 \
  ranking=true \
  sampling.physics_correction=true \
  output_dir=./no_pocket_no_shift \
  "$@"
