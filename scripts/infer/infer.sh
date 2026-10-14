#!/usr/bin/env bash
# Demo: redock one PoseBusters system with physics correction (config: configs/infer.yaml).
#   sh scripts/infer/infer.sh                 # -> out/5SAK_ZRY
#   sh scripts/infer/infer.sh seed=0          # extra arguments are Hydra overrides
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${PYTHON:-python}           # e.g. PYTHON=/opt/conda/bin/python sh ...

$PYTHON $ROOT/scripts/infer/infer.py \
  input_pkl=$ROOT/data/demo/redocking/Posebusters_subset/5SAK_ZRY_A_1.pkl.gz \
  msa_features=$ROOT/data/demo/redocking/features/msa_features \
  uniprot_msa_features=$ROOT/data/demo/redocking/features/uniprot_msa_features \
  output_dir=$ROOT/out/5SAK_ZRY \
  sampling.max_samples=40 \
  sampling.max_rounds=5 \
  sampling.num_samples_per_round=20 \
  sampling.physics_correction=true \
  features.crop_size=256 \
  features.atom_crop_size=2048 \
  features.use_pocket=true \
  features.use_key_res=true \
  "$@"
