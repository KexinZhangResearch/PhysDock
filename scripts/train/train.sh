#!/usr/bin/env bash
# Demo: quick training test with the toy model (200 steps, wandb offline; config: configs/train.yaml).
# Needs the training set: python scripts/download/download_dataset.py
#   sh scripts/train/train.sh
#   sh scripts/train/train.sh compile=true                       # extra arguments are Hydra overrides
# Full training of the released model size on all GPUs of the node:
#   python scripts/train/train.py trainer=ddp
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${PYTHON:-python}           # e.g. PYTHON=/opt/conda/bin/python sh ...

$PYTHON $ROOT/scripts/train/train.py \
  model=toy \
  data.samples_per_epoch=100 \
  trainer.max_steps=200 \
  trainer.val_check_interval=100 \
  data.num_val_systems=4 \
  logger.offline=true \
  "$@"
