#!/usr/bin/env bash
# Upload the wandb files of runs that were logged offline (no API key at the time): wandb login (or WANDB_API_KEY=...) first.
#   sh scripts/train/wandb_sync.sh [logs/train/<run> ...]        (default: every run in logs/train)
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
for p in "${PYTHON:-}" /opt/conda/envs/PhysDock/bin/python /opt/conda/envs/PhysDock-cu124/bin/python python; do
  [ -n "$p" ] && command -v "$p" >/dev/null 2>&1 && PYTHON=$p && break
done
RUNS=("$@"); [ ${#RUNS[@]} -eq 0 ] && RUNS=("$ROOT"/logs/train/*)
for run in "${RUNS[@]}"; do
  [ -d "$run/wandb" ] || continue
  echo "== $run"
  ls -d "$run"/wandb/offline-run-* 2>/dev/null | sort | xargs -r "$PYTHON" -m wandb sync    # in the order of the restarts: one run id, resumed
done
