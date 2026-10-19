#!/usr/bin/env bash
# One line per run in logs/train: step, speed, loss, checkpoint, last restart. Usage: sh scripts/train/status.sh [RUN_NAME_PREFIX]
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
for d in "$ROOT"/logs/train/${1:-}*/; do
  log=$(ls "$d"train_node*.log 2>/dev/null | head -1); [ -n "$log" ] || continue
  last=$(tr '\r' '\n' < "$log" | grep -E "^(Epoch|Training)" | tail -1 | sed -E 's/.*: +[0-9]+%\|[^|]*\| ([0-9]+\/[0-9?]+) \[([^],]*), *([0-9.]+it\/s).*train\/loss=([0-9.]+).*/step \1 elapsed \2 \3 loss \4/')
  ck=$(ls "$d"checkpoints 2>/dev/null | tr '\n' ' ')
  rs=$(grep -c "resuming" "$log")
  sk=$(grep -c "update skipped" "$log")
  alive=$(pgrep -f "hydra.run.dir=${d%/}" > /dev/null && echo running || echo STOPPED)
  echo "$(basename "$d"): $alive | $last | ckpt: ${ck:-none} | restarts: $((rs > 0 ? rs - 1 : 0)) | skipped updates: $sk"
done
