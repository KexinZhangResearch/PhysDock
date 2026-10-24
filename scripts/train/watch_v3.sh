#!/usr/bin/env bash
# Follow the training logs of several runs side by side, redrawn every INTERVAL seconds: one line per 100 steps (the last ROWS lines), a run that has not reached a step
# yet shows xxx (filled in once it gets there); the last line: the current step of every run. Columns per run: x_mse / slddt / gnorm.
#   sh scripts/train/watch_v3.sh                       (the four v3 runs)
#   sh scripts/train/watch_v3.sh run_a run_b ...       (any runs in logs/train)
#   INTERVAL=5 ROWS=40 sh scripts/train/watch_v3.sh
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUNS=${*:-physdock_base_v3_5k_notempl pomelo_v3_5k_notempl singletrack_v3_5k_notempl fold_v3_5k_notempl}
INTERVAL=${INTERVAL:-5}
ROWS=${ROWS:-40}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"; exit' INT TERM
while true; do
  max=0
  for r in $RUNS; do
    log="$ROOT/logs/train/$r/train.log"
    # step x_mse slddt gnorm, one line per logged step
    grep -ao 'step [0-9]* x_mse [0-9.]* slddt [0-9.]* gnorm(pre-clip) [^ ]*' "$log" 2>/dev/null | awk '{print $2, $4, $6, $8}' > "$tmp/$r"
    # the current step from the progress bar (e.g. "1234/10000")
    tail -c 4000 "$log" 2>/dev/null | tr '\r' '\n' | grep -ao '[0-9]*/[0-9]* \[' | tail -1 | cut -d/ -f1 > "$tmp/$r.now"
    m=$(tail -1 "$tmp/$r" | awk '{print $1}')
    [ -n "$m" ] && [ "$m" -gt "$max" ] && max=$m
  done
  {
    printf "%s   (every %ss, Ctrl+C to quit)\n" "$(date '+%H:%M:%S')" "$INTERVAL"
    printf "%-6s" step; for r in $RUNS; do printf " | %-26s" "${r%%_v3*}"; done; echo
    first=$((max - 100 * (ROWS - 1))); [ "$first" -lt 100 ] && first=100
    s=$first
    while [ "$s" -le "$max" ]; do
      printf "%-6s" "$s"
      for r in $RUNS; do
        v=$(awk -v s="$s" '$1 == s {printf "%s / %s / %s", $2, $3, $4}' "$tmp/$r")
        printf " | %-26s" "${v:-xxx}"
      done
      echo
      s=$((s + 100))
    done
    printf "%-6s" now; for r in $RUNS; do printf " | %-26s" "step $(cat "$tmp/$r.now" 2>/dev/null)"; done; echo
  } > "$tmp/screen"
  clear; cat "$tmp/screen"
  sleep "$INTERVAL"
done
