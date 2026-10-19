#!/usr/bin/env bash
# Start scripts/train/launch.sh on several nodes (over ssh; the repository, data and logs/ are on a shared file system, the Python environment is
# installed on every node). NODES lists the nodes, "local" for the machine this runs on, otherwise [user@]host[:port]:
#
#   # one independent single-GPU run per node (different seeds): logs/train/standard_node0, standard_node1
#   NODES="local root@10.15.171.204:30526" RUN_NAME=standard sh scripts/train/launch_nodes.sh
#   # one DDP run over all nodes (GPUS_PER_NODE each); the first node is the master
#   MODE=ddp NODES="root@10.0.0.1 root@10.0.0.2" MASTER_ADDR=10.0.0.1 RUN_NAME=standard sh scripts/train/launch_nodes.sh
#   sh scripts/train/launch_nodes.sh trainer.max_steps=2000          # extra arguments are Hydra overrides
# The runs go to the background (nohup) on every node and keep running when this script ends; follow them with
#   tail -f logs/train/<RUN_NAME>*/train_node*.log          (stop: ssh <node> pkill -f scripts/train/train.py, and the torchrun above it)
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
NODES=${NODES:?NODES="local host[:port] ..." is required}
MODE=${MODE:-independent}                    # independent: one run per node | ddp: one run over all nodes
RUN_NAME=${RUN_NAME:-standard}
SEED=${SEED:-42}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
MASTER_PORT=${MASTER_PORT:-29500}
EXTRA=""; [ $# -gt 0 ] && EXTRA=$(printf ' %q' "$@")           # the Hydra overrides, quoted for the remote shell
n=$(echo $NODES | wc -w)
i=0
for node in $NODES; do
  if [ "$MODE" = ddp ]; then
    ENVS="NNODES=$n NODE_RANK=$i MASTER_ADDR=${MASTER_ADDR:?MASTER_ADDR is required with MODE=ddp} MASTER_PORT=$MASTER_PORT RUN_NAME=$RUN_NAME SEED=$SEED"
    LOG=$ROOT/logs/train/$RUN_NAME
  else
    ENVS="NNODES=1 NODE_RANK=0 RUN_NAME=${RUN_NAME}_node$i SEED=$((SEED + i))"
    LOG=$ROOT/logs/train/${RUN_NAME}_node$i
  fi
  ENVS="$ENVS GPUS_PER_NODE=$GPUS_PER_NODE MAX_STEPS=${MAX_STEPS:-100000} NUM_WORKERS=${NUM_WORKERS:-32} BACKEND=${BACKEND:-torch} COMPILE=${COMPILE:-true}"
  CMD="mkdir -p $LOG && cd $ROOT && $ENVS setsid nohup bash scripts/train/launch.sh$EXTRA > $LOG/launch_node$i.out 2>&1 < /dev/null &"
  if [ "$node" = local ]; then
    bash -c "$CMD" > /dev/null 2>&1 < /dev/null
  else
    host=${node%%:*}; port=22; [ "$host" != "$node" ] && port=${node##*:}
    ssh -n -p "$port" -o ConnectTimeout=20 "$host" "$CMD" > /dev/null 2>&1       # -n and no output: the ssh session ends with the command, the run goes on
  fi
  echo "node $i ($node): started, log $LOG/train_node${i}.log"
  i=$((i + 1))
done
