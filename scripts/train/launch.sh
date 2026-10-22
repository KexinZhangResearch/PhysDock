#!/usr/bin/env bash
# Training of the standard PhysDock (configs/model/medium.yaml: Pairformer 24, atom DiTs 3 + 3, token DiT 12; crop 256 / 2048 atoms, bf16-mixed,
# torch.compile, no activation checkpointing, Triton kernels; BACKEND=torch for the plain torch ops) on one node, or on several nodes with DDP. Run it on
# EVERY node. The optimiser (fused AdamW with AlphaFold 3's betas, eps and weight decay), the learning rate (AlphaFold 3's 1.8e-3 at a batch of 256,
# scaled to the batch of this run by the square-root rule, see configs/trainer/default.yaml), the schedule and the EMA are in configs/trainer/default.yaml;
# a model trained from scratch conditions on the noise level as ln(t) / 4 (the default "legacy" form is only for the released weights), so inference
# of such a checkpoint needs model.diffusion.noise_conditioning=af3.
#
#   # one node, one GPU (or all of them: GPUS_PER_NODE=8)
#   RUN_NAME=standard sh scripts/train/launch.sh
#   # two nodes, one DDP run (NODE_RANK=0 on the node MASTER_ADDR points to)
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 RUN_NAME=standard sh scripts/train/launch.sh
#   NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 RUN_NAME=standard sh scripts/train/launch.sh
#   # extra arguments are Hydra overrides of configs/train.yaml:  sh scripts/train/launch.sh trainer.max_steps=2000 model=toy
# (scripts/train/launch_nodes.sh starts this on several machines over ssh.)
#
# Everything lives in logs/train/$RUN_NAME: checkpoints/ (a checkpoint every 1000 steps, only the newest is kept), wandb/, train_node<rank>.log.
# torchrun restarts the workers after a crash (up to MAX_RESTARTS times) and the restarted run resumes from the newest complete checkpoint
# (ckpt_path=auto), also the same wandb run. Started again with the same RUN_NAME the run continues where it stopped.
# wandb: online with an API key (wandb login, or WANDB_API_KEY=...), otherwise local files that `sh scripts/train/wandb_sync.sh` uploads later.
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
for p in "${PYTHON:-}" /opt/conda/envs/PhysDock/bin/python /opt/conda/envs/PhysDock-cu124/bin/python python; do
  [ -n "$p" ] && command -v "$p" >/dev/null 2>&1 && PYTHON=$p && break
done
RUN_NAME=${RUN_NAME:-standard}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
MAX_STEPS=${MAX_STEPS:-100000}
BACKEND=${BACKEND:-triton}                    # triton | torch
PRECISION=${PRECISION:-bf16-mixed}            # bf16-mixed | 32 (float32 with TF32 matmuls)
CKPT=${CKPT:-none}                            # activation checkpointing of every stack: none | default (the setting of configs/train.yaml: same values, less memory)
CKPT_ARGS="checkpointing.AtomTransformer=none checkpointing.Triangleformer=none checkpointing.Evoformer=none checkpointing.Pairformer=none checkpointing.DiT=none"
[ "$CKPT" = default ] && CKPT_ARGS=""
COMPILE=${COMPILE:-true}
SEED=${SEED:-42}
NUM_WORKERS=${NUM_WORKERS:-32}                # featurisation workers per GPU
MAX_RESTARTS=${MAX_RESTARTS:-30}
RUN_DIR=$ROOT/logs/train/$RUN_NAME
mkdir -p "$RUN_DIR"

export PHYSDOCK_ROOT=$ROOT
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export NCCL_ASYNC_ERROR_HANDLING=1
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT:-300}
export PYTHONUNBUFFERED=1

WORLD=$((NNODES * GPUS_PER_NODE))
STRATEGY=auto; [ "$WORLD" -gt 1 ] && STRATEGY=ddp
if [ "$NNODES" -gt 1 ]; then
  RDZV="--rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT --rdzv_id=$RUN_NAME"
else
  RDZV="--standalone"
fi
echo "[launch] $(date '+%F %T') run $RUN_NAME node $NODE_RANK/$NNODES, $GPUS_PER_NODE GPU(s), seed $SEED, $MAX_STEPS steps, backend $BACKEND, compile $COMPILE, python $PYTHON, output $RUN_DIR"

exec "$PYTHON" -m torch.distributed.run $RDZV --nnodes="$NNODES" --nproc_per_node="$GPUS_PER_NODE" --node_rank="$NODE_RANK" \
  --max-restarts="$MAX_RESTARTS" --monitor-interval=10 \
  "$ROOT/scripts/train/train.py" \
  hydra.run.dir="$RUN_DIR" \
  trainer.num_nodes="$NNODES" trainer.devices="$GPUS_PER_NODE" trainer.strategy=$STRATEGY \
  trainer.max_steps="$MAX_STEPS" trainer.precision=$PRECISION compile="$COMPILE" model.backend="$BACKEND" seed="$SEED" \
  model.diffusion.noise_conditioning=af3 \
  $CKPT_ARGS \
  data.num_workers="$NUM_WORKERS" grad_stats=module logger.name="$RUN_NAME" \
  "$@" 2>&1 | tee -a "$RUN_DIR/train_node$NODE_RANK.log"
