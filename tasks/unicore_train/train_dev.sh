[ -z "${MASTER_PORT}" ] && MASTER_PORT=23333
[ -z "${MASTER_IP}" ] && MASTER_IP=$(hostname -I)
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1
[ -z "${OMPI_COMM_WORLD_RANK}" ] && OMPI_COMM_WORLD_RANK=0
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${update_freq}" ] && update_freq=1
[ -z "${total_step}" ] && total_step=30720000
[ -z "${warmup_step}" ] && warmup_step=1000
[ -z "${decay_step}" ] && decay_step=1000
[ -z "${decay_ratio}" ] && decay_ratio=0.998
[ -z "${lr}" ] && lr=1.8e-3
[ -z "${seed}" ] && seed=42



#[ -z "${MASTER_PORT}" ] && MASTER_PORT=${MASTER_PORT}
#[ -z "${MASTER_IP}" ] && MASTER_IP=${MASTER_ADDR}
#[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=${WORLD_SIZE}
#[ -z "${OMPI_COMM_WORLD_RANK}" ] && OMPI_COMM_WORLD_RANK=${RANK}


# 1.8e-3 af3
# 5e-5 Diffusion Traning?
# 1e-6 stable diffusion/midjounary

cd $(dirname $0)
save_dir=output_dev

num_processors=$((${OMPI_COMM_WORLD_SIZE}*(${n_gpu})))

export NUMEXPR_MAX_THREADS=32


export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
echo "n_gpu per node" $n_gpu
echo "OMPI_COMM_WORLD_SIZE" $OMPI_COMM_WORLD_SIZE
echo "num processors" $num_processors
echo "OMPI_COMM_WORLD_RANK" $OMPI_COMM_WORLD_RANK
echo "MASTER_IP" $MASTER_IP
echo "MASTER_PORT" $MASTER_PORT
echo "save_dir" $save_dir
echo "decay_step" $decay_step
echo "warmup_step" $warmup_step
echo "decay_ratio" $decay_ratio
echo "lr" $lr
echo "total_step" $total_step
echo "update_freq" $update_freq
echo "seed" $seed
echo "data_folder:"
echo "create folder for save"
mkdir -p $save_dir
echo "start training"

tmp_dir=`mktemp -d`

#VP-SDE AF3
#VE-SDE AF3

# TODO: torch compile cache setting
# TORCHINDUCTOR_FX_GRAPH_CACHE=0 TORCHINDUCTOR_CACHE_DIR=../../../COMPILE_CACHE_AF3_FP16_NEW \
# TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
python -m torch.distributed.run \
    --nproc_per_node=$n_gpu --master_port $MASTER_PORT --nnodes=$OMPI_COMM_WORLD_SIZE --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=$MASTER_IP \
  $(which unicore-train) \
  --num-processors $num_processors \
  --user-dir ./ \
  --num-workers 32 \
  --data-buffer-size 4 \
  --task stfold \
  --loss stfoldloss \
  --arch stfold \
  --optimizer adam \
  --adam-betas '(0.9, 0.95)' \
  --adam-eps 1e-8 \
  --clip-norm 10.0 \
  --per-sample-clip-norm 0.1 \
  --allreduce-fp32-grad \
  --lr-scheduler exponential_decay \
  --lr $lr \
  --warmup-updates $warmup_step \
  --decay-ratio $decay_ratio \
  --decay-steps $decay_step \
  --stair-decay \
  --batch-size 1 \
  --update-freq $update_freq \
  --seed $seed \
  --tensorboard-logdir $save_dir/tsb/ \
  --max-update $total_step \
  --max-epoch 1 \
  --log-interval 1 \
  --log-format simple \
  --save-interval-updates 400 \
  --validate-interval-updates 500 \
  --keep-interval-updates 40 \
  --required-batch-size-multiple 1 \
  --disable-validation \
  --max-epoch 1000 \
  --save-dir $save_dir \
  --tmp-save-dir $save_dir/tmp \
  --save-interval 1 \
  --ddp-backend=no_c10d \
  --matmul-precision high \
  --model-name toy \
  --crop-size 256 \
  --max-recycling-iters 1\
  --max-msa-clusters 128 \
  --num-augmentation-sample 48 \
  --alpha-bond 1 \
  --alpha-pae 1 \
  --use-template \
  --mini-rollout-steps 12 \
  --ema-decay 0.999 \
  --atom-attention-type spatial \
  --use-bf16 \
  --interaction-aware \
  --templ-dim 40

#  --compile \
#  --use-mini-rollout \
#  --use-flash-attn \
#  --bf16 \
#  --bf16-sr \

# --reset-optimizer \
# --init-from-ckpt /2022133002/zkx/stfold/stfold_v7.1.4/tasks/ligand_af3_fp32/output/checkpoint_1_11500.pt \
