BASE=$(dirname $0)

python $BASE/redocking.py \
  -i $BASE/demo/redocking/Posebusters_subset \
  -f $BASE/demo/redocking/features \
  --max_samples 40 \
  --max_rounds 5 \
  --num_samples_per_round 20 \
  --crop_size 256 \
  --atom_crop_size 2048 \
  --enable_physics_correction \
  --use_pocket \
  --use_key_res \
  --enable_ranking
