BASE=$(dirname $0)

python $BASE/screening.py \
  -i $BASE/demo/screening/6kzd.pkl.gz \
  -f $BASE/demo/screening/features \
  -s $BASE/demo/screening/demo_db.txt \
  --max_samples 40 \
  --max_rounds 5 \
  --num_samples_per_round 20 \
  --crop_size 256 \
  --atom_crop_size 2048 \
  --use_pocket \
  --use_key_res \
  --enable_ranking
