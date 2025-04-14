BASE=$(dirname $0)


# Generate systems pkl.gz
python $BASE/prepare_system.py \
  --receptor_pdb_path $BASE/demo/system_preparation/receptor.pdb \
  --ligand_sdf_path $BASE/demo/system_preparation/EJQ.sdf \
  --ligand_ccd_id EJQ \
  --systems_dir $BASE/demo/system_preparation/systems

#
# Get MSA features
python $BASE/run_homo_search.py \
  --input_fasta_path $BASE/demo/system_preparation/systems/fastas \
  --features_dir $BASE/demo/system_preparation/features \
  --bfd_database_path /2022133193/data/libs/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
  --uniclust30_database_path /2022133193/data/libs/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
  --uniref90_database_path /2022133193/data/libs/uniref90.fasta \
  --mgnify_database_path /2022133193/data/libs/mgy_clusters.fa \
  --uniprot_database_path /2022133193/data/libs/uniprot.fasta \
  --jackhmmer_binary_path /usr/bin/jackhmmer \
  --hhblits_binary_path /usr/bin/hhblits

# Docking
python $BASE/redocking.py \
  -i $BASE/demo/system_preparation/systems \
  -f $BASE/demo/system_preparation/features \
  --max_samples 40 \
  --max_rounds 5 \
  --num_samples_per_round 20 \
  --crop_size 256 \
  --atom_crop_size 2048 \
  --enable_physics_correction \
  --use_pocket \
  --use_key_res \
  --enable_ranking \
#  --enable_sidechain_relaxation
