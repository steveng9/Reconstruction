#!/bin/bash
# run_private_gsd_full_pipeline.sh
#
# Brings PrivateGSD up to parity with PrivBayes/MWEMPGM/PrivSyn: full 9-point
# epsilon sweep x 4 trials (samples 0-3) x 3 QIs (QI1, QI_behavioral, QI_large)
# x 3 attacks (RandomForest, NaiveBayes, CoBP-RA), at adult size_1000 (GSD
# does not scale to size_10000 on CPU-only jax; sample_00 was already spot-
# checked there previously and does not scale to size_10000 on CPU-only jax).
#
# sample_00 x 9 epsilons already generated. This script generates samples 1-3,
# then runs the full attack grid (all 4 samples x 3 QIs x 3 attacks), then
# inserts into results.db.
set -e
cd /home/golobs/Reconstruction
source /home/golobs/miniconda3/etc/profile.d/conda.sh
conda activate recon_

echo "=== STEP 1: generate PrivateGSD synth, samples 1-3 x 9 epsilons @ size_1000 ==="
python experiment_scripts/generate_new_dp_sweep.py \
  --data-root /home/golobs/data/reconstruction_data/adult/size_1000 \
  --meta-path /home/golobs/data/reconstruction_data/adult/meta.json \
  --methods PrivateGSD \
  --samples 1 2 3 \
  --workers 4

echo ""
echo "=== STEP 2: attack sweep - 3 attacks x 3 QIs x 4 samples x 9 epsilons, PrivateGSD @ size_1000 ==="
SWEEP_DATASET_SIZE=1000 \
SWEEP_DATA_ROOT=/home/golobs/data/reconstruction_data/adult/size_1000 \
SWEEP_N_SAMPLES=4 \
SWEEP_QI_VARIANTS="QI_large,QI_behavioral,QI1" \
SWEEP_WANDB_GROUP="new-dp-epsilon-sweep-adult-1k-privategsd" \
python experiment_scripts/run_new_dp_epsilon_sweep.py \
  --sdg-methods PrivateGSD \
  --samples 0 1 2 3 \
  --workers 6

echo ""
echo "=== STEP 3: insert results into results.db ==="
LATEST_CSV=$(ls -t experiment_scripts/new_dp_epsilon_sweep_sz1000_*.csv | head -1)
echo "Inserting $LATEST_CSV"
python experiment_scripts/insert_new_dp_sweep_to_db.py "$LATEST_CSV" \
  --dataset-size 1000 \
  --wandb-group "new-dp-epsilon-sweep-adult-1k-privategsd"

echo ""
echo "PIPELINE DONE: PrivateGSD full 1k sweep (4 trials x 9 eps x 3 QIs x 3 attacks)"
