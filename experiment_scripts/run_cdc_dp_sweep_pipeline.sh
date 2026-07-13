#!/bin/bash
# run_cdc_dp_sweep_pipeline.sh
#
# Cross-dataset validation of the epsilon-plateau finding: same 9-point
# epsilon sweep (0.1 ... 1000), same 3 attacks (RandomForest, NaiveBayes,
# CoBP-RA), same 3 DP generators (MST, PrivBayes, PrivSyn) as the adult 10k
# sweep, but run on cdc_diabetes size_1000, 5 trials (samples 0-4), 2 QI
# variants (QI1, QI_large -- mirrors the QI1/QI_large variants used
# throughout the adult sweep).
cd /home/golobs/Reconstruction
source /home/golobs/miniconda3/etc/profile.d/conda.sh
conda activate recon_

# Cap per-process thread usage so this doesn't monopolize all 48 cores and
# starve other users (daniilf, sikha) sharing the machine.
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

echo "=== STEP 1: generate MST/PrivBayes/PrivSyn synth, samples 0-4 x 9 epsilons @ cdc_diabetes size_1000 ==="
python experiment_scripts/generate_new_dp_sweep.py \
  --data-root /home/golobs/data/reconstruction_data/cdc_diabetes/size_1000 \
  --meta-path /home/golobs/data/reconstruction_data/cdc_diabetes/meta.json \
  --methods MST PrivBayes PrivSyn \
  --samples 0 1 2 3 4 \
  --workers 8

echo ""
echo "=== STEP 2: attack sweep - 3 attacks x 2 QIs x 5 samples x 9 epsilons x 3 generators, cdc_diabetes size_1000 ==="
SWEEP_DATASET_NAME=cdc_diabetes \
SWEEP_DATASET_SIZE=1000 \
SWEEP_DATA_ROOT=/home/golobs/data/reconstruction_data/cdc_diabetes/size_1000 \
SWEEP_N_SAMPLES=5 \
SWEEP_QI_VARIANTS="QI1,QI_large" \
SWEEP_WANDB_GROUP="new-dp-epsilon-sweep-cdc-1k" \
python experiment_scripts/run_new_dp_epsilon_sweep.py \
  --sdg-methods MST PrivBayes PrivSyn \
  --samples 0 1 2 3 4 \
  --workers 8

echo ""
echo "=== STEP 3: insert results into results.db ==="
LATEST_CSV=$(ls -t experiment_scripts/new_dp_epsilon_sweep_cdc_diabetes_sz1000_*.csv | head -1)
echo "Inserting $LATEST_CSV"
python experiment_scripts/insert_new_dp_sweep_to_db.py "$LATEST_CSV" \
  --dataset cdc_diabetes \
  --dataset-size 1000 \
  --wandb-group "new-dp-epsilon-sweep-cdc-1k"

echo ""
echo "PIPELINE DONE: cdc_diabetes DP sweep (MST/PrivBayes/PrivSyn, 5 trials x 9 eps x 2 QIs x 3 attacks)"
