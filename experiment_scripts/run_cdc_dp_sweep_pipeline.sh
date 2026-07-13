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
# CPU-affinity pinned (cores 0-23). Two DISTINCT bugs were found and fixed
# this session, both mattered:
#  1) OMP/MKL/OPENBLAS env caps alone do NOT bound PyTorch/jax thread pools
#     (default to os.cpu_count()=48 regardless) -- taskset bounds real core
#     usage no matter what any library does internally re: thread count.
#  2) The actual root cause of the "dozens and dozens of processes" incident:
#     DataSynthesizer's PrivBayes.py (a PrivBayes dependency) does a bare
#     `with Pool() as pool:` INSIDE its greedy Bayesian-network loop (once per
#     remaining attribute) -- defaulting to os.cpu_count()=48 *worker
#     processes*, spawned repeatedly, inside whichever one of our own N
#     outer ProcessPoolExecutor workers happened to be running a PrivBayes
#     job. taskset does NOT bound process *count*, only which cores they run
#     on -- so this needed a real code fix, not just tighter pinning. Fixed
#     in sdg/privbayes_method.py by monkeypatching DataSynthesizer's Pool to
#     a fixed small size (PRIVBAYES_INNER_POOL_SIZE, default 2) before it's
#     ever called. With that fixed, taskset's core-range pinning is now
#     sufficient on its own to bound total resource usage.
taskset -c 0-23 python experiment_scripts/generate_new_dp_sweep.py \
  --data-root /home/golobs/data/reconstruction_data/cdc_diabetes/size_1000 \
  --meta-path /home/golobs/data/reconstruction_data/cdc_diabetes/meta.json \
  --methods MST PrivBayes PrivSyn \
  --samples 0 1 2 3 4 \
  --workers 12

echo ""
echo "=== STEP 2: attack sweep - 3 attacks x 2 QIs x 5 samples x 9 epsilons x 3 generators, cdc_diabetes size_1000 ==="
SWEEP_DATASET_NAME=cdc_diabetes \
SWEEP_DATASET_SIZE=1000 \
SWEEP_DATA_ROOT=/home/golobs/data/reconstruction_data/cdc_diabetes/size_1000 \
SWEEP_N_SAMPLES=5 \
SWEEP_QI_VARIANTS="QI1,QI_large" \
SWEEP_WANDB_GROUP="new-dp-epsilon-sweep-cdc-1k" \
taskset -c 0-23 python experiment_scripts/run_new_dp_epsilon_sweep.py \
  --sdg-methods MST PrivBayes PrivSyn \
  --samples 0 1 2 3 4 \
  --workers 12

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
