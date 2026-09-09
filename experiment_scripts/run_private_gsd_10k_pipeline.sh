#!/bin/bash
# run_private_gsd_10k_pipeline.sh
#
# Extends PrivateGSD to adult size_10000 (1 trial, sample_00 only) across the
# full 9-point epsilon sweep, then attacks with the same 3 attacks x 3 QIs
# used everywhere else in this rebuttal sweep.
#
# CAUTION: earlier testing found PrivateGSD's genetic-algorithm synthesis does
# not scale cleanly from n=1000 to n=10000 on CPU-only jax (a prior attempt
# saw 32+ min with no progress before being killed). This run is not wrapped
# in an artificial timeout at the per-job level, but the whole generation step
# is capped at 6h; if it's still stuck by then, the script moves on to attacks
# using whatever synth.csv files did complete (missing ones just fail as
# individual job errors in step 2, not fatal to the rest of the sweep).

# Resolve the repository root (the directory containing paths.py) without
# hard-coding an absolute path, and default the data root beneath it.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" || {
  echo "error: cannot resolve script directory" >&2; exit 1; }
while [ ! -f "$REPO_ROOT/paths.py" ] && [ "$REPO_ROOT" != "/" ]; do
  REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[ -f "$REPO_ROOT/paths.py" ] || {
  echo "error: cannot locate repository root (no paths.py found above $0)" >&2; exit 1; }
: "${RECON_DATA_ROOT:=$REPO_ROOT/data}"

# Write results to a separate database, so a re-run cannot move the numbers
# that the shipped results.db backs (Experiment 1 and test.sh compare
# against tables regenerated from it).
export RECON_RESULTS_DB="${RECON_RESULTS_DB:-$REPO_ROOT/experiment_scripts/results_reproduction.db}"

cd "$REPO_ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate recon_

# Cap per-process thread usage so this doesn't monopolize all 48 cores and
# starve other users of the shared machine it was developed on.
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

echo "=== STEP 1: generate PrivateGSD synth, sample_00 x 9 epsilons @ size_10000 ==="
# CPU-affinity pinned (cores 24-35) -- OMP/MKL/OPENBLAS env caps alone do NOT
# bound jax's XLA thread pool (defaults to os.cpu_count()=48 regardless).
# taskset hard-caps real core usage instead. NOTE: this only bounds *which
# cores* a process may run on -- it does NOT bound how many processes get
# spawned. That distinction mattered a lot for PrivBayes (see cdc pipeline
# comment below); PrivateGSD/jax does not spawn its own subprocess pools, so
# taskset alone is sufficient here.
timeout 6h taskset -c 24-35 python experiment_scripts/generate_new_dp_sweep.py \
  --data-root ${RECON_DATA_ROOT}/adult/size_10000 \
  --meta-path ${RECON_DATA_ROOT}/adult/meta.json \
  --methods PrivateGSD \
  --samples 0 \
  --workers 4 || echo "  (generation step exited non-zero / timed out -- continuing to attack step with whatever synth.csv files exist)"

echo ""
echo "=== STEP 2: attack sweep - 3 attacks x 3 QIs x 1 sample x 9 epsilons, PrivateGSD @ size_10000 ==="
SWEEP_DATASET_NAME=adult \
SWEEP_DATASET_SIZE=10000 \
SWEEP_DATA_ROOT=${RECON_DATA_ROOT}/adult/size_10000 \
SWEEP_N_SAMPLES=5 \
SWEEP_QI_VARIANTS="QI_large,QI_behavioral,QI1" \
SWEEP_WANDB_GROUP="new-dp-epsilon-sweep-adult-10k-privategsd" \
taskset -c 24-35 python experiment_scripts/run_new_dp_epsilon_sweep.py \
  --sdg-methods PrivateGSD \
  --samples 0 \
  --workers 4

echo ""
echo "=== STEP 3: insert results into results.db ==="
LATEST_CSV=$(ls -t experiment_scripts/new_dp_epsilon_sweep_adult_sz10000_*.csv | head -1)
echo "Inserting $LATEST_CSV"
python experiment_scripts/insert_new_dp_sweep_to_db.py "$LATEST_CSV" \
  --dataset adult \
  --dataset-size 10000 \
  --wandb-group "new-dp-epsilon-sweep-adult-10k-privategsd"

echo ""
echo "PIPELINE DONE: PrivateGSD adult 10k sweep (1 trial x 9 eps x 3 QIs x 3 attacks)"
