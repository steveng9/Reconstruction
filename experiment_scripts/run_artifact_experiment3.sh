#!/bin/bash
# Artifact Experiment 3: reduced-scale attack x SDG grid on Adult.
#
# Reproduces a defined corner of Table 1 from raw data -- the MST and AIM
# columns, attacked by five of the paper's attacks -- to show the numbers in
# results.db come out of this code rather than merely being stored in it.
#
# Everything it needs is selected here rather than by editing source files, and
# every generator and attack used runs in the light Docker image (no GPU, no R,
# no Gurobi).
#
#     bash experiment_scripts/run_artifact_experiment3.sh
#
# Prerequisite (downloads Adult and carves the training samples):
#     python experiment_scripts/fetch_dataset.py adult
#
# Override the parallelism with WORKERS=<n> if you have fewer cores.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" || {
  echo "error: cannot resolve script directory" >&2; exit 1; }
while [ ! -f "$REPO_ROOT/paths.py" ] && [ "$REPO_ROOT" != "/" ]; do
  REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[ -f "$REPO_ROOT/paths.py" ] || {
  echo "error: cannot locate repository root (no paths.py found above $0)" >&2; exit 1; }
cd "$REPO_ROOT"

# Activate the conda env on the authors' machine; inside the Docker image the
# environment is already on PATH and there is no conda.
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate recon_
fi

# Write results to a separate database. Reviewers re-run test.sh after this,
# and that compares tables regenerated from the shipped results.db against the
# committed copies -- appending fresh runs to it would move those numbers.
export RECON_RESULTS_DB="${RECON_RESULTS_DB:-$REPO_ROOT/experiment_scripts/results_reproduction.db}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export WANDB_MODE=offline

DATASET=adult
SIZE=10000
SAMPLES="${SAMPLES:-2}"
WORKERS="${WORKERS:-4}"

# Four generator settings that are real Table 1 columns and run on CPU.
GENERATORS="${GENERATORS:-MST:1,MST:10,MST:100,AIM:1}"
# Five attacks that are real Table 1 rows: a baseline, three classical, one novel.
ATTACKS="${ATTACKS:-Mode,KNN,RandomForest,NaiveBayes,CoBP-RA}"

DATA_DIR="$(python -c 'from paths import DATA_ROOT; print(DATA_ROOT)')/$DATASET/size_$SIZE"
if [ ! -d "$DATA_DIR" ]; then
  echo "error: $DATA_DIR does not exist." >&2
  echo "Run this first:  python experiment_scripts/fetch_dataset.py $DATASET" >&2
  exit 1
fi

echo "=== STEP 1: generate synthetic data ($GENERATORS) ==="
SDG_DATASET="$DATASET" \
SDG_SAMPLE_SIZE="$SIZE" \
SDG_NUM_SAMPLES="$SAMPLES" \
SDG_JOBS="$GENERATORS" \
SDG_BIN_CONTINUOUS=1 \
  python sdg/generate_synth.py sdg || exit 1

echo ""
echo "=== STEP 2: run the attacks ($ATTACKS) ==="
SWEEP_DATASET="$DATASET" \
SWEEP_DATASET_SIZE="$SIZE" \
SWEEP_N_SAMPLES="$SAMPLES" \
SWEEP_SDG_METHODS="$GENERATORS" \
SWEEP_ATTACKS="$ATTACKS" \
  python experiment_scripts/run_production_sweep.py --workers "$WORKERS" || exit 1

echo ""
echo "DONE. Compare the printed per-attack means against the MST and AIM columns"
echo "of Table 1. Two samples are averaged here against the paper's five, so"
echo "expect individual cells to differ by a few points."
