#!/bin/bash
# run_mia_rebuttal_pipeline.sh
#
# POPETS rebuttal: address the critique that our MIA baselines (SynthDistance,
# NNDR) might just be too weak to detect a real epsilon-sensitivity, masking
# it as a "plateau". Extends Table `mia_comparison` (previously only MST
# eps=1/1000) to the same 4 mechanistically-distinct DP generators used in
# the epsilon-plateau sweep: PrivBayes, PrivSyn, MWEMPGM, PrivateGSD, each at
# eps=1 and eps=1000, across adult 10k / cdc_diabetes 1k / nist_arizona 10k.
#
# This is the FAST leg only (PrivBayes/PrivSyn/MWEMPGM generation, all
# PGM/Bayes-net based, minutes per job) + the full MIA sweep. PrivateGSD on
# nist_arizona is handled by the separate run_mia_rebuttal_privategsd.sh,
# since it's jax/genetic-algorithm based and untested at this dataset's size
# (10k rows x 25 features) -- could take a long time, and shouldn't block
# getting the other 3 generators' results back quickly. Once that separate
# job finishes, rerun run_mia_rebuttal_sweep.py alone to fill in the
# nist_arizona/PrivateGSD cells (it's idempotent: skips nothing, just
# recomputes everything from whatever synth.csv files exist at the time).
#
# adult 10k already has all 4 generators at both epsilons in sample_00,
# EXCEPT PrivateGSD eps=1000, which is still being produced by the
# already-running, separately-launched private_gsd_10k_pipeline.sh (detached,
# started earlier today, cores 24-35). This script does NOT touch that job.
# run_mia_rebuttal_sweep.py checks for the synth.csv's existence and skips
# gracefully (with a clear log line) if it's not ready yet.
#
# Cores 0-23 are free (the earlier cdc_dp_sweep_pipeline.sh finished and
# released them). This pipeline uses 0-11. Cores 24-35 are in use by the
# still-running private_gsd_10k_pipeline.sh. Cores 36-47 stay untouched for
# other users of the shared machine it was developed on.

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

cd "$REPO_ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate recon_

export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

echo "=== STEP 1: generate missing MWEMPGM synth for cdc_diabetes 1k (eps=1,1000, sample_01) ==="
taskset -c 0-11 python experiment_scripts/generate_new_dp_sweep.py \
  --data-root ${RECON_DATA_ROOT}/cdc_diabetes/size_1000 \
  --meta-path ${RECON_DATA_ROOT}/cdc_diabetes/meta.json \
  --methods MWEMPGM \
  --epsilons 1 1000 \
  --samples 1 \
  --workers 2

echo ""
echo "=== STEP 2: generate missing PrivBayes/PrivSyn/MWEMPGM synth for nist_arizona 10k (eps=1,1000, sample_01) ==="
taskset -c 0-11 python experiment_scripts/generate_new_dp_sweep.py \
  --data-root ${RECON_DATA_ROOT}/nist_arizona_data/size_10000_25feat \
  --meta-path ${RECON_DATA_ROOT}/nist_arizona_data/meta.json \
  --methods PrivBayes PrivSyn MWEMPGM \
  --epsilons 1 1000 \
  --samples 1 \
  --workers 6

echo ""
echo "=== STEP 3: run MIA rebuttal sweep (SynthDistance, NNDR, RA-as-MIA) ==="
echo "  (nist_arizona/PrivateGSD and adult/PrivateGSD_eps1000 will show as SKIP -- expected, filled in later)"
taskset -c 0-11 python experiment_scripts/run_mia_rebuttal_sweep.py

echo ""
echo "PIPELINE DONE: MIA rebuttal sweep, fast leg (PrivBayes/PrivSyn/MWEMPGM x eps{1,1000} x adult/cdc/nist_arizona)"
