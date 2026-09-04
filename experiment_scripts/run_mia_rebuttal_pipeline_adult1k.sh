#!/bin/bash
# run_mia_rebuttal_pipeline_adult1k.sh
#
# Follow-up to run_mia_rebuttal_pipeline.sh: user's hypothesis is that these
# generators DO leak membership signal, and our MIA baselines just aren't
# sensitive enough to see it at n=10k/1k -- signal should be easier to detect
# at smaller n (less averaging-out, more overfitting per record). Reruns the
# same 3 attacks (SynthDistance, NNDR, RA-as-MIA) on adult at size_1000,
# across a wider generator set: the 5 DP generators (PrivBayes, PrivSyn,
# MWEMPGM, MST, PrivateGSD) at eps=1,1000, plus non-DP baselines known to
# leak (CellSuppression, RankSwap), Synthpop, and TabDDPM.
#
# adult/size_1000/sample_00 already has MST, PrivateGSD (full 9-eps sweeps),
# CellSuppression, RankSwap, Synthpop, TabDDPM -- only PrivBayes/PrivSyn/
# MWEMPGM at eps=1,1000 need generating (6 jobs, small dataset so fast).
#
# Cores 0-11 are free (the earlier fast MIA-rebuttal pipeline finished and
# released them). Cores 12-19 and 24-35 are in use by the still-running
# PrivateGSD generation jobs (nist_arizona and adult-10k respectively).
# Cores 36-47 stay untouched for other users.

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

echo "=== STEP 1: generate missing PrivBayes/PrivSyn/MWEMPGM synth for adult 1k (eps=1,1000, sample_00) ==="
taskset -c 0-11 python experiment_scripts/generate_new_dp_sweep.py \
  --data-root ${RECON_DATA_ROOT}/adult/size_1000 \
  --meta-path ${RECON_DATA_ROOT}/adult/meta.json \
  --methods PrivBayes PrivSyn MWEMPGM \
  --epsilons 1 1000 \
  --samples 0 \
  --workers 4

echo ""
echo "=== STEP 2: run MIA rebuttal sweep on adult 1k (SynthDistance, NNDR, RA-as-MIA) ==="
taskset -c 0-11 python experiment_scripts/run_mia_rebuttal_sweep_adult1k.py

echo ""
echo "PIPELINE DONE: MIA rebuttal sweep, adult 1k (PrivBayes/PrivSyn/MWEMPGM/MST/PrivateGSD x eps{1,1000} + CellSuppression/RankSwap/Synthpop/TabDDPM)"
