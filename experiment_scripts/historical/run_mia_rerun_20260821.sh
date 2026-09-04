#!/bin/bash
# MIA sweep re-run on POST-encoding-repair synth (2026-08-21).
#
# Why: mia_rebuttal_sweep_results.csv is dated 2026-07-13/15, i.e. it was
# computed BEFORE the float bin-midpoint repair. The affected cells are
# adult 10k PrivBayes + PrivateGSD, cdc 1k PrivBayes, and nist_arizona
# PrivBayes + PrivateGSD; MST (added to this sweep now) was affected on
# adult 10k as well. PrivSyn and MWEMPGM were never float-encoded.
#
# Scope change vs the July run: MST is now swept here rather than in the older
# compare_mia_ra.py / run_mia_comparison_{cdc,arizona}.py scripts, and the four
# non-DP reference generators are recomputed too, so every column of Table
# `mia_comparison` comes from one code path, one train/holdout pairing, and
# post-repair synth. 42 cells = 3 datasets x (5 DP x 2 eps + 4 non-DP).
#
# Pre-repair outputs preserved as *.prerepair.bak.csv.
# Cores 0-11. 36-47 stay free for other users per this machine's convention.

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

echo "=== MIA rebuttal sweep re-run (post-repair) started $(date) ==="
taskset -c 0-11 python -u experiment_scripts/run_mia_rebuttal_sweep.py
echo "=== DONE $(date) rc=$? ==="
