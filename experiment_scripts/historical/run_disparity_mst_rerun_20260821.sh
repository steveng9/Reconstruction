#!/bin/bash
# Disparate-impact re-run for the MST rows only (2026-08-21).
#
# Why: per_attack_disparity.csv is dated 2026-05-30, i.e. pre-repair. On the
# table's setting (adult 10k, QI1, sample_01) the repaired release dirs include
# MST_eps1 and MST_eps1000 -- both of which are rows in Table
# `disparate_impact`. AIM_eps1 was NOT repaired in that sample (only AIM_eps0.3
# and AIM_eps3 were), and no non-DP generator was affected, so MST is the whole
# stale set. Pre-repair outputs kept as per_attack_disparity*.prerepair.bak.csv.
#
# run_per_attack_disparity.py takes a single --sdg, hence two invocations.
# Cores 12-23 (the MIA re-run holds 0-11; 36-47 stay free for other users).

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

for SDG in MST_eps1 MST_eps1000; do
  echo "=== disparity re-run: $SDG  $(date) ==="
  taskset -c 12-23 python -u experiment_scripts/run_per_attack_disparity.py \
    --sdg "$SDG" --out "experiment_scripts/per_attack_disparity_rerun_${SDG}.csv"
done
echo "=== DONE $(date) ==="
