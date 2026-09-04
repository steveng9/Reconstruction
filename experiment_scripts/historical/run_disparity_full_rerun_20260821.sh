#!/bin/bash
# FULL disparate-impact re-run, all SDGs x all attacks (2026-08-21).
#
# Why the full sweep and not just the MST rows: Table `disparate_impact` in the
# manuscript does not match per_attack_disparity.csv (2026-05-30) on ANY row --
# Cell Supp. 33.1 vs 34.65, TabDDPM 1.4 vs 1.69, Synthpop 1.6 vs 1.89 -- so the
# printed table came from some earlier run than that CSV. Patching only the two
# encoding-affected MST rows would leave the table sourced from two different
# runs, which is exactly the provenance failure the float-binning hunt cost us.
# One coherent run on post-repair synth instead.
#
# Setting matches the table: adult 10k, QI1, sample_01. 10 SDGs x 7 attacks.
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

echo "=== full disparity re-run started $(date) ==="
taskset -c 12-23 python -u experiment_scripts/run_per_attack_disparity.py \
  --out experiment_scripts/per_attack_disparity_postrepair.csv
echo "=== DONE $(date) rc=$? ==="
