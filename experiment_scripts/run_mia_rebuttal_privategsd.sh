#!/bin/bash
# run_mia_rebuttal_privategsd.sh
#
# Slow leg of the MIA-rebuttal sweep: generate PrivateGSD synth for
# nist_arizona 10k (25 features) at eps=1 and eps=1000, sample_01.
# PrivateGSD (jax genetic-algorithm synthesizer) has not been run at this
# dataset's size/feature-count before; on adult 10k (14 features) it took
# ~25-85 minutes per epsilon and climbs with epsilon, so this could take
# well over an hour per job with more features. Deliberately separated from
# run_mia_rebuttal_pipeline.sh so it doesn't block getting the other 3
# generators' MIA results back quickly.
#
# Runs the two epsilons strictly serially (--workers 1 --serial) since this
# is a first-time run at this size and memory footprint is unverified.
#
# Once this finishes, rerun run_mia_rebuttal_sweep.py alone (no need to
# rerun this script) to fill in the nist_arizona/PrivateGSD cells into
# mia_rebuttal_sweep_results.csv and wandb.

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

echo "=== generate PrivateGSD synth for nist_arizona 10k (eps=1,1000, sample_01) ==="
taskset -c 12-19 python experiment_scripts/generate_new_dp_sweep.py \
  --data-root ${RECON_DATA_ROOT}/nist_arizona_data/size_10000_25feat \
  --meta-path ${RECON_DATA_ROOT}/nist_arizona_data/meta.json \
  --methods PrivateGSD \
  --epsilons 1 1000 \
  --samples 1 \
  --workers 1 --serial

echo ""
echo "=== re-running MIA rebuttal sweep to fill in nist_arizona/PrivateGSD cells ==="
taskset -c 12-19 python experiment_scripts/run_mia_rebuttal_sweep.py

echo ""
echo "PIPELINE DONE: PrivateGSD generation + MIA rebuttal sweep re-run (nist_arizona 10k)"
