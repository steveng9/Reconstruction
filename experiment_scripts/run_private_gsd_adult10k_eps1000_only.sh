#!/bin/bash
# run_private_gsd_adult10k_eps1000_only.sh
#
# Backfill job: run_private_gsd_10k_pipeline.sh capped the full 9-epsilon
# PrivateGSD generation sweep at 6h and got through 8/9 epsilons (0.1 through
# 300) before the cap killed eps=1000 mid-run. This script generates just the
# missing eps=1000 point, with no artificial timeout (nist_arizona's single
# eps=1 run took ~3.8h at 25 features; adult has only 14 features so this
# should be comparable or faster, but we don't cap it since we've already
# seen a hard cap truncate a real in-progress run once).
#
# Once synth.csv lands, reruns run_mia_rebuttal_sweep.py to backfill the
# adult/PrivateGSD_eps1000 cell in mia_rebuttal_sweep_results.csv (10k table).
#
# Cores 24-35 are free (private_gsd_10k_pipeline.sh released them when it
# finished). Cores 12-19 in use by the nist_arizona PrivateGSD job. Cores
# 36-47 stay untouched for other users (daniilf, sikha).
cd /home/golobs/Reconstruction
source /home/golobs/miniconda3/etc/profile.d/conda.sh
conda activate recon_

export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

echo "=== generate PrivateGSD synth for adult 10k (eps=1000 only, sample_00) ==="
taskset -c 24-35 python experiment_scripts/generate_new_dp_sweep.py \
  --data-root /home/golobs/data/reconstruction_data/adult/size_10000 \
  --meta-path /home/golobs/data/reconstruction_data/adult/meta.json \
  --methods PrivateGSD \
  --epsilons 1000 \
  --samples 0 \
  --workers 1

echo ""
echo "=== re-running MIA rebuttal sweep to fill in adult/PrivateGSD_eps1000 cell ==="
taskset -c 24-35 python experiment_scripts/run_mia_rebuttal_sweep.py

echo ""
echo "PIPELINE DONE: PrivateGSD adult 10k eps=1000 backfill (generation + MIA rebuttal sweep re-run)"
