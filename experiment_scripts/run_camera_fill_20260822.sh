#!/bin/bash
# Camera-ready gap-fill, 2026-08-22. Three independent detached jobs.
#
#  1. quality   — recompute ALL quality metrics in the current environment.
#                 Necessary (not merely convenient): `sdv_col_pairs` changed with the
#                 SDV version, so the March value in Table 2 (RankSwap 0.929) is not
#                 reproducible today (0.845). Adding the new generators without
#                 recomputing the old rows would put two SDV versions in one column.
#                 All datasets, so the four appendix quality tables stay consistent too.
#  2. wass_ohe  — the custom one-hot Wasserstein metric for every synth file, incl.
#                 the four new DP generators (absent from the March merge).
#  3. table7    — memorization (train vs round-robin holdout) for the four new DP
#                 generators at eps 1 and 1000, Adult 1k, QI_linear, RF+KNN+MLP —
#                 matching Table 7's existing rows exactly. Includes MST eps=1 and
#                 CellSuppression as controls: they must reproduce 53.7/53.9 and
#                 92.7/72.5, otherwise the new rows are not comparable.
#
# Cores 36-47 are deliberately left free for other users on this machine.
# NOTE: no `set -u` — conda activate.d/libblas_mkl_activate.sh reads an unset
# variable and would abort the whole job under nounset.
cd /home/golobs/Reconstruction
source /home/golobs/miniconda3/etc/profile.d/conda.sh
conda activate recon_
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2

log() { echo "=== $1 START $(date) ==="; }

case "${1:-}" in
  quality)
    log quality
    taskset -c 0-15 python -u experiment_scripts/evaluate_synth_quality.py --workers 12 \
      --out experiment_scripts/synth_quality_results_20260822_camera.csv
    echo "=== quality DONE $(date) rc=$? ===" ;;
  wass_ohe)
    log wass_ohe
    taskset -c 16-23 python -u experiment_scripts/compute_wasserstein_ohe.py --workers 6 \
      --out experiment_scripts/wasserstein_ohe_20260822_camera.csv
    echo "=== wass_ohe DONE $(date) rc=$? ===" ;;
  table7)
    log table7
    taskset -c 24-35 python -u experiment_scripts/run_table7_memorization_newdp.py \
      --dataset adult --size 1000 --workers 8
    echo "=== table7 DONE $(date) rc=$? ===" ;;
  *) echo "usage: $0 {quality|wass_ohe|table7}"; exit 2 ;;
esac
