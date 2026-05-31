#!/usr/bin/env bash
# run_mst_eps_fill_pipeline.sh
#
# Two-phase pipeline to complete the MST epsilon sweep:
#
#   Phase 1 — Regenerate MST synth for the four missing epsilons
#             [0.3, 3, 30, 300] × 5 samples with bin_continuous_as_ordinal=True,
#             matching the already-regenerated eps={0.1,1,10,100,1000} files.
#
#   Phase 2 — Rerun RF + NaiveBayes across ALL 9 MST epsilons,
#             QI_large and QI_behavioral, samples 00-03 (4 samples).
#             Output: mst_epsilon_sweep_<ts>.csv  → used to update tab:mst_eps_sweep.
#
# Usage (detached, survives logout):
#   nohup bash experiment_scripts/run_mst_eps_fill_pipeline.sh \
#       >experiment_scripts/outfiles/mst_eps_fill_pipeline.log 2>&1 &
#   echo "Pipeline PID=$!"
#
# Monitor:
#   tail -f experiment_scripts/outfiles/mst_eps_fill_pipeline.log
#
# Expected runtimes:
#   Phase 1: ~20-40 min  (MST gen for 4 eps × 5 samples, 4 workers)
#   Phase 2: ~30-60 min  (RF+NB are fast sklearn; 144 jobs, 8 workers)

set -e

REPO="/home/golobs/Reconstruction"
SCRIPTS="$REPO/experiment_scripts"
OUTFILES="$SCRIPTS/outfiles"

mkdir -p "$OUTFILES"

echo "================================================================"
echo "MST Epsilon Fill Pipeline — $(date)"
echo "================================================================"
echo ""

# ── Phase 1: Regenerate missing MST epsilons ──────────────────────────────────
echo "=== Phase 1: Regenerate MST synth for eps={0.3, 3, 30, 300} ==="
echo "    4 epsilons × 5 samples = 20 jobs  (4 workers)"
conda run -n recon_ python "$SCRIPTS/regen_mst_missing_eps.py" --workers 4
echo ""
echo "Phase 1 complete at $(date)"
echo ""

# ── Phase 2: RF + NaiveBayes sweep, all 9 epsilons ────────────────────────────
echo "=== Phase 2: RF + NaiveBayes sweep — all 9 MST epsilons ==="
echo "    9 eps × 2 QI × 2 attacks × 4 samples = 144 jobs  (8 workers)"
conda run -n recon_ python "$SCRIPTS/run_mst_epsilon_rf_nb_sweep.py" --workers 8
echo ""
echo "Phase 2 complete at $(date)"
echo ""

echo "================================================================"
echo "Pipeline DONE at $(date)"
echo "================================================================"
echo ""
echo "Next steps:"
echo "  1. Find the mst_epsilon_sweep_*.csv in experiment_scripts/"
echo "  2. Compute means by (attack, qi, epsilon) — pivot table printed above"
echo "  3. Update tab:mst_eps_sweep in manuscript/main_restructured_1.tex"
echo "     with the 4 rows × 9 epsilon columns"
