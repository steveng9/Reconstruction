#!/usr/bin/env bash
# run_mst_regen_pipeline.sh
#
# Three-phase pipeline to fix the MST float-encoding artefact in adult 10k:
#
#   Phase 1 — Regenerate MST synth for epsilons [0.1,1,10,100,1000] × 5 samples
#             with bin_continuous_as_ordinal=True throughout.
#   Phase 2 — Recompute Wasserstein OHE for all adult SDG methods
#             (captures the corrected MST eps=0.1 values).
#   Phase 3 — Rerun all Table 1 attacks against MST eps=0.1 using new synth.
#
# Detached usage (survives logout):
#   nohup bash experiment_scripts/run_mst_regen_pipeline.sh \
#       >experiment_scripts/outfiles/mst_regen_pipeline.log 2>&1 &
#   echo "Pipeline PID=$!"
#
# Monitor:
#   tail -f experiment_scripts/outfiles/mst_regen_pipeline.log
#
# Expected runtimes:
#   Phase 1: ~30–60 min  (MST generation, CPU-bound, 5 parallel workers)
#   Phase 2: ~5 min      (Wasserstein OHE computation)
#   Phase 3: ~2–3 hrs    (non-diffusion attacks fast; diffusion attacks slow)
#
# Output CSV for Phase 3: experiment_scripts/mst_corrected_attacks_<ts>.csv
# Output CSV for Phase 2: experiment_scripts/wasserstein_ohe_regen_<date>.csv

set -e   # abort on any error

REPO="/home/golobs/Reconstruction"
SCRIPTS="$REPO/experiment_scripts"
OUTFILES="$SCRIPTS/outfiles"
DATE=$(date +%Y%m%d)

mkdir -p "$OUTFILES"

echo "================================================================"
echo "MST Regen Pipeline — $(date)"
echo "================================================================"
echo ""

# ── Phase 1: Regenerate MST synth ─────────────────────────────────────────────
echo "=== Phase 1: Regenerate MST synth (bin_continuous_as_ordinal=True) ==="
echo "    Epsilons: 0.1, 1, 10, 100, 1000  ×  5 samples = 25 jobs"
conda run -n recon_ python "$SCRIPTS/regen_mst_adult10k.py" --workers 5
echo ""
echo "Phase 1 complete at $(date)"
echo ""

# ── Phase 2: Recompute Wasserstein OHE ────────────────────────────────────────
echo "=== Phase 2: Recompute Wasserstein OHE for adult ==="
WASS_OUT="$SCRIPTS/wasserstein_ohe_regen_${DATE}.csv"
conda run -n recon_ python "$SCRIPTS/compute_wasserstein_ohe.py" \
    --dataset adult \
    --out "$WASS_OUT"
echo ""
echo "Phase 2 complete at $(date)"
echo "Wasserstein results: $WASS_OUT"
echo ""

# Quick summary of new Wasserstein values for MST epsilons
echo "--- Wasserstein OHE for MST epsilons (mean over 5 samples) ---"
conda run -n recon_ python - <<'PYEOF'
import pandas as pd, glob, sys

# Use the most recent regen file
files = sorted(glob.glob("/home/golobs/Reconstruction/experiment_scripts/wasserstein_ohe_regen_*.csv"))
if not files:
    print("  No regen wasserstein file found yet.")
    sys.exit(0)
df = pd.read_csv(files[-1])
mst = df[df["method"].str.startswith("MST")].copy() if "method" in df.columns else df
print(mst.groupby("method")["wasserstein_ohe"].mean().sort_values().to_string())
PYEOF
echo ""

# ── Phase 3: Attack sweep (MST eps=0.1, corrected synth, all Table 1 attacks) ──
echo "=== Phase 3: Rerun all Table 1 attacks on MST eps=0.1 ==="
echo "    19 attacks × 5 samples = 95 jobs  (SVM excluded)"
echo "    Diffusion attacks will retrain from scratch (~1-2 hrs/sample)"
conda run -n recon_ python "$SCRIPTS/run_adult10k_mst_eps01_attacks.py" --workers 8
echo ""
echo "Phase 3 complete at $(date)"
echo ""

echo "================================================================"
echo "Pipeline DONE at $(date)"
echo "================================================================"
echo ""
echo "Next steps:"
echo "  1. Check wasserstein_ohe_regen_${DATE}.csv — should now decrease"
echo "     monotonically with eps for MST (or at least not show the large"
echo "     spike at eps=10)."
echo "  2. Check mst_corrected_attacks_*.csv — Mode/Random should now be ~10.5"
echo "     instead of 9.5; all ML attacks should shift accordingly."
echo "  3. Update manuscript tables (tab:ra_mean_adult and tab:quality_overview)"
echo "     with the corrected values."
