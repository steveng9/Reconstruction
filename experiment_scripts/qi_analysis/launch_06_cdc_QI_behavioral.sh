#!/usr/bin/env bash
# Priority 6: CDC Diabetes QI_behavioral (1k)
# Story: lifestyle/behavior prior — same size as QI1 (10 known), different feature types
# QI size:  10 known  →  12 hidden (identity + health outcomes hidden)
# Answers:  composition contrast with QI1 — in health data, does RA differ when
#           the adversary knows lifestyle vs. demographic features?
#
# NOTE: reuses existing synth.csv files unchanged — only the attack-time QI changes.
#
# Usage:   bash launch_06_cdc_QI_behavioral.sh
# Monitor: tail -f <repo>/outfiles/qi_cdc_diabetes_QI_behavioral.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="/home/golobs/miniconda3/envs/recon_/bin/python"
LOG_FILE="$REPO_ROOT/outfiles/qi_cdc_diabetes_QI_behavioral.log"

mkdir -p "$REPO_ROOT/outfiles"

echo "Launching sweep 6/6: cdc_diabetes QI_behavioral (1k)"
echo "Log: $LOG_FILE"

nohup "$PYTHON" "$SCRIPT_DIR/run_qi_sweep.py" \
    --dataset cdc_diabetes \
    --qi QI_behavioral \
    --workers 8 \
    --progress-log "$LOG_FILE" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "PID: $PID"
echo "Monitor: tail -f $LOG_FILE"
