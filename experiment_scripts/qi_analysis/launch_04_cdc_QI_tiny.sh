#!/usr/bin/env bash
# Priority 4: CDC Diabetes QI_tiny (1k)
# Story: bare demographic knowledge — Sex, Age, BMI, GenHlth (4 known features)
# QI size:  4 known  →  18 hidden
# Answers:  bottom point of the QI-size RA curve for the health dataset
#
# Usage:   bash launch_04_cdc_QI_tiny.sh
# Monitor: tail -f <repo>/outfiles/qi_cdc_diabetes_QI_tiny.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="/home/golobs/miniconda3/envs/recon_/bin/python"
LOG_FILE="$REPO_ROOT/outfiles/qi_cdc_diabetes_QI_tiny.log"

mkdir -p "$REPO_ROOT/outfiles"

echo "Launching sweep 4/6: cdc_diabetes QI_tiny (1k)"
echo "Log: $LOG_FILE"

nohup "$PYTHON" "$SCRIPT_DIR/run_qi_sweep.py" \
    --dataset cdc_diabetes \
    --qi QI_tiny \
    --workers 8 \
    --progress-log "$LOG_FILE" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "PID: $PID"
echo "Monitor: tail -f $LOG_FILE"
