#!/usr/bin/env bash
# Priority 1: Adult QI_tiny (10k)
# Story: bare demographic identity — age, sex, race only (3 known features)
# QI size:  3 known  →  12 hidden
# Answers:  bottom point of the QI-size RA curve
#
# Usage:   bash launch_01_adult_QI_tiny.sh
# Monitor: tail -f <repo>/outfiles/qi_adult_QI_tiny.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="/home/golobs/miniconda3/envs/recon_/bin/python"
LOG_FILE="$REPO_ROOT/outfiles/qi_adult_QI_tiny.log"

mkdir -p "$REPO_ROOT/outfiles"

echo "Launching sweep 1/6: adult QI_tiny (10k)"
echo "Log: $LOG_FILE"

nohup "$PYTHON" "$SCRIPT_DIR/run_qi_sweep.py" \
    --dataset adult \
    --qi QI_tiny \
    --workers 8 \
    --progress-log "$LOG_FILE" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "PID: $PID"
echo "Monitor: tail -f $LOG_FILE"
