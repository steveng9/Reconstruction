#!/usr/bin/env bash
# Priority 2: Adult QI_large (10k)
# Story: strong adversary — demographics + full employment context (10 known features)
# QI size:  10 known  →  5 hidden (fnlwgt, education-num, capital-gain, capital-loss, income)
# Answers:  top point of the QI-size RA curve
#
# Usage:   bash launch_02_adult_QI_large.sh
# Monitor: tail -f <repo>/outfiles/qi_adult_QI_large.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="/home/golobs/miniconda3/envs/recon_/bin/python"
LOG_FILE="$REPO_ROOT/outfiles/qi_adult_QI_large.log"

mkdir -p "$REPO_ROOT/outfiles"

echo "Launching sweep 2/6: adult QI_large (10k)"
echo "Log: $LOG_FILE"

nohup "$PYTHON" "$SCRIPT_DIR/run_qi_sweep.py" \
    --dataset adult \
    --qi QI_large \
    --workers 8 \
    --progress-log "$LOG_FILE" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "PID: $PID"
echo "Monitor: tail -f $LOG_FILE"
