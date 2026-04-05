#!/usr/bin/env bash
# Priority 3: Adult QI_behavioral (10k)
# Story: employment-only prior — same size as QI1 (6 known), totally different features
# QI size:  6 known  →  9 hidden (demographic + income identity hidden)
# Answers:  composition contrast with QI1 — does RA change when QI content shifts?
#
# NOTE: reuses existing synth.csv files unchanged — only the attack-time QI changes.
#
# Usage:   bash launch_03_adult_QI_behavioral.sh
# Monitor: tail -f <repo>/outfiles/qi_adult_QI_behavioral.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="/home/golobs/miniconda3/envs/recon_/bin/python"
LOG_FILE="$REPO_ROOT/outfiles/qi_adult_QI_behavioral.log"

mkdir -p "$REPO_ROOT/outfiles"

echo "Launching sweep 3/6: adult QI_behavioral (10k)"
echo "Log: $LOG_FILE"

nohup "$PYTHON" "$SCRIPT_DIR/run_qi_sweep.py" \
    --dataset adult \
    --qi QI_behavioral \
    --workers 8 \
    --progress-log "$LOG_FILE" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "PID: $PID"
echo "Monitor: tail -f $LOG_FILE"
