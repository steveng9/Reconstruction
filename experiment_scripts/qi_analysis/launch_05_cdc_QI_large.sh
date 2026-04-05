#!/usr/bin/env bash
# Priority 5: CDC Diabetes QI_large (1k)
# Story: strong health profile — demographic + health indicators + lifestyle (16 known features)
# QI size:  16 known  →  6 hidden (Diabetes_binary, Stroke, HeartDiseaseorAttack,
#                                   HvyAlcoholConsump, MentHlth, PhysHlth)
# Answers:  top point of the QI-size RA curve for the health dataset
#
# Usage:   bash launch_05_cdc_QI_large.sh
# Monitor: tail -f <repo>/outfiles/qi_cdc_diabetes_QI_large.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="/home/golobs/miniconda3/envs/recon_/bin/python"
LOG_FILE="$REPO_ROOT/outfiles/qi_cdc_diabetes_QI_large.log"

mkdir -p "$REPO_ROOT/outfiles"

echo "Launching sweep 5/6: cdc_diabetes QI_large (1k)"
echo "Log: $LOG_FILE"

nohup "$PYTHON" "$SCRIPT_DIR/run_qi_sweep.py" \
    --dataset cdc_diabetes \
    --qi QI_large \
    --workers 8 \
    --progress-log "$LOG_FILE" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "PID: $PID"
echo "Monitor: tail -f $LOG_FILE"
