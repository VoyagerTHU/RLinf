#!/usr/bin/env bash
# Stop a run started by launch_gr1_training.sh: kill the launcher's process
# group, then stop the private Ray head.
#   bash examples/embodiment/migration/stop_gr1_training.sh <EXP_ROOT> [TRAIN_PYTHON]
set -uo pipefail
EXP_ROOT=${1:?usage: stop_gr1_training.sh <EXP_ROOT> [TRAIN_PYTHON]}
TRAIN_PYTHON=${2:-${TRAIN_PYTHON:-python}}
if [[ -f "$EXP_ROOT/launcher.pid" ]]; then
    pid=$(cat "$EXP_ROOT/launcher.pid")
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    if [[ -n "$pgid" ]]; then
        echo "TERM -> process group $pgid (launcher pid $pid)"
        kill -TERM -- "-$pgid" 2>/dev/null || true
        sleep 20
        kill -KILL -- "-$pgid" 2>/dev/null || true
    fi
fi
"$(dirname "$TRAIN_PYTHON")/ray" stop --force >/dev/null 2>&1 || true
pkill -u "$(id -un)" -f "train_embodied_agent.py" 2>/dev/null || true
echo "stopped"
