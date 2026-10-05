#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_ROOT"
export PYTHONPATH="$TASK_ROOT/wheel_legged_isaaclab${PYTHONPATH:+:$PYTHONPATH}"
exec "$TASK_ROOT/run_python.sh" "$TASK_ROOT/wheel_legged_isaaclab/scripts/test_mine_drive.py" "$@"
