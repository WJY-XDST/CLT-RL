#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$MODEL_DIR")"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/wheel_legged_isaaclab${PYTHONPATH:+:$PYTHONPATH}"
ACTION="${1:-help}"
if (( $# )); then shift; fi
case "$ACTION" in
  inspect) exec "$PROJECT_DIR/run_python.sh" "$MODEL_DIR/scripts/inspect_joints.py" "$@" ;;
  build) exec "$PROJECT_DIR/run_python.sh" "$MODEL_DIR/scripts/run_isaac.py" --closures "$MODEL_DIR/config/closures.json" "$@" ;;
  open) exec "$PROJECT_DIR/run_python.sh" "$MODEL_DIR/scripts/run_isaac.py" --open-chain "$@" ;;
  export) exec "$PROJECT_DIR/run_python.sh" "$MODEL_DIR/scripts/export_project_model.py" "$@" ;;
  bench) exec "$PROJECT_DIR/test_mine_drive.sh" "$@" ;;
  evaluate) exec "$PROJECT_DIR/run_python.sh" "$PROJECT_DIR/wheel_legged_isaaclab/scripts/evaluate_mine_policy.py" --output "$MODEL_DIR/results/policy_replay_$(date +%Y%m%d_%H%M%S)" "$@" ;;
  play) exec "$PROJECT_DIR/run_python.sh" "$MODEL_DIR/scripts/play_keyboard.py" "$@" ;;
  test) exec "$PROJECT_DIR/run_python.sh" "$MODEL_DIR/scripts/test_five_bar.py" "$@" ;;
  *) printf '%s\n' 'Usage: mine_model/run.sh {inspect|build|open|export|bench|evaluate|play|test} [options]' ;;
esac
