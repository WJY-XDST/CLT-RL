#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${MUJOCO_PYTHON:-/home/aaa/miniconda3/envs/env_mujoco/bin/python}"
exec env -u PYTHONPATH -u LD_LIBRARY_PATH "$PYTHON" "$ROOT/wheel_legged_isaaclab/sim2sim/run.py" "$@"
