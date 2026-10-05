#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR/IsaacLab"
exec ./_isaac_sim/python.sh \
  ../wheel_legged_isaaclab/scripts/rsl_rl/train.py \
  --task WheelLeggedVMC-Flat-v0 \
  --experiment_name mine_wheel_legged_vmc_flat \
  --run_name mine_fresh --num_envs 256 --max_iterations 1000 --seed 43 "$@"
