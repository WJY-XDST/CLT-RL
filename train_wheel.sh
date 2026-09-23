#!/usr/bin/env bash
set -euo pipefail

VERSION_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${VERSION_ROOT}/IsaacLab"
exec ./_isaac_sim/python.sh \
    ../wheel_legged_isaaclab/scripts/rsl_rl/train.py \
    --task WheelLeggedVMC-Flat-v0 "$@"
