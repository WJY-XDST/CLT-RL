#!/usr/bin/env bash
set -euo pipefail

VERSION_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${VERSION_ROOT}/IsaacLab/_isaac_sim/python.sh" "$@"
