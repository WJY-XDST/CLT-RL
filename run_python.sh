#!/usr/bin/env bash
set -euo pipefail

VERSION_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Native headless Isaac processes have a large host-memory footprint as well
# as GPU memory. Serialize them across the real checkout and staging symlinks.
# CPU analysis and tests continue to run without this lock.
for clt_task_argument in "$@"; do
    if [[ "${clt_task_argument}" == "--headless" ]]; then
        clt_native_lock="${XDG_RUNTIME_DIR:-/tmp}/clt-rl-native-simulation-${UID}.lock"
        exec flock --exclusive --close -- "${clt_native_lock}" \
            "${VERSION_ROOT}/IsaacLab/_isaac_sim/python.sh" "$@"
    fi
done
exec "${VERSION_ROOT}/IsaacLab/_isaac_sim/python.sh" "$@"
