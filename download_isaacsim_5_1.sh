#!/usr/bin/env bash
set -euo pipefail

download_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/downloads"
url="https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip"
archive="${download_dir}/isaac-sim-standalone-5.1.0-linux-x86_64.zip"

mkdir -p "${download_dir}"

ranges=(
  "0-1096052471"
  "1096052472-2192104943"
  "2192104944-3288157415"
  "3288157416-4384209887"
  "4384209888-5480262359"
  "5480262360-6576314831"
  "6576314832-7672367303"
  "7672367304-8768419776"
)

pids=()
for index in "${!ranges[@]}"; do
  part="${download_dir}/isaac-sim-5.1.0.part$(printf '%02d' "${index}")"
  curl -L --fail --silent --show-error \
    --range "${ranges[${index}]}" \
    --output "${part}" \
    "${url}" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

expected_sizes=(1096052472 1096052472 1096052472 1096052472 1096052472 1096052472 1096052472 1096052473)
for index in "${!expected_sizes[@]}"; do
  part="${download_dir}/isaac-sim-5.1.0.part$(printf '%02d' "${index}")"
  actual_size="$(stat -c '%s' "${part}")"
  if [[ "${actual_size}" != "${expected_sizes[${index}]}" ]]; then
    echo "Invalid size for ${part}: expected ${expected_sizes[${index}]}, got ${actual_size}" >&2
    exit 1
  fi
done

cat "${download_dir}"/isaac-sim-5.1.0.part?? > "${archive}"

archive_size="$(stat -c '%s' "${archive}")"
if [[ "${archive_size}" != "8768419777" ]]; then
  echo "Invalid archive size: expected 8768419777, got ${archive_size}" >&2
  exit 1
fi

echo "Download complete: ${archive} (${archive_size} bytes)"
