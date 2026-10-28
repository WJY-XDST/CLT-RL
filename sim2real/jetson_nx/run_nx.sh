#!/usr/bin/env bash
set -euo pipefail
NX_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$NX_PROJECT_DIR"
# Set before importing NumPy: avoid BLAS worker jitter for a batch-one MLP.
export OPENBLAS_NUM_THREADS="${NX_BLAS_THREADS:-1}"
export OMP_NUM_THREADS="${NX_BLAS_THREADS:-1}"
export MKL_NUM_THREADS="${NX_BLAS_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NX_BLAS_THREADS:-1}"
exec "${NX_PYTHON:-python3}" -m wheelleg_deploy "$@"
