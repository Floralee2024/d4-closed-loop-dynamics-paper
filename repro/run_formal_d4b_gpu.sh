#!/usr/bin/env bash
set -euo pipefail

# Run the formal D4b sweep on one CUDA device. This script contains no
# credentials and assumes it is launched from the repository checkout.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${OUT_DIR:-outputs/formal_d4b_gpu}"

exec "${PYTHON_BIN}" source/d4b_residual_desync_gpu.py \\
  --steps 300 \\
  --seeds 0 1 2 3 4 5 6 7 8 9 \\
  --Ks 8 16 \\
  --Cs 0 0.5 1 2 3 5 8 12 16 24 \\
  --variants original \\
  --residual-strengths 0 0.1 0.25 0.5 1.0 \\
  --splits id ood_random ood_inverted \\
  --device cuda \\
  --amp auto \\
  --tf32 \\
  --batch-size 512 \\
  --eval-batch-size 2048 \\
  --num-workers "${NUM_WORKERS:-2}" \\
  --persistent-workers \\
  --no-plots \\
  --out-dir "${OUT_DIR}"
