#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_SCRIPT="${ROOT_DIR}/main.py"
RUN_NAME="${RUN_NAME:-held_gba}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${LOG_DIR}"

python "${PYTHON_SCRIPT}" \
  held \
  --device cpu \
  --dataset gba \
  --eval_loss_iterations 40000 \
  --eval_loss_early_stop True \
  --eval_loss_early_stop_tol 1e-9 \
  --subgraph 3 \
  --d_bif 0.4 \
  --eta 0.0 \
  --discount_components True \
  --better_steiner True \
  --seed 42 \
  --log_dir "${LOG_DIR}"
