#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_SCRIPT="${ROOT_DIR}/main.py"

SUBGRAPH="${SUBGRAPH:-3}"
MASK_METHOD="${MASK_METHOD:-edge}"
TIME_LIMIT="${TIME_LIMIT:-7200}"
MIP_FOCUS="${MIP_FOCUS:-1}"
THREADS="${THREADS:-1}"
RUN_NAME="${RUN_NAME:-gurobi_gba${SUBGRAPH}_${MASK_METHOD}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${LOG_DIR}"

python "${PYTHON_SCRIPT}" \
  gurobi \
  --device cpu \
  --dataset gba \
  --subgraph "${SUBGRAPH}" \
  --mask_method "${MASK_METHOD}" \
  --eval_loss_iterations 40000 \
  --eval_loss_early_stop True \
  --eval_loss_early_stop_tol 1e-9 \
  --gurobi_time_limit "${TIME_LIMIT}" \
  --gurobi_mip_gap 1e-4 \
  --gurobi_mip_focus "${MIP_FOCUS}" \
  --gurobi_threads "${THREADS}" \
  --gurobi_verbose False \
  --log_dir "${LOG_DIR}"
