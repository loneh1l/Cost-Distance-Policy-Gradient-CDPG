#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

DATASET="${DATASET:-gba}"
DEVICE="${DEVICE:-cpu}"
MASK_METHOD="${MASK_METHOD:-edge}"
SUBGRAPH="${SUBGRAPH:-3}"
EVAL_LOSS_ITERATIONS="${EVAL_LOSS_ITERATIONS:-40000}"
EVAL_LOSS_EARLY_STOP="${EVAL_LOSS_EARLY_STOP:-True}"
EVAL_LOSS_EARLY_STOP_TOL="${EVAL_LOSS_EARLY_STOP_TOL:-1e-9}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-meyerson_gba}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S)}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${LOG_DIR}"

"${PYTHON}" "${ROOT_DIR}/main.py" meyerson \
  --device "${DEVICE}" \
  --mask_method "${MASK_METHOD}" \
  --dataset "${DATASET}" \
  --subgraph "${SUBGRAPH}" \
  --save_pred_adj False \
  --eval_loss_iterations "${EVAL_LOSS_ITERATIONS}" \
  --eval_loss_early_stop "${EVAL_LOSS_EARLY_STOP}" \
  --eval_loss_early_stop_tol "${EVAL_LOSS_EARLY_STOP_TOL}" \
  --seed "${SEED}" \
  --log_dir "${LOG_DIR}" \
  ${EXTRA_ARGS}
