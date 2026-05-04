#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_SCRIPT="${ROOT_DIR}/main.py"
RUN_NAME="${RUN_NAME:-chekuri_gba}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S)}"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-base}"

mkdir -p "${LOG_DIR}"

source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

python "${PYTHON_SCRIPT}" \
  chekuri \
  --device cpu \
  --mask_method edge \
  --dataset gba \
  --subgraph 3 \
  --eval_loss_iterations 40000 \
  --eval_loss_early_stop True \
  --eval_loss_early_stop_tol 1e-9 \
  --seed 42 \
  --chekuri_gamma_scale 1.0 \
  --log_dir "${LOG_DIR}"
