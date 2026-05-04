#!/bin/bash
# Graph construction and transport cost simulation shell script
# Usage: ./run_simulation.sh

PYTHON_SCRIPT="./main.py"
RUN_NAME="${RUN_NAME:-cdpg_gba}"
LOG_DIR="${LOG_DIR:-./logs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S)}"

# Ensure execution from project root directory
cd "$(dirname "$0")/.."
mkdir -p "$LOG_DIR"

python "$PYTHON_SCRIPT" \
  cdpg \
  --device gpu \
  --mask_method acyclic \
  --dataset gba \
  --predict_mode topo_greedy  \
  --subgraph 3 \
  --save_pred_adj False \
  --loss_iterations 100 \
  --loss_early_stop False \
  --loss_early_stop_tol 1e-6 \
  --loss_iterations_mode logn \
  --loss_iterations_logn_mul 2 \
  --eval_loss_iterations 40000 \
  --eval_loss_early_stop True \
  --eval_loss_early_stop_tol 1e-9 \
  --lora_rank 12 \
  --lr 0.06 \
  --train_iterations 300 \
  --fixed_train_iters False \
  --train_early_stop_tol 1e-4 \
  --alpha 1.0 \
  --use_lora True \
  --use_unreached True \
  --initial_point_type random \
  --unreached_weight 10 \
  --dynamic_mask True \
  --dynamic_mask_loss_check True \
  --dynamic_mask_loss_check_steps 1 \
  --dynamic_mask_interval 8 \
  --loss_solver iter \
  --unreached_type flow \
  --log_dir "$LOG_DIR"
