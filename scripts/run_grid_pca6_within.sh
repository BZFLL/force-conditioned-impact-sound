#!/bin/bash
# PCA-6 ladder rung (L3'' = data-driven 6-dim linear summary), DDSP x within:
# 5 cells (seeds 42..46). Protocol identical to the other ladder grids.
# Requires stats/pca6_basis_within.pt, the PCA basis fitted on the training
# split. Do not refit it: training and evaluation must share the identical
# basis.
# Usage (with the training environment active; a GPU is required):
#   bash scripts/run_grid_pca6_within.sh
set -u
cd "$(dirname "$0")/../src"
python3 -c "import torch" 2>/dev/null || { echo "FATAL: python3 cannot import torch; activate the training environment first."; exit 1; }
[ -f "../stats/pca6_basis_within.pt" ] || { echo "FATAL: stats/pca6_basis_within.pt missing (it is part of the repository)."; exit 1; }
mkdir -p ../logs

run_cell () {
  local SEED="$1"
  local TAG="ablation_3_within_ddsp_pca6_s${SEED}"
  if [ -f "../experiments/${TAG}/final_model.pt" ]; then echo "[skip] $TAG"; return; fi
  echo "[run ] $TAG  $(date '+%m-%d %H:%M:%S')"
  python3 -u train.py --backbone ddsp --ablation 3 --split within \
    --feature_set pca6 --seed "$SEED" --epochs 500 --batch_size 16 \
    --n_bands 2048 --modal on --save_dir "../experiments/${TAG}" \
    > "../logs/${TAG}.log" 2>&1
  echo "[done] $TAG  $(date '+%m-%d %H:%M:%S')"
}

echo "PCA-6 rung: DDSP x within (5 cells)  $(date)"
for SEED in 42 43 44 45 46; do run_cell "$SEED"; done
echo "PCA6 GRID COMPLETE $(date)"
