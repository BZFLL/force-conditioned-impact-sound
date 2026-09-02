#!/bin/bash
# DDSP main grid (GPU):
#   DDSP x within   4 levels x 5 seeds = 20 runs
#   DDSP x heldout  4 levels x 5 seeds = 20 runs
# Protocol: n_bands=2048, --modal on, epochs=500, batch=16, seeds 42..46.
# Resumable: cells with final_model.pt are skipped, so the script can be
# re-run until every cell is complete.
# Usage (with the training environment active; a GPU is required):
#   bash scripts/run_grid_ddsp.sh
set -u
cd "$(dirname "$0")/../src"
# Without torch every cell fails immediately and the loop "completes" in
# seconds, so check the environment before starting.
python3 -c "import torch" 2>/dev/null || { echo "FATAL: python3 cannot import torch; activate the training environment first."; exit 1; }
mkdir -p ../logs

run_cell () {
  local SPLIT="$1" ABL="$2" SEED="$3"
  local TAG="ablation_${ABL}_${SPLIT}_ddsp_s${SEED}"
  if [ -f "../experiments/${TAG}/final_model.pt" ]; then
    echo "[skip] $TAG"; return
  fi
  echo "[run ] $TAG  $(date '+%m-%d %H:%M:%S')"
  python3 -u train.py --backbone ddsp --ablation "$ABL" --split "$SPLIT" \
    --seed "$SEED" --epochs 500 --batch_size 16 --n_bands 2048 --modal on \
    --save_dir "../experiments/${TAG}" > "../logs/${TAG}.log" 2>&1
  echo "[done] $TAG  $(date '+%m-%d %H:%M:%S')"
}

echo "DDSP x within  $(date)"
for ABL in 1 2 3 4; do for SEED in 42 43 44 45 46; do
  run_cell within "$ABL" "$SEED"
done; done

echo "DDSP x heldout  $(date)"
for ABL in 1 2 3 4; do for SEED in 42 43 44 45 46; do
  run_cell heldout "$ABL" "$SEED"
done; done

echo "DDSP GRID COMPLETE $(date)"
