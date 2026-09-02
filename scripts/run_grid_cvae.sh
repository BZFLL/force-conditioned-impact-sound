#!/bin/bash
# CVAE full grid: 2 splits x 4 levels x 5 seeds = 40 runs.
# Protocol: 500 epochs, batch 16, lr 1e-3, best model selected by validation
# reconstruction loss. Identity embedding: object for the within split,
# material for the held-out split (train.py defaults).
# Runs cells sequentially. Logs to logs/, checkpoints to experiments/<cell>_s<seed>/.
# Resumable: cells with final_model.pt are skipped.
# Usage: bash scripts/run_grid_cvae.sh
set -u
cd "$(dirname "$0")/../src"
mkdir -p ../logs
for SPLIT in within heldout; do
  for ABL in 1 2 3 4; do
    for SEED in 42 43 44 45 46; do
      TAG="ablation_${ABL}_${SPLIT}_cvae_s${SEED}"
      if [ -f "../experiments/${TAG}/final_model.pt" ]; then
        echo "[skip] $TAG (done)"; continue
      fi
      echo "[run ] $TAG  $(date '+%H:%M:%S')"
      python3 train.py --backbone cvae --ablation "$ABL" --split "$SPLIT" \
        --seed "$SEED" --epochs 500 --batch_size 16 \
        --save_dir "../experiments/${TAG}" \
        > "../logs/${TAG}.log" 2>&1
      echo "[done] $TAG  $(date '+%H:%M:%S')  tail: $(grep 'Best val' ../logs/${TAG}.log | tail -1)"
    done
  done
done
echo "CVAE GRID DONE $(date)"
