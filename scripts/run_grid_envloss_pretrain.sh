#!/bin/bash
# Training-side variants of Table 2 (DDSP, ablation 4):
#   synthetic-IR pretraining (1 run, 300 epochs) followed by fine-tuning on
#   the real splits (3 seeds x {within, heldout}), and the envelope-L1
#   auxiliary loss (heldout, 3 seeds; env_lambda 8.8).
# Protocol otherwise identical to the main DDSP grid. Resumable: cells with
# final_model.pt are skipped.
# Usage (with the training environment active; a GPU is required):
#   bash scripts/run_grid_envloss_pretrain.sh
set -u
cd "$(dirname "$0")/../src"
python3 -c "import torch" 2>/dev/null || { echo "FATAL: python3 cannot import torch; activate the training environment first."; exit 1; }
mkdir -p ../logs

# IR pool for the synthetic corpus (2,000 room IRs; one-time build)
[ -f "../cache/synthetic_ir_pool.pt" ] || python3 make_synthetic_ir_pool.py

PT=../experiments/pretrain
if [ ! -f "$PT/pretrain_synthetic/final_model.pt" ]; then
  echo "[run ] pretrain_synthetic  $(date '+%H:%M:%S')"
  python3 -u train.py --backbone ddsp --ablation 4 --split heldout \
    --synthetic --epochs 300 --batch_size 16 --n_bands 2048 --modal on \
    --save_dir "$PT/pretrain_synthetic" > "../logs/pretrain_synthetic.log" 2>&1
  echo "[done] pretrain_synthetic  $(date '+%H:%M:%S')"
fi

for SPLIT in within heldout; do
  for SEED in 42 43 44; do
    TAG="ft_${SPLIT}_L4_s${SEED}"
    if [ -f "$PT/${TAG}/final_model.pt" ]; then echo "[skip] $TAG"; continue; fi
    echo "[run ] $TAG  $(date '+%H:%M:%S')"
    python3 -u train.py --backbone ddsp --ablation 4 --split "$SPLIT" \
      --seed "$SEED" --epochs 500 --batch_size 16 --n_bands 2048 --modal on \
      --init_from "$PT/pretrain_synthetic/best_model.pt" \
      --save_dir "$PT/${TAG}" > "../logs/${TAG}.log" 2>&1
    echo "[done] $TAG  $(date '+%H:%M:%S')"
  done
done

for SEED in 42 43 44; do
  TAG="envloss_heldout_L4_s${SEED}"
  if [ -f "../experiments/envloss/${TAG}/final_model.pt" ]; then echo "[skip] $TAG"; continue; fi
  echo "[run ] $TAG  $(date '+%H:%M:%S')"
  python3 -u train.py --backbone ddsp --ablation 4 --split heldout \
    --seed "$SEED" --epochs 500 --batch_size 16 --n_bands 2048 --modal on \
    --env_lambda 8.8 \
    --save_dir "../experiments/envloss/${TAG}" > "../logs/${TAG}.log" 2>&1
  echo "[done] $TAG  $(date '+%H:%M:%S')"
done
echo "ENVLOSS+PRETRAIN GRID COMPLETE $(date)"
