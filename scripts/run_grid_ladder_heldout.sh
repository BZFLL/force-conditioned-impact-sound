#!/bin/bash
# Fair-descriptor ladder, DDSP x heldout (unseen objects): the same 3 extra
# conditioning variants x 5 seeds = 15 cells, with split=heldout
# (material-stratified; identity = material embedding, resolved automatically
# by train.py for the held-out split). Protocol identical to the within ladder
# and the main held-out DDSP grid, so the cells are directly comparable to the
# L1-L4 held-out results.
#   L2'    = ablation 2, feature_set impulse
#   L3'_A  = ablation 3, feature_set v2A
#   L3'_AB = ablation 3, feature_set v2AB
# Resumable: cells with final_model.pt are skipped.
# Usage (with the training environment active; a GPU is required):
#   bash scripts/run_grid_ladder_heldout.sh
set -u
cd "$(dirname "$0")/../src"
python3 -c "import torch" 2>/dev/null || { echo "FATAL: python3 cannot import torch; activate the training environment first."; exit 1; }
mkdir -p ../logs

run_cell () {
  local ABL="$1" FS="$2" SEED="$3"
  local TAG="ablation_${ABL}_heldout_ddsp_${FS}_s${SEED}"
  if [ -f "../experiments/${TAG}/final_model.pt" ]; then echo "[skip] $TAG"; return; fi
  echo "[run ] $TAG  $(date '+%m-%d %H:%M:%S')"
  python3 -u train.py --backbone ddsp --ablation "$ABL" --split heldout \
    --feature_set "$FS" --seed "$SEED" --epochs 500 --batch_size 16 \
    --n_bands 2048 --modal on --save_dir "../experiments/${TAG}" \
    > "../logs/${TAG}.log" 2>&1
  echo "[done] $TAG  $(date '+%m-%d %H:%M:%S')"
}

echo "Fair-descriptor ladder: DDSP x heldout (15 cells)  $(date)"
for SEED in 42 43 44 45 46; do
  run_cell 2 impulse "$SEED"   # L2'
  run_cell 3 v2A     "$SEED"   # L3'_A
  run_cell 3 v2AB    "$SEED"   # L3'_AB
done
echo "LADDER HELDOUT GRID COMPLETE $(date)"
