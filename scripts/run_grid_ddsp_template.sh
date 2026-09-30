#!/bin/bash
# Template-pulse control: DDSP level 4 whose per-frame
# route receives the split's template pulse scaled to each strike's window
# peak (--force_input template_peak). 2 splits x 5 seeds = 10 runs, the L4
# recipe: 500 epochs, batch 16, lr 1e-3, 2048 noise bands, modal branch on,
# identity object/material by split, best model selected by validation MRSTFT.
# Checkpoints to experiments_template/template_4_<split>_ddsp_s<seed>/, outside
# experiments/: evaluate.py's discover_checkpoints pools every run directory
# under a root by config["ablation"], so a template-pulse run inside experiments/ would be
# taken for an L4 seed.
# Resumable: cells with final_model.pt are skipped.
# CONCURRENCY (default 3) cells train at once.
# Usage: bash scripts/run_grid_ddsp_template.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/src"
mkdir -p ../logs ../experiments_template
CONCURRENCY="${CONCURRENCY:-3}"
# Resolve the interpreter explicitly so the script also runs without an activated venv.
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY=python3
export PY

run_cell () {
  local SPLIT=$1 SEED=$2
  local TAG="template_4_${SPLIT}_ddsp_s${SEED}"
  if [ -f "../experiments_template/${TAG}/final_model.pt" ]; then
    echo "[skip] $TAG (done)"; return 0
  fi
  echo "[run ] $TAG  $(date '+%F %H:%M:%S')"
  "$PY" -u train.py --backbone ddsp --ablation 4 --force_input template_peak \
    --split "$SPLIT" --seed "$SEED" --epochs 500 --batch_size 16 \
    --n_bands 2048 --modal on \
    --save_dir "../experiments_template/${TAG}" \
    > "../logs/${TAG}.log" 2>&1
  local RC=$?
  echo "[done] $TAG rc=$RC $(date '+%F %H:%M:%S')  tail: $(grep 'Best val' ../logs/${TAG}.log | tail -1)"
  return $RC
}
export -f run_cell

printf '%s\n' "within 42" "within 43" "within 44" "within 45" "within 46" \
              "heldout 42" "heldout 43" "heldout 44" "heldout 45" "heldout 46" \
  | xargs -P "$CONCURRENCY" -L 1 bash -c 'run_cell "$0" "$1"'
echo "DDSP TEMPLATE GRID DONE $(date '+%F %H:%M:%S')"
