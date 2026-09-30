#!/bin/bash
# CVAE frames-only control L4' (level 5 in the code): 2 splits x level 5 x 5 seeds
# = 10 runs. Same protocol as scripts/run_grid_cvae.sh: 500 epochs, batch 16,
# lr 1e-3 and the other train.py defaults (latent_dim 32, beta_max 0.1,
# beta_cycle 100, beta_ratio 0.5, free_bits 0.25), best model selected by
# validation reconstruction loss, identity embedding object/material by split.
# Checkpoints to experiments/<cell>_s<seed>/ (explicit --save_dir: train.py's
# default has no seed suffix, so ten runs without it would share one directory).
# Resumable: cells with final_model.pt are skipped.
# CONCURRENCY (default 3) cells train at once; set CONCURRENCY=1 to run cells
# sequentially.
# Usage: bash scripts/run_grid_cvae_ablation5.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/src"
mkdir -p ../logs
CONCURRENCY="${CONCURRENCY:-3}"
# Resolve the interpreter explicitly so the script also runs without an activated venv.
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY=python3
export PY

run_cell () {
  local SPLIT=$1 SEED=$2
  local TAG="ablation_5_${SPLIT}_cvae_s${SEED}"
  if [ -f "../experiments/${TAG}/final_model.pt" ]; then
    echo "[skip] $TAG (done)"; return 0
  fi
  echo "[run ] $TAG  $(date '+%F %H:%M:%S')"
  "$PY" -u train.py --backbone cvae --ablation 5 --split "$SPLIT" \
    --seed "$SEED" --epochs 500 --batch_size 16 \
    --save_dir "../experiments/${TAG}" \
    > "../logs/${TAG}.log" 2>&1
  local RC=$?
  echo "[done] $TAG rc=$RC $(date '+%F %H:%M:%S')  tail: $(grep 'Best val' ../logs/${TAG}.log | tail -1)"
  return $RC
}
export -f run_cell

printf '%s\n' "within 42" "within 43" "within 44" "within 45" "within 46" \
              "heldout 42" "heldout 43" "heldout 44" "heldout 45" "heldout 46" \
  | xargs -P "$CONCURRENCY" -L 1 bash -c 'run_cell "$0" "$1"'
echo "CVAE LEVEL-5 GRID DONE $(date '+%F %H:%M:%S')"
