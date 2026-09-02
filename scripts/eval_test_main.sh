#!/bin/bash
# Re-runs the test evaluation of the main four-level grid for the four
# backbone x split combinations (DDSP/CVAE x within/held-out).
#
# evaluate.py discovers checkpoints by scanning a directory and pools them by
# the "ablation" field of config.json; the fair-descriptor ladder cells
# (impulse/v2A/v2AB/pca6) also carry ablation 2/3, so pointing evaluate.py at
# experiments/ directly would mix them into L2/L3. This wrapper therefore
# stages symlinks to exactly the 80 main-grid cells (run names without a
# feature-set suffix) and passes that directory as --ckpt.
# Pass --dev to run on the validation split instead.
# Usage: bash scripts/eval_test_main.sh [--dev]
set -euo pipefail
# MPS lacks aten::linalg_lstsq, so CVAE's InverseMelScale needs the CPU fallback.
export PYTORCH_ENABLE_MPS_FALLBACK=1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="${TMPDIR:-/tmp}/eval_test_main_stage"
rm -rf "$STAGE"; mkdir -p "$STAGE"

for a in 1 2 3 4; do
  for sp in within heldout; do
    for bk in ddsp cvae; do
      for s in 42 43 44 45 46; do
        d="$ROOT/experiments/ablation_${a}_${sp}_${bk}_s${s}"
        [ -d "$d" ] && ln -s "$d" "$STAGE/ablation_${a}_${sp}_${bk}_s${s}"
      done
    done
  done
done

n=$(ls "$STAGE" | wc -l | tr -d ' ')
if [ "$n" -ne 80 ]; then
  echo "FATAL: staged $n cells, expected exactly the 80 main-grid cells" >&2
  exit 1
fi
echo "[eval_test_main] staged 80/80 main-grid cells in $STAGE"

for bk in ddsp cvae; do
  for sp in within heldout; do
    echo "backbone=$bk split=$sp"
    python3 "$ROOT/src/evaluate.py" --backbone "$bk" --split "$sp" \
      --ckpt "$STAGE" "$@"
  done
done
