#!/usr/bin/env bash
# Official scoring loop: submission vs. real held-out data, normalised against
# the control-mean baseline exactly as the leaderboard does it.
#
#   scripts/score_with_cell_eval.sh <pred.h5ad> <real.h5ad> <outdir> [threads]
set -euo pipefail

PRED=${1:?pred h5ad}
REAL=${2:?real h5ad}
OUT=${3:?output dir}
THREADS=${4:-4}

mkdir -p "$OUT"
cell-eval baseline -a "$REAL" -o "$OUT/baseline.h5ad" -O "$OUT/baseline_de.csv" --skip-de -t "$THREADS"
cell-eval run -ap "$PRED"            -ar "$REAL" --profile full -o "$OUT/user" --num-threads "$THREADS"
cell-eval run -ap "$OUT/baseline.h5ad" -ar "$REAL" --profile full -o "$OUT/base" --num-threads "$THREADS"
cell-eval score -i "$OUT/user/agg_results.csv" -I "$OUT/base/agg_results.csv" -o "$OUT/score.csv"

echo
echo "=== score vs baseline (clipped at 0, averaged) ==="
cat "$OUT/score.csv"
