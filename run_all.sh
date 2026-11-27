#!/usr/bin/env bash
# Official runs: plain vs ResNet at depth 20 and 56 (He et al. 2016, CIFAR-10 §4.2),
# then plot the curves and write the results table.
#
# Usage:
#   ./run_all.sh                              # full run: 164 epochs each, results in runs/
#   EPOCHS=1 OUT=runs_test ./run_all.sh       # quick check: 1 epoch each, results in runs_test/
#
# Environment variables:
#   EPOCHS   epochs per model        (default 164)
#   OUT      output directory        (default runs)
#   WORKERS  data loader workers     (default 8)
#
# Output: $OUT/{plain20,resnet20,plain56,resnet56}/{log.csv,config.json,model.pt},
#         $OUT/curves.png, $OUT/results.md
# Stops at the first error (set -e).

set -e

EPOCHS="${EPOCHS:-164}"
OUT="${OUT:-runs}"
WORKERS="${WORKERS:-8}"

echo "EPOCHS=$EPOCHS OUT=$OUT WORKERS=$WORKERS"

python train.py --depth 20 --plain --epochs "$EPOCHS" --out "$OUT" --workers "$WORKERS"
python train.py --depth 20         --epochs "$EPOCHS" --out "$OUT" --workers "$WORKERS"
python train.py --depth 56 --plain --epochs "$EPOCHS" --out "$OUT" --workers "$WORKERS"
python train.py --depth 56         --epochs "$EPOCHS" --out "$OUT" --workers "$WORKERS"
python plot.py --runs "$OUT"
