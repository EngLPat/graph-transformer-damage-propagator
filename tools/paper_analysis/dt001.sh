#!/bin/bash
# Full-resolution (every Delta t = 0.01 step) rollouts for the timing analysis.
# Reads data/ and results/ under GTP_ROOT (default: this repository) and writes
# into OUT_DIR (default: results/paper_analysis).
# Usage, in the activated environment:
#   [GTP_ROOT=...] [OUT_DIR=...] bash tools/paper_analysis/dt001.sh
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${GTP_ROOT:-$REPO}"; export GTP_SKIP_ANIM=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
S=${OUT_DIR:-results/paper_analysis}; mkdir -p "$S"
LOG=$S/dt001_summary.txt; echo "start $(date '+%F %T')" > $LOG
run() { # name dataset checkpoint stats
  OUT=$S/dt001_$1; mkdir -p $OUT
  python -m src.gtp --mode test --dataset $2 --checkpoint $3 --stats $4 --output-dir $OUT --device cuda:0 --rollout-downsample 1 > $OUT/test.log 2>&1 || { echo "$1 FAILED" >> $LOG; return; }
  python "$REPO/tools/rescore_r2.py" $OUT >> $OUT/test.log 2>&1
  echo "=== $1" >> $LOG; grep -E "Rollout settings|D-state feedback [A-Z]*" -o $OUT/test.log | head -2 >> $LOG
  grep -E "FI integrated \(all\)|damage ω \(all\)" $OUT/test_results/rescored_metrics.txt >> $LOG
  rm -rf $OUT/test_results/vtk $OUT/test_results/damage_evolution
}
run fixed data/processed/laminate/laminate_dmgstate_dataset_uniform_dt0.01.pt \
    results/laminate/best_models/model_laminate_nl9_hd128.pt results/laminate/normalization_stats.pt
run yt data/processed/laminate_yt_variability/laminate_yt_variability_dmgstate_ytfeat_dataset_uniform_dt0.01.pt \
    results/yt_variability/best_models/model_yt_variability_nl9_hd128.pt results/yt_variability/normalization_stats.pt
echo "done $(date '+%F %T')" >> $LOG
