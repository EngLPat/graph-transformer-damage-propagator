#!/bin/bash
# Pooled Dice for the four attention-ablation arms, each at its reported
# checkpoint. Re-runs the rollout (the arm directories keep no rollout arrays),
# checks that rescore_r2.py reproduces the reported R^2, computes Dice with the
# recipe of the paper, then deletes the heavy artefacts (vtk and
# rollout_arrays.npz, ~1 GB each) inside the directory it created.
# Reads data/ and results/ under GTP_ROOT (default: this repository) and writes
# into OUT_DIR (default: results/paper_analysis).
# Usage, in the activated environment:
#   [GTP_ROOT=...] [OUT_DIR=...] bash tools/paper_analysis/dice_ablation.sh
set -o pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${GTP_ROOT:-$REPO}"
export GTP_SKIP_ANIM=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
S=${OUT_DIR:-results/paper_analysis}; mkdir -p "$S"
DATASET=data/processed/laminate/laminate_dmgstate_dataset_uniform_dt0.01.pt
SUMMARY=$S/dice_ablation_summary.txt
: > "$SUMMARY"
for spec in "B_message_passing ablation_no_attention" \
            "C_local_only ablation_local_only" \
            "D_global_only ablation_global_only" \
            "A_local_global ablation_reference"; do
  set -- $spec
  NAME=$1; RUN=results/$2
  OUT=$S/dice_abl/${NAME}
  mkdir -p "$OUT"
  echo "=========== $NAME  ($(date +%H:%M:%S)) ===========" | tee -a "$SUMMARY"
  python -m src.gtp --mode test --dataset "$DATASET" \
      --checkpoint "$RUN/best_models/model_$2_nl9_hd128.pt" \
      --stats "$RUN/normalization_stats.pt" \
      --output-dir "$OUT" --device cuda:0 > "$OUT/test.log" 2>&1 \
    || { echo "  FAILED, see $OUT/test.log" | tee -a "$SUMMARY"; continue; }
  python "$REPO/tools/rescore_r2.py" "$OUT" >> "$OUT/test.log" 2>&1 || true
  grep -E "FI integrated \(all\)|damage ω \(all\)" "$OUT/test_results/rescored_metrics.txt" \
      | sed 's/^/  R2 check: /' | tee -a "$SUMMARY"
  python "$REPO/tools/paper_analysis/dice_pooled.py" "$OUT/test_results/rollout_arrays.npz" "  Dice:" | tee -a "$SUMMARY"
  rm -rf "$OUT/test_results/vtk" "$OUT/test_results/damage_evolution" \
         "$OUT/test_results/rollout_arrays.npz"
done
echo "=========== done ($(date +%H:%M:%S)) ===========" | tee -a "$SUMMARY"
