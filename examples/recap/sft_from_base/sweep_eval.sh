#!/usr/bin/env bash
# Bracketing sweep: quick 20-episode (2/task) eval of the zero-shot proxy and a
# sparse set of SFT checkpoints, to locate where SR crosses ~50%.
# Writes one consolidated summary to /tmp/n17_sweep_summary.txt.
set -o pipefail
REPO=/home/duynguyen/Desktop/RESEARCH/VR/RLinf
cd "$REPO"
EVAL=examples/recap/sft_from_base/eval_n1d7_sft_ckpt.sh
SFT=${SFT:-/data/checkpoints/n1d7_sft_from_base_20k/n1d7_sft_from_base_20k}
STEPS=${STEPS:-512}   # 512 = 20 episodes (2/task) quick bracket
SUM=/tmp/n17_sweep_summary.txt
: > "$SUM"

# (ckpt_dir, tag) -- 20k checkpoint first (best), then bracket to find ~50%
ITEMS=(
  "$SFT/checkpoint-20000|sft20000"
  "$SFT/checkpoint-8000|sft8000"
  "$SFT/checkpoint-12000|sft12000"
  "$SFT/checkpoint-16000|sft16000"
  "$SFT/checkpoint-4000|sft4000"
)

for item in "${ITEMS[@]}"; do
  ck="${item%%|*}"; tag="${item##*|}"
  echo "######## EVAL $tag ($ck) ########"
  bash "$EVAL" "$ck" "$tag" "$STEPS"
  # extract overall SR line printed by parse_per_task_sr.py
  sr=$(grep -iE "overall|total|mean|average" "/tmp/n17_sft_eval_${tag}.log" 2>/dev/null | tail -3)
  echo "[$tag] $ck" >> "$SUM"
  echo "$sr" >> "$SUM"
  echo "----" >> "$SUM"
done
echo "SWEEP_DONE" | tee -a "$SUM"
echo "==== SUMMARY ===="; cat "$SUM"
