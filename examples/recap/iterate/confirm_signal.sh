#!/bin/bash
# Confirm the RECAP-film +3 over SFT is real (not 100-episode noise): run SFT and
# RECAP-film@w=1.5 at extra seeds, combine with the existing seed-0 runs (86/89).
set -o pipefail
REPO="/home/duynguyen/Desktop/RESEARCH/VR/RLinf"
N1D7="$REPO/vla-rlft-n1d7/bin/python"
FILMCK="$REPO/examples/recap/results/recap_iter1_film/checkpoints/global_step_3000/actor/model_state_dict/full_weights.pt"
STATUS=/tmp/confirm_signal_status.log
mark(){ echo "=== $(date +%H:%M:%S) | $* ===" | tee -a "$STATUS"; }
envset(){ export REPO_PATH="$REPO" EMBODIED_PATH="$REPO/examples/embodiment" PYTHONPATH="$REPO:${PYTHONPATH:-}"
  export HF_LEROBOT_HOME=/data/datasets HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
  export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa; }
srof(){ "$N1D7" "$REPO/evaluations/libero/parse_per_task_sr.py" "$1" --runs 10 2>/dev/null | grep -oE "total_sr=[0-9.]+" | head -1; }
echo "CONFIRM_START $(date)" > "$STATUS"
for SEED in 1 2; do
  # SFT plain
  L=/tmp/confirm_sft_s${SEED}.log
  if ! grep -q "SUMMARY total_sr" "$L" 2>/dev/null; then
    mark "SFT eval seed=$SEED"
    ( source "$REPO/vla-rlft-n1d7/bin/activate"; envset
      bash "$REPO/evaluations/run_eval.sh" libero libero_10_gr00t_n1d7_sft_eval \
        "env.eval.seed=$SEED" "runner.logger.experiment_name=confirm_sft_s${SEED}" 2>&1 | tee "$L" | tail -1 )
  fi
  mark "SFT seed=$SEED -> $(srof "$L")"
  # RECAP-film w=1.5
  L=/tmp/confirm_film_w1.5_s${SEED}.log
  if ! grep -q "SUMMARY total_sr" "$L" 2>/dev/null; then
    mark "RECAP-film w=1.5 eval seed=$SEED"
    ( source "$REPO/vla-rlft-n1d7/bin/activate"; envset
      bash "$REPO/evaluations/run_eval.sh" libero libero_10_gr00t_n1d7_cfg_eval \
        "+runner.ckpt_path=$FILMCK" "+rollout.model.conditioning=film" \
        "rollout.model.cfg_guidance_weight=1.5" "env.eval.seed=$SEED" \
        "runner.logger.experiment_name=confirm_film_w1.5_s${SEED}" 2>&1 | tee "$L" | tail -1 )
  fi
  mark "RECAP-film w=1.5 seed=$SEED -> $(srof "$L")"
done
mark "CONFIRM COMPLETE"
echo "CONFIRM_EXIT=0" | tee -a "$STATUS"
