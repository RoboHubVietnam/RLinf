#!/bin/bash
# Train CFG-FILM on round-1 advantages (reused) and sweep guidance weight to test
# whether classifier-free guidance (w>1) lifts RECAP above the 86% SFT baseline.
# Film supports dual-pass guidance (text does not), so this is the real RECAP lever.
set -o pipefail
REPO="/home/duynguyen/Desktop/RESEARCH/VR/RLinf"
N1D7="$REPO/vla-rlft-n1d7/bin/python"
STATUS=/tmp/film_sweep_status.log
DS="/data/datasets/recap_iter1"
CFGEXP="recap_iter1_film"
CFGOUT="$REPO/examples/recap/results/${CFGEXP}/checkpoints"
NEWCK="$CFGOUT/global_step_3000/actor/model_state_dict/full_weights.pt"
mark(){ echo "=== $(date +%H:%M:%S) | $* ===" | tee -a "$STATUS"; }
common_env(){ export REPO_PATH="$REPO" EMBODIED_PATH="$REPO/examples/embodiment" PYTHONPATH="$REPO:${PYTHONPATH:-}"
  export HF_LEROBOT_HOME=/data/datasets HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
  export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; }

echo "FILM_SWEEP_START $(date)" > "$STATUS"
# ---- train CFG-film (masked, reuse round-1 advantages) ----
if [ -f "$NEWCK" ]; then
  mark "CFG-film train SKIP (ckpt exists)"
else
  mark "CFG-film train (max_steps=3000) from SFT on round-1 advantages"
  ( source "$REPO/vla-rlft-n1d7/bin/activate"; common_env; cd "$REPO/examples/recap/cfg"
    python train_cfg.py --config-name libero_cfg_gr00t_n1d7 \
      "data.train_data_paths.0.dataset_path=$DS" data.advantage_tag=rollout \
      "runner.logger.experiment_name=$CFGEXP" \
      "runner.max_steps=3000" \
      "actor.optim.total_training_steps=3000" "actor.optim.lr_warmup_steps=300" 2>&1 | tail -3 )
  [ -f "$NEWCK" ] || { mark "ABORT: CFG-film ckpt missing"; exit 1; }
  mark "CFG-film train done"
fi

# ---- guidance sweep ----
for W in 1.0 1.5 2.0 2.5; do
  TAG="film_w${W}"
  if grep -q "SUMMARY total_sr" "/tmp/n17_cfg_eval_${TAG}.log" 2>/dev/null; then
    mark "eval w=$W SKIP (log exists)"
  else
    mark "eval CFG-film w=$W"
    ( source "$REPO/vla-rlft-n1d7/bin/activate"; common_env
      bash "$REPO/evaluations/libero/eval_n1d7_cfg_ckpt.sh" "$NEWCK" "$TAG" "$W" film 2>&1 | tail -2 )
  fi
  SR=$("$N1D7" "$REPO/evaluations/libero/parse_per_task_sr.py" "/tmp/n17_cfg_eval_${TAG}.log" --runs 10 2>/dev/null | grep -oE "total_sr=[0-9.]+" | head -1)
  mark "RESULT film w=$W -> $SR"
done
mark "FILM_SWEEP COMPLETE"
echo "FILM_SWEEP_EXIT=0" | tee -a "$STATUS"
