#!/usr/bin/env bash
# Fine-tune base GR00T-N1.7-3B on LIBERO-Long (LeRobot) using the public N1.7
# OSS finetune entrypoint (gr00t.experiment.launch_finetune).
#
# Goal: bring zero-shot base (~0%) up to ~50% with a *short* SFT, so we can then
# run RECAP on an under-trained (non-ceiling) policy and measure a larger gain.
#
# Produces a Gr00tN1d7 HF checkpoint (config.json + model-*.safetensors +
# experiment_cfg/dataset_statistics.json) that loads in the RLinf collect/eval/
# CFG pipeline by construction (same gr00t package the pipeline imports).
#
# Env knobs (all optional, sane defaults):
#   BASE_MODEL   base checkpoint dir            (default: GR00T-N1.7-3B-base)
#   DATASET      LeRobot dataset root           (default: libero_10_no_noops_lerobot)
#   OUTPUT_DIR   where checkpoints land
#   EXP_NAME     wandb run name
#   MAX_STEPS    total training steps           (default: 4000)
#   SAVE_STEPS   checkpoint interval            (default: 500)
#   GLOBAL_BATCH per-optimizer-step batch       (default: 16; per-device = GLOBAL_BATCH/num_gpus)
#   GRAD_ACCUM   gradient accumulation steps    (default: 2)  effective batch = GLOBAL_BATCH*GRAD_ACCUM
#   LR           learning rate                  (default: 1e-4)
#   USE_WANDB    1/0                            (default: 1)
set -euo pipefail

REPO=/home/duynguyen/Desktop/RESEARCH/VR/RLinf
PY="$REPO/vla-rlft-n1d7/bin/python"

BASE_MODEL=${BASE_MODEL:-/data/checkpoints/GR00T-N1.7-3B-base}
# clean copy: phantom return/reward features stripped from info.json (the RECAP
# value pipeline added them with no matching data columns -> breaks stats gen).
DATASET=${DATASET:-/data/datasets/libero_10_sft_clean}
OUTPUT_DIR=${OUTPUT_DIR:-/data/checkpoints/n1d7_sft_from_base}
EXP_NAME=${EXP_NAME:-n1d7_sft_from_base}
MAX_STEPS=${MAX_STEPS:-4000}
SAVE_STEPS=${SAVE_STEPS:-500}
GLOBAL_BATCH=${GLOBAL_BATCH:-16}
GRAD_ACCUM=${GRAD_ACCUM:-2}
LR=${LR:-1e-4}
USE_WANDB=${USE_WANDB:-1}

export USE_TF=0
export TRANSFORMERS_NO_TF=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/hf_cache
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
if [[ "$USE_WANDB" == "1" ]]; then export WANDB_PROJECT=recap-n1d7-from-base; WANDB_FLAG=--use-wandb; else WANDB_FLAG=--no-use-wandb; fi

echo "[finetune] base=$BASE_MODEL"
echo "[finetune] data=$DATASET"
echo "[finetune] out=$OUTPUT_DIR  exp=$EXP_NAME"
echo "[finetune] max_steps=$MAX_STEPS save_steps=$SAVE_STEPS global_batch=$GLOBAL_BATCH grad_accum=$GRAD_ACCUM lr=$LR wandb=$USE_WANDB"

# NOTE: path must contain the substring "nvidia/Cosmos-Reason2" for the gr00t
# backbone-class selector (get_backbone_cls); we use a local symlink for that.
export COSMOS_PATH=${COSMOS_PATH:-/data/checkpoints/nvidia/Cosmos-Reason2-2B}
export HF_HUB_OFFLINE=1
cd "$REPO/vla-rlft-n1d7/gr00t"
exec "$PY" "$REPO/examples/recap/sft_from_base/launch_finetune_local.py" \
  --base-model-path "$BASE_MODEL" \
  --dataset-path "$DATASET" \
  --embodiment-tag libero_sim \
  --output-dir "$OUTPUT_DIR" \
  --experiment-name "$EXP_NAME" \
  --wandb-project recap-n1d7-from-base \
  --tune-projector \
  --tune-diffusion-model \
  --no-tune-llm \
  --no-tune-visual \
  --max-steps "$MAX_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --save-total-limit 20 \
  --global-batch-size "$GLOBAL_BATCH" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --learning-rate "$LR" \
  --num-gpus 1 \
  --dataloader-num-workers 4 \
  $WANDB_FLAG
