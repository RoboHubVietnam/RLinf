#!/usr/bin/env bash
# Finetune pi0.5 (LoRA) on the BDS VR_H5D sim pick dataset with RLinf SFT.
# Mirrors openpi scripts/train_bds.sh: all caches, pretrained weights, wandb files
# and checkpoints are kept on the /mnt mount (nothing in $HOME).
#
# Usage: [CONFIG=<config_name>] [DATA=<path>] [DATA2=<path>|DATA2=] \
#        [OPENPI_CONFIG=<name>] [ASSET_ID=<asset_id>] [CUDA_VISIBLE_DEVICES=<gpu>] \
#        bash examples/sft/run_bds_sft.sh <exp_name> [extra hydra overrides...]
# Results (checkpoints, wandb, tensorboard): $VLA_ROOT/vr_checkpoints/rlinf/<exp_name>
set -euo pipefail

CONFIG=${CONFIG:-bds_sft_openpi_pi05}
EXP_NAME=${1:?"usage: $0 <exp_name> [extra overrides]"}
shift

VLA_ROOT=/mnt/data/sftp/data/vla
PRETRAINED=$VLA_ROOT/vr_pretrained_checkpoints
RESULTS=$VLA_ROOT/vr_checkpoints/rlinf

export OPENPI_DATA_HOME=$PRETRAINED/openpi           # paligemma tokenizer
export HF_HOME=$PRETRAINED/huggingface
export HF_LEROBOT_HOME=$PRETRAINED/lerobot
export XDG_CACHE_HOME=$PRETRAINED/xdg_cache
export WANDB_DIR=$VLA_ROOT/vr_checkpoints/wandb
mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$HF_LEROBOT_HOME" "$XDG_CACHE_HOME" "$WANDB_DIR" "$RESULTS"

EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EMBODIED_PATH
export REPO_PATH=$(dirname "$(dirname "$EMBODIED_PATH")")
export PYTHONPATH=${REPO_PATH}:${PYTHONPATH:-}
cd "$REPO_PATH"

PYTHON=${PYTHON:-$REPO_PATH/.venv/bin/python}
DATA=${DATA:-/mnt/data/sftp/data/vla/data_sim_ac/20261012_VR_H5D_VFE_sim_teleop_pick_success}
DATA2=${DATA2-/mnt/data/sftp/data/vla/data_sim_ac/20261006_VR_H5D_VFE_sim_teleop_pick_success_openpi}
# PyTorch pi0.5 base (converted from $PRETRAINED/openpi/openpi-assets/checkpoints/pi05_base).
MODEL=${MODEL:-$PRETRAINED/pi05_base_pytorch}
OPENPI_CONFIG=${OPENPI_CONFIG:-pi05_bds_vfe_sim_pick_left}
ASSET_ID=${ASSET_ID:-bds/vfe_sim_pick_success_0710}
ASSET_DIR=$MODEL/$ASSET_ID
# Empty DATA2 trains on DATA only.
DATA_PATHS=$DATA
EXTRA_REPO_ARGS=()
if [ -n "$DATA2" ]; then
    DATA_PATHS="$DATA,$DATA2"
    EXTRA_REPO_ARGS=(--extra-repo-ids "$DATA2")
fi

if [ ! -f "$MODEL/model.safetensors" ]; then
    echo "PyTorch base checkpoint not found at $MODEL (expected model.safetensors)." >&2
    echo "Convert it first, e.g. with the openpi venv:" >&2
    echo "  openpi/.venv/bin/python openpi/examples/convert_jax_model_to_pytorch.py \\" >&2
    echo "    --checkpoint-dir $PRETRAINED/openpi/openpi-assets/checkpoints/pi05_base \\" >&2
    echo "    --config-name pi05_libero --output-path $MODEL" >&2
    exit 1
fi

# Compute normalization stats once on the training data (stride must match training).
if [ ! -f "$ASSET_DIR/norm_stats.json" ]; then
    "$PYTHON" toolkits/lerobot/calculate_norm_stats.py \
        --config-name "$OPENPI_CONFIG" --repo-id "$DATA" \
        --action-stride 2 "${EXTRA_REPO_ARGS[@]}" --output-dir "$ASSET_DIR"
fi

"$PYTHON" examples/sft/train_vla_sft.py \
    --config-path "$EMBODIED_PATH/config/" --config-name "$CONFIG" \
    runner.logger.log_path="$RESULTS/$EXP_NAME" \
    runner.logger.experiment_name="$EXP_NAME" \
    "data.train_data_paths=[$DATA_PATHS]" \
    actor.model.model_path="$MODEL" \
    actor.model.openpi.config_name="$OPENPI_CONFIG" \
    "$@"
