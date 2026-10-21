#!/usr/bin/env bash
# Serve a BDS pi0.5 checkpoint trained with run_bds_sft.sh over websocket (default port 8000).
# Same protocol as openpi scripts/serve_bds.sh, so the same openpi_client works.
# All caches are kept on the /mnt mount (nothing in $HOME).
#
# Usage: [CUDA_VISIBLE_DEVICES=<gpu>] bash examples/sft/serve_bds.sh <checkpoint_dir> [port]
#   e.g. bash examples/sft/serve_bds.sh \
#     /mnt/data/sftp/data/vla/vr_checkpoints/rlinf/<exp_name>/<exp_name>/checkpoints/global_step_33000
# Exported/HF checkpoint (model.safetensors + bds/<asset>/norm_stats.json): MODEL=<dir> bash $0 <dir>
# Optional env: NUM_STEPS=<denoise steps> (default 10), COMPILE=1 (torch.compile; first request takes minutes).
set -euo pipefail

CKPT_DIR=${1:?"usage: $0 <checkpoint_dir> [port]"}
PORT=${2:-8000}
OPENPI_CONFIG=${OPENPI_CONFIG:-pi05_bds_vfe_sim_pick_left}

PRETRAINED=/mnt/data/sftp/data/vla/vr_pretrained_checkpoints
# Base model dir used for training; norm stats live in $MODEL/bds/vfe_sim_pick_success_0710.
MODEL=${MODEL:-$PRETRAINED/pi05_base_pytorch}
export OPENPI_DATA_HOME=$PRETRAINED/openpi # paligemma tokenizer
export HF_HOME=$PRETRAINED/huggingface
export XDG_CACHE_HOME=$PRETRAINED/xdg_cache

REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH=${REPO_PATH}:${PYTHONPATH:-}
cd "$REPO_PATH"

PYTHON=${PYTHON:-$REPO_PATH/.venv/bin/python}
"$PYTHON" examples/sft/serve_policy.py --port "$PORT" \
    --checkpoint "$CKPT_DIR" --model-path "$MODEL" --config-name "$OPENPI_CONFIG" \
    ${NUM_STEPS:+--num-steps "$NUM_STEPS"} ${COMPILE:+--compile}
