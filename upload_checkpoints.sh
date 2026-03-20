#!/usr/bin/env bash
set -euo pipefail

# Background checkpoint uploader for HuggingFace.
# Monitors the output directory and uploads new checkpoints as they appear.
#
# Usage:
#   HF_TOKEN=xxx HF_USERNAME=seopbo bash upload_checkpoints.sh &
#
# Required env vars:
#   HF_TOKEN     - HuggingFace token with write permission
#   HF_USERNAME  - HuggingFace username (e.g., "seopbo")

OUTPUT_DIR="${OUTPUT_DIR:-./qwen3-1.7b-grpo-rlvr-dolci}"
POLL_INTERVAL=60  # seconds between checks
UPLOADED_FILE="${OUTPUT_DIR}/.uploaded_checkpoints"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set. Export it before running."
    exit 1
fi

if [ -z "${HF_USERNAME:-}" ]; then
    echo "ERROR: HF_USERNAME is not set. Export it before running."
    exit 1
fi

touch "$UPLOADED_FILE"

echo "[upload_checkpoints] Monitoring ${OUTPUT_DIR} for new checkpoints..."
echo "[upload_checkpoints] Will upload to ${HF_USERNAME}/qwen3-1.7b-grpo-ckpt{step}"

while true; do
    # Find checkpoint directories (checkpoint-300, checkpoint-600, etc.)
    for ckpt_dir in "${OUTPUT_DIR}"/checkpoint-*; do
        [ -d "$ckpt_dir" ] || continue

        ckpt_name=$(basename "$ckpt_dir")
        step="${ckpt_name#checkpoint-}"

        # Skip if already uploaded
        if grep -qx "$ckpt_name" "$UPLOADED_FILE" 2>/dev/null; then
            continue
        fi

        # Check if checkpoint is complete (has config.json or model.safetensors)
        if [ ! -f "$ckpt_dir/config.json" ] && [ ! -f "$ckpt_dir/model.safetensors" ]; then
            continue
        fi

        echo "[upload_checkpoints] Uploading ${ckpt_name} (step ${step})..."
        REPO_NAME="${HF_USERNAME}/qwen3-1.7b-grpo-ckpt${step}"

        if huggingface-cli upload "$REPO_NAME" "$ckpt_dir" --token "$HF_TOKEN" 2>&1; then
            echo "$ckpt_name" >> "$UPLOADED_FILE"
            echo "[upload_checkpoints] Successfully uploaded ${ckpt_name} → ${REPO_NAME}"
        else
            echo "[upload_checkpoints] Failed to upload ${ckpt_name}, will retry next cycle"
        fi
    done

    sleep "$POLL_INTERVAL"
done
