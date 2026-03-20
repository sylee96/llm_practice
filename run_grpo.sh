#!/usr/bin/env bash
set -euo pipefail

# GRPO-RLVR: Qwen3-1.7B + Dolci (Math + IFEval) on RTX PRO 6000 Blackwell × 8
# GPU layout: cuda:0~6 = training (FSDP), cuda:7 = vLLM rollout

# ── Environment ──
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── wandb (set WANDB_API_KEY before running) ──
export WANDB_PROJECT="qwen3-grpo"

# ── HuggingFace (set HF_TOKEN before running for checkpoint upload) ──
# export HF_TOKEN="your_hf_token_here"

# ── NLTK data for IFEval verifier ──
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" 2>/dev/null || true

# ── Launch training ──
accelerate launch \
    --config_file configs/accelerate_config.yaml \
    run_grpo.py

echo "Training finished!"

# ── Upload final checkpoint to HuggingFace (optional) ──
if [ -n "${HF_TOKEN:-}" ] && [ -n "${HF_USERNAME:-}" ]; then
    FINAL_DIR="./qwen3-1.7b-grpo-rlvr-dolci"
    if [ -d "$FINAL_DIR" ]; then
        echo "Uploading final model to HuggingFace..."
        huggingface-cli upload "${HF_USERNAME}/qwen3-1.7b-grpo-final" "$FINAL_DIR" --token "$HF_TOKEN"
    fi
fi
