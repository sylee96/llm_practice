#!/usr/bin/env bash
set -euo pipefail

# GRPO-RLVR: Qwen3-1.7B + Dolci (Math + IFEval) on RTX PRO 6000 Blackwell × 8
# GPU layout: cuda:6,7 = vLLM rollout (TP=2), cuda:0~5 = DDP training

MODEL_NAME="seopbo/qwen3-1.7b-sft-by-tulu3-subsets"
HOSTNAME=$(hostname -f)

# ── Environment ──
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── wandb (set WANDB_API_KEY before running) ──
export WANDB_PROJECT="qwen3-grpo"

# ── NLTK data for IFEval verifier ──
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" 2>/dev/null || true

# ── Step 1: Start vLLM server on GPU 6,7 (tensor parallel=2) via TRL CLI ──
echo "Starting vLLM server on GPU 6,7 (TP=2)..."
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 python -m trl.cli vllm-serve \
    --model "$MODEL_NAME" \
    --port 8000 \
    --max-model-len 4096 \
    --gpu_memory_utilization 0.9 \
    --tensor-parallel-size 2 &
VLLM_PID=$!

# ── Step 2: Wait for vLLM server to be ready ──
echo "Waiting for vLLM server..."
while ! curl -s "http://${HOSTNAME}:8000/health/" > /dev/null 2>&1; do
    sleep 3
done
echo "vLLM server is ready!"

# ── Step 3: Launch training on GPU 0~5 ──
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
accelerate launch --num_processes=6 --gpu_ids="0,1,2,3,4,5" \
    run_grpo.py \
    --vllm_server_host "$HOSTNAME" \
    --vllm_server_port 8000

# ── Step 4: Cleanup vLLM server ──
kill -- -$(ps -o pgid= -p $VLLM_PID | tr -d ' ') 2>/dev/null || true

echo "Training finished!"

# ── Upload final checkpoint to HuggingFace (optional) ──
if [ -n "${HF_TOKEN:-}" ] && [ -n "${HF_USERNAME:-}" ]; then
    FINAL_DIR="./qwen3-1.7b-grpo-rlvr-dolci"
    if [ -d "$FINAL_DIR" ]; then
        echo "Uploading final model to HuggingFace..."
        huggingface-cli upload "${HF_USERNAME}/qwen3-1.7b-grpo-final" "$FINAL_DIR" --token "$HF_TOKEN"
    fi
fi