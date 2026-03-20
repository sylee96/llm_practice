"""
GRPO-RLVR: Qwen3-1.7B + Math/IFEval on RTX PRO 6000 Blackwell × 8

Model:   seopbo/qwen3-1.7b-sft-by-tulu3-subsets
Dataset: allenai/Dolci-Instruct-RL (Math + IFEval subsets)
Method:  GRPO with rule-based verifiers as reward (RLVR)

GPU layout:
  - cuda:0~6  → 7 GPUs for FSDP training
  - cuda:7    → 1 GPU for vLLM rollout server

Hyperparameters:
  beta=0.04, num_generations=8, max_completion_length=2048
  per_device_train_batch_size=8, gradient_accumulation_steps=4
  num_iterations=3 (steps_per_generation)
  save_steps=300

Usage:
  bash run_grpo.sh
"""

import ast
import json
import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from reward_functions import reward_fn

# ===================== Config =====================
MODEL_ID = "seopbo/qwen3-1.7b-sft-by-tulu3-subsets"
DATASET_ID = "allenai/Dolci-Instruct-RL"
OUTPUT_DIR = "./qwen3-1.7b-grpo-rlvr-dolci"
CHAT_TEMPLATE_PATH = "./assets/qwen3_instruct_trl_no_think.jinja"

wandb.init(project="qwen3-grpo", name="qwen3-1.7b-grpo-rlvr-dolci-8gpu")


# ===================== Dataset =====================
def prepare_dataset():
    """Load Dolci-Instruct-RL, filter Math + IFEval, format for GRPOTrainer."""
    ds = load_dataset(DATASET_ID, split="train")

    # Filter: only math and ifeval
    ds = ds.filter(lambda x: x["dataset"] in ("math", "ifeval"))
    print(f"Filtered dataset: {len(ds)} samples")

    # Convert prompt field: Dolci uses "prompt" as a string, but also has "source_prompt"
    # GRPOTrainer expects "prompt" as list[dict] (conversational format)
    def format_example(example):
        # Use source_prompt if available (list[dict] format), otherwise wrap prompt string
        source_prompt = example.get("source_prompt")
        if source_prompt and isinstance(source_prompt, list) and len(source_prompt) > 0:
            prompt = source_prompt
        else:
            prompt = [{"role": "user", "content": example["prompt"]}]
        return {
            "prompt": prompt,
            "dataset": example["dataset"],
            "ground_truth": example["ground_truth"],
        }

    ds = ds.map(format_example, remove_columns=ds.column_names)
    ds = ds.shuffle(seed=42)

    print(f"Final dataset: {len(ds)} samples, columns: {ds.column_names}")
    return ds


# ===================== Main =====================
def main():
    # 1. Dataset
    ds = prepare_dataset()

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    chat_template = open(CHAT_TEMPLATE_PATH).read()
    tokenizer.chat_template = chat_template
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    tokenizer.padding_side = "left"

    # 3. Model (no flash_attention — Blackwell compatibility)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # 4. GRPO Config — RTX PRO 6000 Blackwell × 8
    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        # ── Multi-GPU: 7 train + 1 vLLM ──
        use_vllm=True,
        vllm_device="cuda:7",
        vllm_gpu_memory_utilization=0.9,
        # ── Batch & Memory ──
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        bf16=True,
        # ── GRPO core ──
        num_generations=8,
        max_completion_length=2048,
        beta=0.04,
        num_iterations=3,
        # ── Generation ──
        temperature=1.0,
        top_p=0.95,
        # ── Training hyperparams ──
        learning_rate=1e-5,
        num_train_epochs=1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        # ── Logging & Saving ──
        logging_steps=10,
        save_strategy="steps",
        save_steps=300,
        save_total_limit=3,
        save_only_model=True,
        report_to="wandb",
        # ── Misc ──
        seed=42,
        torch_empty_cache_steps=10,
    )

    # 5. Trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Training complete! Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
