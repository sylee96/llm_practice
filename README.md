# llm_practice

GRPO-RLVR training for Qwen3-1.7B with Math + IFEval rewards on multi-GPU setup.

## Overview

- **Model**: `seopbo/qwen3-1.7b-sft-by-tulu3-subsets`
- **Dataset**: `allenai/Dolci-Instruct-RL` (Math + Precise Instruction Following)
- **Method**: GRPO with rule-based verifiers (RLVR)
- **Hardware**: RTX PRO 6000 Blackwell × 8 (1 vLLM rollout + 7 FSDP training)

## Project Structure

```
llm_practice/
├── pyproject.toml                          # Dependencies (uv sync)
├── run_grpo.sh                             # Training entry point
├── run_grpo.py                             # GRPO main script
├── reward_functions.py                     # Unified reward (Math + IFEval)
├── upload_checkpoints.sh                   # Auto-upload checkpoints to HF
├── configs/
│   └── accelerate_config.yaml              # 7-GPU FSDP config
├── assets/
│   └── qwen3_instruct_trl_no_think.jinja   # Qwen3 chat template
└── ifeval_verifier/                        # IFEval verifier (from open-instruct)
    ├── __init__.py                         # run_ifeval_verifier()
    ├── instructions.py                     # 57 checker classes
    ├── instructions_registry.py            # instruction_id → checker mapping
    └── instructions_util.py               # Utilities
```

## Setup

```bash
# 1. Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create virtual environment and install dependencies
cd llm_practice
uv sync
```

## Training

```bash
# 1. Set environment variables
export WANDB_API_KEY="your_wandb_api_key"
export HF_TOKEN="your_hf_write_token"
export HF_USERNAME="your_hf_username"

# 2. (Optional) Start background checkpoint uploader
bash upload_checkpoints.sh &

# 3. Run training
bash run_grpo.sh
```

### Key Hyperparameters

| Parameter | Value |
|---|---|
| beta | 0.04 |
| num_generations | 8 |
| max_completion_length | 2048 |
| per_device_train_batch_size | 8 |
| gradient_accumulation_steps | 4 |
| num_iterations (steps_per_generation) | 3 |
| learning_rate | 1e-5 |
| save_steps | 300 |
| num_train_epochs | 1 |

### Expected Steps

- Total samples: ~133.5K (Math ~96K + IFEval ~37.5K)
- Effective batch size: 8 × 7 GPUs × 4 grad_accum = 224
- With num_iterations=3: **~600 gradient steps** per epoch
- Checkpoints saved at: step 300, final

## Checkpoint Upload

`upload_checkpoints.sh` runs in the background and polls for new checkpoints every 60 seconds.
Uploads to HuggingFace as `{HF_USERNAME}/qwen3-1.7b-grpo-ckpt{step}`.

## Reward Functions

Two reward types dispatched by `dataset` column:

- **Math** (`dataset="math"`): Extracts answer from `\boxed{}`, Minerva format, or last number. Compares with ground truth using normalization + numeric equivalence.
- **IFEval** (`dataset="ifeval"`): Multi-constraint verification using 57 instruction checkers (ported from [open-instruct IFEvalG](https://github.com/allenai/open-instruct)). Returns mean score across all constraints.

## References

- [TRL GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [allenai/Dolci-Instruct-RL](https://huggingface.co/datasets/allenai/Dolci-Instruct-RL)
- [allenai/open-instruct](https://github.com/allenai/open-instruct)
