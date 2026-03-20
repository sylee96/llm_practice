"""
Unified reward functions for GRPO-RLVR training.
Handles both Math and IFEval (Precise Instruction Following) tasks.

Dispatches based on the `dataset` column:
  - "math"   → MathVerifier (boxed extraction + normalize + equivalence)
  - "ifeval" → IFEvalVerifier (instruction_id-based multi-constraint checking)
"""

from __future__ import annotations

import re
from typing import Any, List

from ifeval_verifier import run_ifeval_verifier


# ===================== Math Verifier (from ground_truth_utils_math.py) =====================

def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _extract_label_text(label: Any) -> str:
    if isinstance(label, dict):
        for k in ["answer", "ground_truth", "target", "label", "solution"]:
            if k in label:
                return _safe_str(label[k])
    return _safe_str(label)


def last_boxed_only_string(text: str) -> str | None:
    starts = [m.start() for m in re.finditer(r"\\boxed\{", text)]
    if not starts:
        return None
    start = starts[-1]
    i = start + len(r"\boxed{")
    depth = 1
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
        i += 1
    return None


def remove_boxed(boxed: str) -> str:
    if not boxed.startswith(r"\boxed{") or not boxed.endswith("}"):
        return boxed
    return boxed[len(r"\boxed{") : -1]


def get_unnormalized_answer(text: str) -> str:
    m = re.search(r"(?is)(?:final answer|answer)\s*[:：]\s*(.+)$", text)
    if m:
        return m.group(1).strip()
    return text.strip()


def normalize_final_answer(text: str) -> str:
    s = _safe_str(text)
    s = s.replace("$", "")
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\!", "")
    s = s.replace(",", "")
    s = re.sub(r"\s+", "", s)
    s = s.lower()
    return s


def _to_float_if_possible(s: str) -> float | None:
    s = s.strip()
    if not s:
        return None
    if re.fullmatch(r"[-+]?\d+/\d+", s):
        num, den = s.split("/")
        try:
            return float(num) / float(den)
        except Exception:
            return None
    try:
        return float(s)
    except Exception:
        return None


def is_equiv(a: str, b: str) -> bool:
    na = normalize_final_answer(a)
    nb = normalize_final_answer(b)
    if na == nb:
        return True
    fa = _to_float_if_possible(na)
    fb = _to_float_if_possible(nb)
    if fa is not None and fb is not None:
        return abs(fa - fb) <= 1e-6
    return False


def math_verify(prediction: str, label: Any) -> float:
    """Math verification: extract answer from prediction, compare with ground truth."""
    gt = _extract_label_text(label)
    all_answers: list[str] = []

    boxed_answer = last_boxed_only_string(prediction)
    if boxed_answer is not None:
        boxed_answer = remove_boxed(boxed_answer)
        if boxed_answer:
            all_answers.append(boxed_answer)

    minerva_answer = normalize_final_answer(get_unnormalized_answer(prediction))
    if minerva_answer and minerva_answer != "[invalidanswer]":
        all_answers.append(minerva_answer)

    if not all_answers:
        dollars = [m.start() for m in re.finditer(r"\$", prediction)]
        if len(dollars) > 1:
            answer = normalize_final_answer(prediction[dollars[-2] + 1 : dollars[-1]])
            all_answers.append(answer)

    if not all_answers:
        all_answers.append(normalize_final_answer(prediction))

    all_answers.append(prediction)

    for answer in all_answers:
        if is_equiv(answer, gt):
            return 1.0
    return 0.0


# ===================== Think Block Stripping =====================

def strip_think_block(text: str) -> str:
    """Remove <think>...</think> blocks (Qwen3 thinking mode)."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


# ===================== Unified Reward Function =====================

_reward_call_count = 0
ROLLOUT_LOG_EVERY = 50


def reward_fn(
    completions,
    dataset: List[str],
    ground_truth: List[Any],
    **kwargs,
) -> List[float]:
    """
    Unified reward function for GRPOTrainer.
    Dispatches to math or ifeval verifier based on `dataset` column.

    Args:
        completions: List of model completions (str or list[dict]).
        dataset: List of dataset type strings ("math" or "ifeval").
        ground_truth: List of ground truth labels.

    Returns:
        List of reward scores (0.0 to 1.0).
    """
    global _reward_call_count
    _reward_call_count += 1
    should_log = (_reward_call_count % ROLLOUT_LOG_EVERY == 1)

    rewards: List[float] = []
    for i, (completion, ds, gt) in enumerate(zip(completions, dataset, ground_truth)):
        # Extract text from completion
        if isinstance(completion, list):
            text = completion[-1]["content"] if completion else ""
        else:
            text = str(completion)

        text = strip_think_block(text)

        # Dispatch based on dataset type
        if ds == "math":
            r = math_verify(text, gt)
        elif ds == "ifeval":
            r = run_ifeval_verifier(text, gt)
        else:
            r = 0.0

        # Rollout logging
        if should_log and i < 4:
            print(f"\n{'='*70}")
            print(f"[ROLLOUT #{_reward_call_count} | sample {i}]")
            print(f"  dataset    : {ds}")
            print(f"  reward     : {r}")
            print(f"  gt         : {str(gt)[:200]}")
            print(f"  completion : {text[:300]}{'...' if len(text) > 300 else ''}")
            print(f"{'='*70}")

        rewards.append(float(r))
    return rewards
