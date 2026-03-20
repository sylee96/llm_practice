"""
IFEval Verifier — standalone packaging of open-instruct's IFEvalG.

Usage:
    from ifeval_verifier import run_ifeval_verifier
    score = run_ifeval_verifier(prediction, label)
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any

from ifeval_verifier import instructions_registry

logger = logging.getLogger(__name__)

INSTRUCTION_DICT = instructions_registry.INSTRUCTION_DICT


def remove_thinking_section(text: str) -> str:
    """Strip <think>...</think> and <answer>...</answer> tags."""
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<answer>\s*", "", text)
    text = re.sub(r"\s*</answer>", "", text)
    return text.strip()


def run_ifeval_verifier(prediction: str, label: Any) -> float:
    """
    IFEval verification following open-instruct's IFEvalVerifier logic.

    Args:
        prediction: Model-generated response text.
        label: Ground-truth constraint, either a string repr of a list of dicts
               or already parsed. Expected format:
               [{"instruction_id": [...], "kwargs": [...]}]

    Returns:
        Score between 0.0 and 1.0 (mean of per-instruction pass rates).
    """
    # Parse label
    if isinstance(label, str):
        try:
            constraint_dict = ast.literal_eval(label)
        except Exception:
            try:
                constraint_dict = json.loads(label)
            except Exception:
                return 0.0
    elif isinstance(label, list):
        constraint_dict = label
    else:
        return 0.0

    # Take the first element (Dolci / open-instruct format)
    if isinstance(constraint_dict, list) and len(constraint_dict) > 0:
        constraint_dict = constraint_dict[0]
    else:
        return 0.0

    if isinstance(constraint_dict, str):
        try:
            constraint_dict = json.loads(constraint_dict)
        except Exception:
            return 0.0

    answer = remove_thinking_section(prediction)

    if not prediction.strip() or not answer.strip():
        return 0.0

    instruction_keys = constraint_dict.get("instruction_id", [])
    args_list = constraint_dict.get("kwargs", [])

    if not instruction_keys:
        return 0.0

    rewards = []
    for instruction_key, args in zip(instruction_keys, args_list):
        if args is None:
            args = {}
        args = {k: v for k, v in args.items() if v is not None}

        instruction_cls = INSTRUCTION_DICT.get(instruction_key)
        if instruction_cls is None:
            logger.warning("Unknown instruction_id: %s", instruction_key)
            rewards.append(0.0)
            continue

        try:
            instruction_instance = instruction_cls(instruction_key)
            instruction_instance.build_description(**args)
            if instruction_instance.check_following(answer):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except Exception as e:
            logger.warning("IFEval verifier error for %s: %s", instruction_key, e)
            rewards.append(0.0)

    return sum(rewards) / len(rewards) if rewards else 0.0
