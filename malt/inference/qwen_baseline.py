from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, List, Sequence, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from malt.data import (
    Gsm8kExample,
    extract_gsm8k_answer,
    normalize_gsm8k_answer,
    SomadhanExample,
    extract_Somadhan_answer,
    normalize_Somadhan_answer,
)


# Union type accepted everywhere a question object is expected.
AnyExample = Union[Gsm8kExample, SomadhanExample]


def _get_answer_fns(
    example: AnyExample,
) -> tuple[Callable[[str], str], Callable[[str], str]]:
    """Return the (extract, normalize) pair appropriate for the example type."""
    if isinstance(example, SomadhanExample):
        return extract_Somadhan_answer, normalize_Somadhan_answer
    return extract_gsm8k_answer, normalize_gsm8k_answer


@dataclass
class QwenBaselineConfig:
    """Configuration for Qwen base-model baselines."""

    # E.g. "Qwen/Qwen2.5-1.5B" or "Qwen/Qwen2.5-3B"
    model_name: str = "Qwen/Qwen2.5-1.5B"
    device_map: str = "auto"

    max_new_tokens: int = 256
    temperature: float = 0.3
    top_p: float = 0.95
    top_k: int = 50

    # Usually 1 for a simple zero-shot baseline; raise for majority voting.
    num_samples: int = 1


def load_qwen_model_and_tokenizer(cfg: QwenBaselineConfig):
    """Load a Qwen base model and tokenizer for zero-shot baselines."""
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        device_map=cfg.device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def _simple_zero_shot_prompt(question: str) -> str:
    return (
        "You are a helpful assistant that solves math word problems.\n"
        "Solve the problem step by step and clearly state the final answer.\n\n"
        f"Problem:\n{question}\n"
    )


def _generate_single(
    model,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen_text = tokenizer.decode(
            gen_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return gen_text.strip()


def _majority_vote(
    answers: List[str],
    extract_fn: Callable[[str], str],
    normalize_fn: Callable[[str], str],
) -> str:
    """Return the raw answer whose normalized form appears most often."""
    if not answers:
        return ""
    norm_counts: Counter = Counter(
        normalize_fn(extract_fn(a)) for a in answers
    )
    best_norm, _ = norm_counts.most_common(1)[0]
    for a in answers:
        if normalize_fn(extract_fn(a)) == best_norm:
            return a
    return ""


def run_qwen_zero_shot(
    model,
    tokenizer: PreTrainedTokenizerBase,
    questions: Sequence[AnyExample],
    cfg: QwenBaselineConfig,
) -> List[str]:
    """
    Zero-shot Qwen baseline without any fine-tuning or multi-agent structure.

    Accepts both Gsm8kExample and SomadhanExample questions. For each question,
    generates cfg.num_samples completions and returns the majority-voted answer.
    """
    final_answers: List[str] = []

    for ex in questions:
        extract_fn, normalize_fn = _get_answer_fns(ex)
        answers: List[str] = []

        for _ in range(cfg.num_samples):
            prompt = _simple_zero_shot_prompt(ex.question)
            gen_text = _generate_single(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
            )
            answers.append(gen_text)

        final_answers.append(_majority_vote(answers, extract_fn, normalize_fn))

    return final_answers


# Backwards-compatible alias for callers that used the GSM8K-specific name.
run_qwen_zero_shot_gsm8k = run_qwen_zero_shot