from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, List, Sequence, Union

import torch
from peft import PeftModel
from transformers import PreTrainedTokenizerBase

from malt.data import Gsm8kExample, extract_gsm8k_answer, normalize_gsm8k_answer
from malt.models.prompts import (
    build_generator_prompt,
    build_verifier_prompt,
    build_refiner_prompt,
)
from malt.data import SomadhanExample, extract_Somadhan_answer, normalize_Somadhan_answer


# Union type accepted everywhere a question object is expected.
AnyExample = Union[Gsm8kExample, SomadhanExample]


def _get_answer_fns(
    example: AnyExample,
) -> tuple[Callable[[str], str], Callable[[str], str]]:
    """
    Return the (extract, normalize) function pair appropriate for the example type.
    """
    if isinstance(example, SomadhanExample):
        return extract_Somadhan_answer, normalize_Somadhan_answer
    return extract_gsm8k_answer, normalize_gsm8k_answer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    """Configuration for inference over any supported dataset."""

    max_new_tokens: int = 256
    temperature: float = 0.3
    top_p: float = 0.95
    top_k: int = 50

    # Number of independent trajectories per question for majority voting.
    num_samples: int = 3


# ---------------------------------------------------------------------------
# Low-level generation helper
# ---------------------------------------------------------------------------

def _generate_single(
    model: PeftModel,
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
            truncation=True,
            padding=True,
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


# ---------------------------------------------------------------------------
# Shared majority-vote helper
# ---------------------------------------------------------------------------

def _majority_vote(
    answers: List[str],
    extract_fn: Callable[[str], str],
    normalize_fn: Callable[[str], str],
) -> str:
    """
    Given a list of raw answer strings, return the raw answer whose normalized
    form appears most often. Returns "" if the list is empty.
    """
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


# ---------------------------------------------------------------------------
# Single-agent inference
# ---------------------------------------------------------------------------

def run_single_agent_generator(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    questions: Sequence[AnyExample],
    cfg: InferenceConfig,
) -> List[str]:
    """
    Single-agent baseline: use only the Generator-style prompt and model.

    Accepts both Gsm8kExample and SomadhanExample questions. For each question,
    sample cfg.num_samples generator outputs and return the majority-voted
    final answer.
    """
    final_answers: List[str] = []

    for ex in questions:
        extract_fn, normalize_fn = _get_answer_fns(ex)
        answers: List[str] = []

        for _ in range(cfg.num_samples):
            prompt = build_generator_prompt(ex.question)
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
run_single_agent_generator_gsm8k = run_single_agent_generator


# ---------------------------------------------------------------------------
# Multi-agent MALT inference
# ---------------------------------------------------------------------------

def run_multi_agent_malt(
    generator_model: PeftModel,
    verifier_model: PeftModel,
    refiner_model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    questions: Sequence[AnyExample],
    cfg: InferenceConfig,
) -> List[str]:
    """
    Multi-agent MALT-style inference.

    Accepts both Gsm8kExample and SomadhanExample questions. For each question:
      1. Sample a generator solution.
      2. Feed it to the verifier and sample a critique.
      3. Feed both into the refiner and sample a refined solution.
      4. Repeat cfg.num_samples times and majority-vote over refined answers.
    """
    final_answers: List[str] = []

    for ex in questions:
        extract_fn, normalize_fn = _get_answer_fns(ex)
        answers: List[str] = []

        for _ in range(cfg.num_samples):
            # Generator
            g_text = _generate_single(
                model=generator_model,
                tokenizer=tokenizer,
                prompt=build_generator_prompt(ex.question),
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
            )

            # Verifier
            v_text = _generate_single(
                model=verifier_model,
                tokenizer=tokenizer,
                prompt=build_verifier_prompt(ex.question, g_text),
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
            )

            # Refiner
            r_text = _generate_single(
                model=refiner_model,
                tokenizer=tokenizer,
                prompt=build_refiner_prompt(ex.question, g_text, v_text),
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
            )

            answers.append(r_text)

        final_answers.append(_majority_vote(answers, extract_fn, normalize_fn))

    return final_answers


# Backwards-compatible alias for callers that used the GSM8K-specific name.
run_multi_agent_malt_gsm8k = run_multi_agent_malt