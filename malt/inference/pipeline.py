from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List, Sequence

import torch
from peft import PeftModel
from transformers import PreTrainedTokenizerBase

from malt.data import Gsm8kExample, extract_gsm8k_answer, normalize_gsm8k_answer
from malt.models.prompts import (
    build_generator_prompt,
    build_verifier_prompt,
    build_refiner_prompt,
)


@dataclass
class InferenceConfig:
    """
    Configuration for GSM8K inference.
    """

    max_new_tokens: int = 256
    temperature: float = 0.3
    top_p: float = 0.95
    top_k: int = 50

    # Number of independent trajectories per question for majority voting.
    num_samples: int = 3


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
            gen_ids[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        return gen_text.strip()


def run_single_agent_generator_gsm8k(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    questions: Sequence[Gsm8kExample],
    cfg: InferenceConfig,
) -> List[str]:
    """
    Single-agent baseline: use only the Generator-style prompt and model.

    For each question, sample `cfg.num_samples` generator outputs and return
    the majority-voted final answer (after extraction/normalization).
    """
    final_answers: List[str] = []

    for ex in questions:
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
            answers.append(extract_gsm8k_answer(gen_text))

        # Majority vote over normalized answers.
        norm_counts: Counter = Counter(
            normalize_gsm8k_answer(a) for a in answers
        )
        if not norm_counts:
            final_answers.append("")
            continue
        best_norm, _ = norm_counts.most_common(1)[0]
        # Return a representative raw answer that normalizes to best_norm.
        for a in answers:
            if normalize_gsm8k_answer(a) == best_norm:
                final_answers.append(a)
                break

    return final_answers


def run_multi_agent_malt_gsm8k(
    generator_model: PeftModel,
    verifier_model: PeftModel,
    refiner_model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    questions: Sequence[Gsm8kExample],
    cfg: InferenceConfig,
) -> List[str]:
    """
    Multi-agent MALT-style inference on GSM8K.

    For each question:
      - Sample a generator solution.
      - Feed it to the verifier and sample a critique.
      - Feed both into the refiner and sample a refined solution.
      - Repeat this sequential pipeline `cfg.num_samples` times and majority
        vote over the final refined answers.
    """
    final_answers: List[str] = []

    for ex in questions:
        answers: List[str] = []
        for _ in range(cfg.num_samples):
            # Generator
            g_prompt = build_generator_prompt(ex.question)
            g_text = _generate_single(
                model=generator_model,
                tokenizer=tokenizer,
                prompt=g_prompt,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
            )

            # Verifier
            v_prompt = build_verifier_prompt(ex.question, g_text)
            v_text = _generate_single(
                model=verifier_model,
                tokenizer=tokenizer,
                prompt=v_prompt,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
            )

            # Refiner
            r_prompt = build_refiner_prompt(ex.question, g_text, v_text)
            r_text = _generate_single(
                model=refiner_model,
                tokenizer=tokenizer,
                prompt=r_prompt,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
            )

            answers.append(extract_gsm8k_answer(r_text))

        norm_counts: Counter = Counter(
            normalize_gsm8k_answer(a) for a in answers
        )
        if not norm_counts:
            final_answers.append("")
            continue
        best_norm, _ = norm_counts.most_common(1)[0]
        for a in answers:
            if normalize_gsm8k_answer(a) == best_norm:
                final_answers.append(a)
                break

    return final_answers

