from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from malt.data import Gsm8kExample, extract_gsm8k_answer, normalize_gsm8k_answer


@dataclass
class QwenBaselineConfig:
    """
    Configuration for Qwen base-model baselines.
    """

    # E.g. "Qwen/Qwen2.5-1.5B" or "Qwen/Qwen2.5-3B"
    model_name: str = "Qwen/Qwen2.5-1.5B"
    device_map: str = "auto"

    max_new_tokens: int = 256
    temperature: float = 0.3
    top_p: float = 0.95
    top_k: int = 50

    num_samples: int = 1  # usually 1 for a simple zero-shot baseline


def load_qwen_model_and_tokenizer(
    cfg: QwenBaselineConfig,
):
    """
    Load a Qwen base model and tokenizer for zero-shot baselines.
    """
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
            gen_ids[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        return gen_text.strip()


def run_qwen_zero_shot_gsm8k(
    model,
    tokenizer: PreTrainedTokenizerBase,
    questions: Sequence[Gsm8kExample],
    cfg: QwenBaselineConfig,
) -> List[str]:
    """
    Zero-shot Qwen baseline on GSM8K without any fine-tuning or multi-agent
    structure.
    """
    final_answers: List[str] = []

    for ex in questions:
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
            answers.append(extract_gsm8k_answer(gen_text))

        # Majority vote over normalized answers (even if num_samples == 1).
        norm_counts = {}
        for a in answers:
            norm = normalize_gsm8k_answer(a)
            norm_counts[norm] = norm_counts.get(norm, 0) + 1
        if not norm_counts:
            final_answers.append("")
            continue
        best_norm = max(norm_counts.items(), key=lambda kv: kv[1])[0]
        for a in answers:
            if normalize_gsm8k_answer(a) == best_norm:
                final_answers.append(a)
                break

    return final_answers

