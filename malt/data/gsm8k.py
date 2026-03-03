from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Sequence

import re

from datasets import load_dataset, Dataset


_GSM8K_HF_NAME = "gsm8k"
_GSM8K_HF_CONFIG = "main"


@dataclass
class Gsm8kExample:
    """
    Canonical representation of a GSM8K example.

    Attributes:
        id: String identifier for the example.
        question: Problem statement.
        answer_raw: Original solution text from the dataset.
        answer_target: Canonicalized final answer string (for evaluation).
    """

    id: str
    question: str
    answer_raw: str
    answer_target: str


def _hf_split_name(split: Literal["train", "test", "validation"]) -> str:
    """
    Map our logical split names to the huggingface dataset split names.
    GSM8K only has 'train' and 'test'; we alias 'validation' to 'test'
    for convenience.
    """
    if split in ("train", "test"):
        return split
    if split == "validation":
        return "test"
    raise ValueError(f"Unsupported split: {split}")


def load_gsm8k_hf_split(
    split: Literal["train", "test", "validation"] = "train",
) -> Dataset:
    """
    Load a raw GSM8K split from Hugging Face Datasets.
    """
    hf_split = _hf_split_name(split)
    ds = load_dataset(_GSM8K_HF_NAME, _GSM8K_HF_CONFIG, split=hf_split)
    return ds


def extract_gsm8k_answer(answer_text: str) -> str:
    """
    Extract the final answer from a GSM8K solution string.

    GSM8K answers are usually of the form:
        "... reasoning ... #### 42"

    We first look for the '####' marker; if absent, we fall back to the
    last numeric token in the string. The result is returned as a
    stripped string, suitable for exact-match comparison (possibly after
    numeric normalization).
    """
    # Prefer the official marker.
    if "####" in answer_text:
        final = answer_text.split("####", maxsplit=1)[-1]
        final = final.strip()
        return final

    # Fallback: last number in the string (integer or decimal).
    numeric_matches = re.findall(r"-?\d+(?:\.\d+)?", answer_text)
    if numeric_matches:
        return numeric_matches[-1].strip()

    # As a last resort, return the stripped text.
    return answer_text.strip()


def normalize_gsm8k_answer(text: str) -> str:
    """
    Normalize a GSM8K answer string for comparison.

    Strategy:
    - Try to interpret as a float; if successful, normalize to a compact
      string representation (no trailing .0 unless necessary).
    - Otherwise, lower-case and strip surrounding whitespace.
    """
    text = text.strip()
    # Remove thousands separators (simple heuristic).
    cleaned = text.replace(",", "")
    try:
        value = float(cleaned)
        # Represent integers without decimal part, otherwise keep as is.
        if value.is_integer():
            return str(int(value))
        # Avoid scientific notation for typical GSM8K answers.
        return str(value)
    except ValueError:
        # Non-numeric — compare as normalized text.
        return cleaned.lower()


def gsm8k_exact_match(predicted: str, target: str) -> bool:
    """
    Exact-match comparator for GSM8K answers.

    Both predicted and target strings are passed through `extract_gsm8k_answer`
    (if they contain reasoning traces) and then normalized numerically/textually.
    """
    pred_final = normalize_gsm8k_answer(extract_gsm8k_answer(predicted))
    target_final = normalize_gsm8k_answer(extract_gsm8k_answer(target))
    return pred_final == target_final


def build_gsm8k_examples(
    ds: Dataset,
) -> List[Gsm8kExample]:
    """
    Convert a raw HF GSM8K Dataset into a list of Gsm8kExample objects.
    """
    examples: List[Gsm8kExample] = []
    for idx, row in enumerate(ds):
        question = row.get("question", "")
        answer_raw = row.get("answer", "")
        answer_target = extract_gsm8k_answer(answer_raw)
        examples.append(
            Gsm8kExample(
                id=str(row.get("id", idx)),
                question=question,
                answer_raw=answer_raw,
                answer_target=answer_target,
            )
        )
    return examples


def load_gsm8k_split(
    split: Literal["train", "test", "validation"] = "train",
) -> List[Gsm8kExample]:
    """
    High-level helper: load a GSM8K split and return canonical examples.
    """
    ds = load_gsm8k_hf_split(split)
    return build_gsm8k_examples(ds)

