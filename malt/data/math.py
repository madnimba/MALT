from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import re

from datasets import load_dataset, Dataset


_MATH_HF_NAME = "hendrycks/competition_math"
_MATH_HF_CONFIG = "all"  # can be adjusted if needed


@dataclass
class MathExample:
    """
    Canonical representation of a MATH example.
    """

    id: str
    question: str
    solution_raw: str
    answer_target: str


def load_math_hf_split(
    split: Literal["train", "test"] = "train",
) -> Dataset:
    """
    Load a raw MATH split from Hugging Face Datasets.

    The exact dataset/config may need to be adjusted depending on the
    local environment; this is a reasonable default for the Hendrycks
    competition math dataset.
    """
    ds = load_dataset(_MATH_HF_NAME, _MATH_HF_CONFIG, split=split)
    return ds


def extract_math_answer(solution_text: str) -> str:
    """
    Extract the final answer from a MATH solution string.

    Heuristics:
      1. Prefer the content of the last '\\boxed{...}'.
      2. Otherwise, use the last number in the string (integer or decimal).
      3. Fallback: stripped solution text.
    """
    # Look for \boxed{...}
    boxed_matches = re.findall(r"\\boxed\{([^}]+)\}", solution_text)
    if boxed_matches:
        return boxed_matches[-1].strip()

    # Fallback: last numeric token.
    numeric_matches = re.findall(r"-?\d+(?:\.\d+)?", solution_text)
    if numeric_matches:
        return numeric_matches[-1].strip()

    return solution_text.strip()


def normalize_math_answer(text: str) -> str:
    """
    Normalize MATH answer strings for comparison.

    Similar strategy to GSM8K:
      - Try to parse as float and canonicalize.
      - Otherwise, lower-case and strip whitespace.
    """
    text = text.strip()
    cleaned = text.replace(",", "")
    try:
        value = float(cleaned)
        if value.is_integer():
            return str(int(value))
        return str(value)
    except ValueError:
        return cleaned.lower()


def math_exact_match(predicted: str, target: str) -> bool:
    """
    Exact-match comparator for MATH answers using normalization.
    """
    pred_final = normalize_math_answer(extract_math_answer(predicted))
    target_final = normalize_math_answer(extract_math_answer(target))
    return pred_final == target_final


def build_math_examples(ds: Dataset) -> List[MathExample]:
    """
    Convert a raw HF MATH Dataset into a list of MathExample objects.
    """
    examples: List[MathExample] = []
    for idx, row in enumerate(ds):
        problem = row.get("problem", "")
        solution = row.get("solution", "")
        answer_target = extract_math_answer(solution)
        examples.append(
            MathExample(
                id=str(row.get("id", idx)),
                question=problem,
                solution_raw=solution,
                answer_target=answer_target,
            )
        )
    return examples


def load_math_split(
    split: Literal["train", "test"] = "train",
) -> List[MathExample]:
    """
    High-level helper: load a MATH split and return canonical examples.
    """
    ds = load_math_hf_split(split)
    return build_math_examples(ds)

