from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class SomadhanExample:
    """
    Canonical representation of a Somadhan example.

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


def extract_Somadhan_answer(answer_text: str) -> str:
    """
    Extract the final answer from a Somadhan solution string.

    Somadhan answers are usually of the form:
        "... reasoning ... #### 42"

    We first look for the '####' marker; if absent, we fall back to the
    last numeric token in the string.
    """
    if "####" in answer_text:
        final = answer_text.split("####", maxsplit=1)[-1]
        return final.strip()

    numeric_matches = re.findall(r"-?\d+(?:\.\d+)?", answer_text)
    if numeric_matches:
        return numeric_matches[-1].strip()

    return answer_text.strip()


def normalize_Somadhan_answer(text: str) -> str:
    """
    Normalize a Somadhan answer string for comparison.

    - Numeric strings are normalized to compact form (no trailing .0).
    - Non-numeric strings are lowercased and stripped.
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


def Somadhan_exact_match(predicted: str, target: str) -> bool:
    """
    Exact-match comparator for Somadhan answers.

    Both strings are passed through extract + normalize before comparison.
    """
    pred_final = normalize_Somadhan_answer(extract_Somadhan_answer(predicted))
    target_final = normalize_Somadhan_answer(extract_Somadhan_answer(target))
    return pred_final == target_final


def load_Somadhan_split(csv_path: str | Path) -> List[SomadhanExample]:
    """
    Load the full Somadhan dataset from a CSV file.

    The CSV is expected to have at minimum the columns:
        - question
        - answer

    An optional 'id' column is used if present; otherwise the row index is
    used as the id.

    All rows are returned — no train/test split is applied.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    examples: List[SomadhanExample] = []

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"CSV file appears to be empty: {csv_path}")

        missing = {"question", "answer"} - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}. "
                f"Found columns: {list(reader.fieldnames)}"
            )

        for idx, row in enumerate(reader):
            question = row.get("question", "").strip()
            answer_raw = row.get("answer", "").strip()
            answer_target = extract_Somadhan_answer(answer_raw)
            examples.append(
                SomadhanExample(
                    id=str(row.get("id", idx)),
                    question=question,
                    answer_raw=answer_raw,
                    answer_target=answer_target,
                )
            )

    return examples