from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from malt.data import gsm8k_exact_match, math_exact_match, Somadhan_exact_match


@dataclass
class EvalStats:
    total: int
    correct: int

    @property
    def accuracy(self) -> float:
        return 0.0 if self.total == 0 else self.correct / self.total


def evaluate_gsm8k_predictions(
    predictions: Sequence[str],
    targets: Sequence[str],
) -> EvalStats:
    """
    Compute exact-match accuracy for GSM8K predictions.

    Both predictions and targets are compared using GSM8K-specific answer
    normalization rules.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) does not match "
            f"number of targets ({len(targets)})."
        )

    total = len(targets)
    correct = 0
    for pred, tgt in zip(predictions, targets):
        if gsm8k_exact_match(pred, tgt):
            correct += 1

    return EvalStats(total=total, correct=correct)


def evaluate_math_predictions(
    predictions: Sequence[str],
    targets: Sequence[str],
) -> EvalStats:
    """
    Compute exact-match accuracy for MATH predictions.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) does not match "
            f"number of targets ({len(targets)})."
        )

    total = len(targets)
    correct = 0
    for pred, tgt in zip(predictions, targets):
        if math_exact_match(pred, tgt):
            correct += 1

    return EvalStats(total=total, correct=correct)


def evaluate_somadhan_predictions(
    predictions: Sequence[str],
    targets: Sequence[str],
) -> EvalStats:
    """
    Compute exact-match accuracy for Somadhan predictions.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) does not match "
            f"number of targets ({len(targets)})."
        )

    total = len(targets)
    correct = 0
    for pred, tgt in zip(predictions, targets):
        if Somadhan_exact_match(pred, tgt):
            correct += 1

    return EvalStats(total=total, correct=correct)
