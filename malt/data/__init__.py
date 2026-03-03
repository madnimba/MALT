"""
Dataset loaders and evaluation helpers for MALT.

Currently includes:
- GSM8K loading and answer normalization utilities.
- MATH loading and answer normalization utilities.
"""

from .gsm8k import (
    Gsm8kExample,
    load_gsm8k_split,
    extract_gsm8k_answer,
    normalize_gsm8k_answer,
    gsm8k_exact_match,
)
from .math import (
    MathExample,
    load_math_split,
    extract_math_answer,
    normalize_math_answer,
    math_exact_match,
)

__all__ = [
    "Gsm8kExample",
    "load_gsm8k_split",
    "extract_gsm8k_answer",
    "normalize_gsm8k_answer",
    "gsm8k_exact_match",
    "MathExample",
    "load_math_split",
    "extract_math_answer",
    "normalize_math_answer",
    "math_exact_match",
]


