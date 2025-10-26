"""Prompt complexity estimation utilities for routing decisions."""

from __future__ import annotations

import re
from statistics import fmean
from typing import Protocol, Sequence

from model_router.domain.interfaces import IComplexityEstimator


class IFeatureExtractor(Protocol):
    """Extracts a normalized feature score (0-1) from a prompt."""

    def extract(self, prompt: str) -> float:  # pragma: no cover - Protocol signature
        ...


class ComplexityEstimator(IComplexityEstimator):
    """Aggregates feature extractors to produce a 0-1 complexity score."""

    def __init__(self, features: Sequence[IFeatureExtractor]):
        if not features:
            raise ValueError("At least one feature extractor must be provided")
        self._features = list(features)

    def estimate(self, prompt: str) -> float:
        if not prompt or not prompt.strip():
            return 0.0
        scores = [_clamp(feature.extract(prompt)) for feature in self._features]
        return _clamp(fmean(scores))


class LengthFeatureExtractor:
    """Scores prompts higher as they grow longer relative to a target length."""

    def __init__(self, target_chars: int = 300):
        self.target_chars = max(1, target_chars)

    def extract(self, prompt: str) -> float:
        return _clamp(len(prompt) / self.target_chars)


class CodeBlockFeatureExtractor:
    """Detects code-specific structures such as fenced blocks or syntax tokens."""

    FENCE_PATTERN = re.compile(r"```.+?```", re.DOTALL)
    CODE_KEYWORDS = ("def ", "class ", "SELECT ", "function", "public ", "{", ";", "</")

    def extract(self, prompt: str) -> float:
        if self.FENCE_PATTERN.search(prompt):
            return 1.0
        lowered = prompt.lower()
        matches = sum(
            1 for keyword in self.CODE_KEYWORDS if keyword.strip().lower() in lowered
        )
        return _clamp(matches / len(self.CODE_KEYWORDS))


class ReasoningKeywordExtractor:
    """Counts reasoning verbs/phrases that usually imply higher complexity."""

    KEYWORDS = (
        "reason",
        "explain",
        "derive",
        "analyze",
        "justify",
        "step-by-step",
        "compare",
        "evaluate",
    )

    def extract(self, prompt: str) -> float:
        lowered = prompt.lower()
        matches = sum(1 for keyword in self.KEYWORDS if keyword in lowered)
        return _clamp(matches / 2)


class TechnicalTermExtractor:
    """Detects technical vocabulary spanning math, CS, and engineering domains."""

    TERMS = (
        "tensor",
        "gradient",
        "database",
        "encryption",
        "neural",
        "api",
        "schema",
        "complexity",
        "algorithm",
        "probability",
        "latency",
    )

    def extract(self, prompt: str) -> float:
        lowered = prompt.lower()
        matches = sum(1 for term in self.TERMS if term in lowered)
        return _clamp(matches / 3)


def default_complexity_estimator() -> ComplexityEstimator:
    """Factory producing an estimator with the built-in feature set."""

    return ComplexityEstimator(
        [
            LengthFeatureExtractor(),
            CodeBlockFeatureExtractor(),
            ReasoningKeywordExtractor(),
            TechnicalTermExtractor(),
        ]
    )


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(value, maximum))
