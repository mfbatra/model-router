"""Cost calculation helpers for model usage."""

from __future__ import annotations

from typing import Mapping, Union

from model_router.domain.models import ModelConfig

PRICING: Mapping[str, Mapping[str, float]] = {
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4o": {"input": 0.01, "output": 0.03},
    "claude-3": {"input": 0.008, "output": 0.024},
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "gemini-1.5-pro": {"input": 0.01, "output": 0.02},
}

DEFAULT_PRICING = {"input": 0.02, "output": 0.02}


def calculate_cost(
    model: Union[ModelConfig, str], tokens_in: int, tokens_out: int
) -> float:
    """Return estimated USD cost based on configured pricing."""

    model_name = model.model_name if isinstance(model, ModelConfig) else str(model)
    pricing = PRICING.get(model_name, DEFAULT_PRICING)
    cost = (tokens_in / 1000) * pricing["input"] + (tokens_out / 1000) * pricing[
        "output"
    ]
    return round(cost, 6)


def estimate_tokens(text: str) -> int:
    """Rough token estimation derived from whitespace and punctuation."""

    if not text:
        return 0
    words = text.strip().split()
    punctuation_bonus = text.count(",") + text.count(".") + text.count("\n")
    return max(1, int(len(words) * 1.3) + punctuation_bonus)
