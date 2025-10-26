import time

import pytest

from model_router.domain.models import (
    ModelConfig,
    Provider,
    RoutingConstraints,
    RoutingStrategy,
)
from model_router.utils import cost_calculator, retry, token_counter, validators


def test_calculate_cost_known_model():
    model = ModelConfig(
        provider=Provider.OPENAI,
        model_name="gpt-4",
        pricing=0.02,
        capabilities=frozenset(),
    )
    cost = cost_calculator.calculate_cost(model, tokens_in=1000, tokens_out=500)
    assert cost == pytest.approx(0.06)


def test_calculate_cost_unknown_model_uses_default():
    cost = cost_calculator.calculate_cost("unknown", tokens_in=1000, tokens_out=1000)
    assert cost == pytest.approx(0.04)


def test_estimate_tokens_counts_words_and_punctuation():
    text = "Hello world, how are you?"
    tokens = cost_calculator.estimate_tokens(text)
    assert tokens >= 6


def test_token_counter_returns_positive_count():
    tokens = token_counter.count_tokens_approximate("This is a test!")
    assert tokens >= 4


def test_validate_prompt_rejects_empty():
    with pytest.raises(ValueError):
        validators.validate_prompt("   ")


def test_validate_constraints_rejects_invalid_values():
    constraints = RoutingConstraints.model_construct(
        max_cost=1,
        max_latency=None,
        min_quality=1.5,
        strategy=RoutingStrategy.BALANCED,
    )
    with pytest.raises(ValueError):
        validators.validate_constraints(constraints)


def test_retry_decorator_retries_specified_attempts(monkeypatch):
    calls = {"count": 0}

    @retry.retry(attempts=3, delay=0.01)
    def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise ValueError("fail")
        return "ok"

    assert flaky() == "ok"
    assert calls["count"] == 3
