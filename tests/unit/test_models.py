from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from model_router.domain.models import (
    ModelConfig,
    Provider,
    Request,
    Response,
    RoutingConstraints,
    RoutingDecision,
    RoutingStrategy,
)


def test_model_config_immutable():
    config = ModelConfig(
        provider=Provider.OPENAI,
        model_name="gpt-4",
        pricing=0.02,
        capabilities=frozenset({"chat", "reasoning"}),
    )

    with pytest.raises(FrozenInstanceError):
        config.model_name = "gpt-3.5"  # type: ignore[misc]


def test_request_rejects_empty_prompt_and_is_immutable():
    with pytest.raises(ValueError):
        Request(prompt="   ")

    req = Request(prompt="Hello", params={"temperature": 0.5})
    with pytest.raises((TypeError, ValidationError)):
        req.prompt = "updated"  # type: ignore[misc]


def test_response_factory_normalizes_payload():
    response = Response.from_provider_response(
        {
            "content": "ok",
            "model_used": "gpt-4",
            "cost": 0.02,
            "latency": 120,
            "tokens": 256,
        }
    )
    assert response.model_used == "gpt-4"
    assert response.cost == 0.02


def test_routing_constraints_require_bounds():
    with pytest.raises(ValueError):
        RoutingConstraints(min_quality=0.9)

    constraints = RoutingConstraints(max_cost=0.05, min_quality=0.8)
    assert constraints.strategy == RoutingStrategy.BALANCED


def test_routing_decision_rejects_selected_in_alternatives_and_is_immutable():
    primary = ModelConfig(
        provider=Provider.ANTHROPIC,
        model_name="claude-2",
        pricing=0.03,
        capabilities=frozenset({"chat"}),
    )
    alt = ModelConfig(
        provider=Provider.OPENAI,
        model_name="gpt-4",
        pricing=0.02,
        capabilities=frozenset({"chat", "code"}),
    )

    with pytest.raises(ValueError):
        RoutingDecision(
            selected_model=primary,
            estimated_cost=0.03,
            reasoning="preferred",
            alternatives_considered=(primary,),
        )

    decision = RoutingDecision(
        selected_model=primary,
        estimated_cost=0.03,
        reasoning="cost-effective",
        alternatives_considered=(alt,),
    )

    with pytest.raises((TypeError, ValidationError)):
        decision.reasoning = "updated"  # type: ignore[misc]
