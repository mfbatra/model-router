import pytest

from model_router.domain.exceptions import NoSuitableModelError
from model_router.domain.models import (
    ModelConfig,
    Provider,
    Request,
    RoutingConstraints,
)
from model_router.routing.selector import ModelSelector
from model_router.routing.strategies.base import IRoutingStrategy


class _FakeStrategy(IRoutingStrategy):
    def __init__(self):
        self.calls = []

    def name(self) -> str:
        return "fake"

    def score_model(self, model, complexity, constraints) -> float:
        score = 1 / (model.pricing * 10)
        self.calls.append((model.model_name, complexity))
        return score


def _model(name: str, price: float, caps: set[str]):
    return ModelConfig(
        provider=Provider.OPENAI,
        model_name=name,
        pricing=price,
        capabilities=frozenset(caps),
    )


def _request(prompt: str = "Explain something complex") -> Request:
    return Request(prompt=prompt)


def test_selector_picks_highest_score_and_lists_alternatives():
    strategy = _FakeStrategy()
    selector = ModelSelector(strategy)
    candidates = [
        _model("cheap", 0.02, {"chat"}),
        _model("premium", 0.05, {"reasoning", "code", "low-latency"}),
    ]
    constraints = RoutingConstraints(max_cost=0.1)

    decision = selector.select(candidates, _request(), constraints)

    assert decision.selected_model.model_name == "cheap"
    assert decision.alternatives_considered[0].model_name == "premium"
    assert "fake" in decision.reasoning


def test_selector_applies_cost_constraint():
    strategy = _FakeStrategy()
    selector = ModelSelector(strategy)
    cheap = _model("cheap", 0.02, {"chat"})
    pricey = _model("pricey", 0.2, {"reasoning"})
    constraints = RoutingConstraints(max_cost=0.05)

    decision = selector.select([cheap, pricey], _request(), constraints)

    assert decision.selected_model == cheap
    assert decision.alternatives_considered == ()


def test_selector_applies_quality_constraint():
    strategy = _FakeStrategy()
    selector = ModelSelector(strategy)
    high_quality = _model("hq", 0.05, {"reasoning", "code", "analysis"})
    low_quality = _model("lq", 0.03, {"chat"})
    constraints = RoutingConstraints(max_cost=0.1, min_quality=0.7)

    decision = selector.select([high_quality, low_quality], _request(), constraints)

    assert decision.selected_model == high_quality


def test_selector_applies_latency_constraint():
    strategy = _FakeStrategy()
    selector = ModelSelector(strategy)
    fast = _model("fast", 0.05, {"low-latency", "chat"})
    slow = _model("slow", 0.04, {"chat"})
    constraints = RoutingConstraints(max_latency=100)

    decision = selector.select([fast, slow], _request(), constraints)

    assert decision.selected_model == fast


def test_selector_raises_when_no_models_remaining():
    strategy = _FakeStrategy()
    selector = ModelSelector(strategy)
    models = [_model("expensive", 0.2, {"chat"})]
    constraints = RoutingConstraints(max_cost=0.05, min_quality=0.9)

    with pytest.raises(NoSuitableModelError):
        selector.select(models, _request(), constraints)
