import pytest

from model_router.domain.models import (
    ModelConfig,
    Provider,
    Request,
    RoutingConstraints,
    RoutingDecision,
)
from model_router.routing.engine import RoutingEngine
from model_router.domain.interfaces import IComplexityEstimator, IModelSelector


class _FakeEstimator(IComplexityEstimator):
    def __init__(self, value: float):
        self.value = value
        self.calls = []

    def estimate(self, prompt: str) -> float:
        self.calls.append(prompt)
        return self.value


class _FakeSelector(IModelSelector):
    def __init__(self, decision: RoutingDecision):
        self.decision = decision
        self.calls = []

    def select(self, models, request, constraints):
        self.calls.append((tuple(models), request, constraints))
        return self.decision


@pytest.fixture
def models():
    return [
        ModelConfig(
            provider=Provider.OPENAI,
            model_name="gpt-4",
            pricing=0.03,
            capabilities=frozenset({"chat", "reasoning"}),
        ),
        ModelConfig(
            provider=Provider.ANTHROPIC,
            model_name="claude-3",
            pricing=0.04,
            capabilities=frozenset({"chat", "reasoning", "low-latency"}),
        ),
    ]


@pytest.fixture
def sample_request():
    return Request(prompt="Explain the differences between tensors and matrices")


@pytest.fixture
def constraints():
    return RoutingConstraints(max_cost=0.05)


def test_routing_engine_routes_with_injected_dependencies(
    models, sample_request, constraints
):
    base_decision = RoutingDecision(
        selected_model=models[0],
        estimated_cost=models[0].pricing,
        reasoning="base reasoning",
        alternatives_considered=(models[1],),
    )
    estimator = _FakeEstimator(0.72)
    selector = _FakeSelector(base_decision)
    engine = RoutingEngine(estimator, selector, models)

    decision = engine.route(sample_request, constraints)

    assert estimator.calls == [sample_request.prompt]
    assert selector.calls[0][0][0] == models[0]
    assert "complexity=0.72" in decision.reasoning
    assert decision.selected_model == models[0]


def test_routing_engine_explain_includes_alternatives(models):
    engine = RoutingEngine(
        _FakeEstimator(0.5),
        _FakeSelector(
            RoutingDecision(
                selected_model=models[0],
                estimated_cost=models[0].pricing,
                reasoning="selected due to cost",
                alternatives_considered=(models[1],),
            )
        ),
        models,
    )

    decision = engine.route(
        Request(prompt="Describe"),
        RoutingConstraints(max_cost=0.1),
    )

    explanation = engine.explain(decision)

    assert models[0].model_name in explanation
    assert models[1].model_name in explanation
    assert "Reasoning" in explanation


def test_routing_engine_requires_models():
    with pytest.raises(ValueError):
        RoutingEngine(
            _FakeEstimator(0.1),
            _FakeSelector(
                RoutingDecision(
                    selected_model=ModelConfig(
                        provider=Provider.CUSTOM,
                        model_name="custom",
                        pricing=0.01,
                        capabilities=frozenset(),
                    ),
                    estimated_cost=0.01,
                    reasoning="",
                    alternatives_considered=(),
                )
            ),
            [],
        )
