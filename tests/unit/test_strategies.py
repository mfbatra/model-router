import pytest

from model_router.domain.models import ModelConfig, Provider, RoutingConstraints
from model_router.routing.strategies.balanced_strategy import BalancedStrategy
from model_router.routing.strategies.cost_strategy import CostOptimizedStrategy
from model_router.routing.strategies.latency_strategy import LatencyOptimizedStrategy
from model_router.routing.strategies.quality_strategy import QualityOptimizedStrategy


def _model(provider, name, price, caps):
    return ModelConfig(
        provider=provider,
        model_name=name,
        pricing=price,
        capabilities=frozenset(caps),
    )


def test_cost_strategy_prefers_cheaper_models():
    cheap = _model(Provider.OPENAI, "cheap", 0.02, {"chat"})
    expensive = _model(Provider.OPENAI, "expensive", 0.08, {"chat", "reasoning"})
    constraints = RoutingConstraints(max_cost=0.05)
    strategy = CostOptimizedStrategy()

    score_cheap = strategy.score_model(cheap, complexity=0.5, constraints=constraints)
    score_expensive = strategy.score_model(
        expensive, complexity=0.5, constraints=constraints
    )

    assert score_cheap > score_expensive


def test_quality_strategy_prefers_reasoning_models():
    reasoning = _model(
        Provider.ANTHROPIC, "reasoning", 0.06, {"reasoning", "code", "chat"}
    )
    chat_only = _model(Provider.OPENAI, "chat", 0.03, {"chat"})
    constraints = RoutingConstraints(max_cost=0.1, min_quality=0.8)
    strategy = QualityOptimizedStrategy()

    score_reasoning = strategy.score_model(
        reasoning, complexity=0.9, constraints=constraints
    )
    score_chat = strategy.score_model(
        chat_only, complexity=0.9, constraints=constraints
    )

    assert score_reasoning > score_chat


def test_latency_strategy_prefers_low_latency_caps():
    realtime = _model(Provider.GOOGLE, "fast", 0.05, {"low-latency", "streaming"})
    batch = _model(Provider.OPENAI, "batch", 0.03, {"batch", "chat"})
    constraints = RoutingConstraints(max_latency=200)
    strategy = LatencyOptimizedStrategy()

    score_fast = strategy.score_model(realtime, complexity=0.4, constraints=constraints)
    score_batch = strategy.score_model(batch, complexity=0.4, constraints=constraints)

    assert score_fast > score_batch


def test_balanced_strategy_combines_multiple_signals():
    cheap_basic = _model(Provider.OPENAI, "cheap-basic", 0.02, {"chat"})
    pricey_quality = _model(
        Provider.ANTHROPIC, "premium", 0.09, {"reasoning", "code", "low-latency"}
    )
    constraints = RoutingConstraints(max_cost=0.08, max_latency=300, min_quality=0.6)
    strategy = BalancedStrategy()

    cheap_score = strategy.score_model(
        cheap_basic, complexity=0.6, constraints=constraints
    )
    premium_score = strategy.score_model(
        pricey_quality, complexity=0.6, constraints=constraints
    )

    # Balanced strategy should still lean toward capability-rich model
    assert premium_score > cheap_score
