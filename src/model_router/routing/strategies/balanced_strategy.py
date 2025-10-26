"""Balanced routing strategy that blends cost, quality, and latency signals."""

from __future__ import annotations

from model_router.domain.models import ModelConfig, RoutingConstraints

from .base import IRoutingStrategy
from .cost_strategy import CostOptimizedStrategy
from .latency_strategy import LatencyOptimizedStrategy
from .quality_strategy import QualityOptimizedStrategy


class BalancedStrategy(IRoutingStrategy):
    """Weighted combination of the other strategies."""

    def __init__(self) -> None:
        self._cost = CostOptimizedStrategy()
        self._quality = QualityOptimizedStrategy()
        self._latency = LatencyOptimizedStrategy()

    def name(self) -> str:
        return "balanced"

    def score_model(
        self,
        model: ModelConfig,
        complexity: float,
        constraints: RoutingConstraints,
    ) -> float:
        cost_score = self._cost.score_model(model, complexity, constraints)
        quality_score = self._quality.score_model(model, complexity, constraints)
        latency_score = self._latency.score_model(model, complexity, constraints)

        score = 0.3 * cost_score + 0.45 * quality_score + 0.25 * latency_score
        return _clamp(score)


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(value, maximum))
