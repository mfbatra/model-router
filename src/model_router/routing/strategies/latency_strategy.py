"""Latency-focused routing strategy."""

from __future__ import annotations

from model_router.domain.models import ModelConfig, RoutingConstraints

from .base import IRoutingStrategy


class LatencyOptimizedStrategy(IRoutingStrategy):
    """Prefers models tagged with low-latency or streaming capabilities."""

    LATENCY_WEIGHTS = {
        "low-latency": 1.0,
        "realtime": 0.9,
        "streaming": 0.7,
        "batch": -0.4,
    }

    def name(self) -> str:
        return "latency_optimized"

    def score_model(
        self,
        model: ModelConfig,
        complexity: float,
        constraints: RoutingConstraints,
    ) -> float:
        capability_score = sum(
            self.LATENCY_WEIGHTS.get(cap, 0.0) for cap in model.capabilities
        )
        capability_score = max(capability_score, 0.0) / max(
            len(self.LATENCY_WEIGHTS), 1
        )

        cost_component = 1 / (1 + model.pricing * 5)
        complexity_penalty = 0.1 * _clamp(complexity)

        score = capability_score + 0.3 * cost_component - complexity_penalty

        if constraints.max_latency is not None:
            score *= 1 + (1 / (constraints.max_latency + 1))

        return _clamp(score)


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(value, maximum))
