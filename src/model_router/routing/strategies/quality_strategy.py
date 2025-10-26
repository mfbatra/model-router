"""Quality-first routing strategy based on capability coverage."""

from __future__ import annotations

from model_router.domain.models import ModelConfig, RoutingConstraints

from .base import IRoutingStrategy


class QualityOptimizedStrategy(IRoutingStrategy):
    """Rewards models that advertise richer reasoning/code capabilities."""

    WEIGHTS = {
        "reasoning": 0.4,
        "code": 0.3,
        "vision": 0.15,
        "chat": 0.1,
        "analysis": 0.05,
    }

    def name(self) -> str:
        return "quality_optimized"

    def score_model(
        self,
        model: ModelConfig,
        complexity: float,
        constraints: RoutingConstraints,
    ) -> float:
        capability_score = self._capability_score(model)
        complexity_boost = 0.5 + 0.5 * _clamp(complexity)
        score = capability_score * complexity_boost

        if constraints.max_cost is not None and model.pricing > constraints.max_cost:
            score *= 0.9
        if constraints.min_quality is not None:
            score += 0.1 * _clamp(constraints.min_quality)

        return _clamp(score)

    def _capability_score(self, model: ModelConfig) -> float:
        total = sum(self.WEIGHTS.values())
        raw = sum(self.WEIGHTS.get(cap, 0.0) for cap in model.capabilities)
        if total == 0:
            return 0.0
        return raw / total


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(value, maximum))
