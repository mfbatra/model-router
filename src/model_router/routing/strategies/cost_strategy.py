"""Cost-first routing strategy."""

from __future__ import annotations

from model_router.domain.models import ModelConfig, RoutingConstraints

from .base import IRoutingStrategy


class CostOptimizedStrategy(IRoutingStrategy):
    """Prefers the lowest-priced models while respecting constraints."""

    def name(self) -> str:
        return "cost_optimized"

    def score_model(
        self,
        model: ModelConfig,
        complexity: float,
        constraints: RoutingConstraints,
    ) -> float:
        price = max(model.pricing, 1e-6)
        base = 1 / (1 + price * 10)

        if constraints.max_cost is not None and price > constraints.max_cost:
            base *= 0.2

        penalty = 0.15 * _clamp(complexity)
        score = base * (1 - penalty)
        return _clamp(score)


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(value, maximum))
