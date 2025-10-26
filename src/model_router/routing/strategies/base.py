"""Routing strategy protocol used by scorer implementations."""

from __future__ import annotations

from typing import Protocol

from model_router.domain.models import ModelConfig, RoutingConstraints


class IRoutingStrategy(Protocol):
    """Scores model configurations for ranking ahead of selection."""

    def score_model(
        self,
        model: ModelConfig,
        complexity: float,
        constraints: RoutingConstraints,
    ) -> float:
        """Return a normalized score where higher means more desirable."""

    def name(self) -> str:
        """Stable identifier used for observability/analytics."""
