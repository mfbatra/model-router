"""Routing engine facade coordinating estimation and selection."""

from __future__ import annotations

from typing import Sequence

from model_router.domain.interfaces import IComplexityEstimator, IModelSelector
from model_router.domain.models import (
    ModelConfig,
    Request,
    RoutingConstraints,
    RoutingDecision,
)


class RoutingEngine:
    """High-level facade that orchestrates routing without owning the logic."""

    def __init__(
        self,
        estimator: IComplexityEstimator,
        selector: IModelSelector,
        models: Sequence[ModelConfig],
    ) -> None:
        if not models:
            raise ValueError("At least one model must be provided to the engine")
        self._estimator = estimator
        self._selector = selector
        self._models = list(models)

    def route(
        self, request: Request, constraints: RoutingConstraints
    ) -> RoutingDecision:
        complexity = self._estimator.estimate(request.prompt)
        decision = self._selector.select(self._models, request, constraints)
        return RoutingDecision(
            selected_model=decision.selected_model,
            estimated_cost=decision.estimated_cost,
            reasoning=decision.reasoning + f" | complexity={complexity:.2f}",
            alternatives_considered=decision.alternatives_considered,
        )

    def explain(self, decision: RoutingDecision) -> str:
        explanation = (
            f"Selected {decision.selected_model.model_name} from provider "
            f"{decision.selected_model.provider.value} at estimated cost {decision.estimated_cost:.4f}."
        )
        if decision.alternatives_considered:
            alt_names = ", ".join(
                model.model_name for model in decision.alternatives_considered[:3]
            )
            explanation += f" Alternatives considered: {alt_names}."
        explanation += f" Reasoning: {decision.reasoning}"
        return explanation
