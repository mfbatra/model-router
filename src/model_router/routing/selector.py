"""Model selection logic orchestrating constraint filtering and scoring."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from model_router.domain.exceptions import NoSuitableModelError
from model_router.domain.interfaces import IModelSelector
from model_router.domain.models import (
    ModelConfig,
    Request,
    RoutingConstraints,
    RoutingDecision,
)

from .estimator import default_complexity_estimator
from .strategies.base import IRoutingStrategy


class ModelSelector(IModelSelector):
    """Selects the best model given constraints and a routing strategy."""

    QUALITY_WEIGHTS = {
        "reasoning": 1.0,
        "code": 0.8,
        "analysis": 0.6,
        "vision": 0.5,
        "chat": 0.4,
    }
    LOW_LATENCY_TAGS = {"low-latency", "realtime", "streaming"}

    def __init__(self, strategy: IRoutingStrategy) -> None:
        self._strategy = strategy
        self._complexity_estimator = default_complexity_estimator()

    def select(
        self,
        available_models: Sequence[ModelConfig],
        request: Request,
        constraints: RoutingConstraints,
    ) -> RoutingDecision:
        eligible = self._filter_by_constraints(available_models, constraints)
        if not eligible:
            raise NoSuitableModelError("No models satisfy hard constraints")

        complexity = self._complexity_estimator.estimate(request.prompt)
        scored = self._score_models(eligible, complexity, constraints)
        scored.sort(key=lambda item: item[1], reverse=True)

        best_model, best_score = scored[0]
        decision = self._create_decision(best_model, best_score, scored, constraints)
        return decision

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _filter_by_constraints(
        self, models: Sequence[ModelConfig], constraints: RoutingConstraints
    ) -> List[ModelConfig]:
        filtered: List[ModelConfig] = []
        for model in models:
            if (
                constraints.max_cost is not None
                and model.pricing > constraints.max_cost
            ):
                continue
            if constraints.min_quality is not None:
                if self._quality_score(model) < constraints.min_quality:
                    continue
            if constraints.max_latency is not None and constraints.max_latency <= 200:
                if not self._supports_low_latency(model):
                    continue
            filtered.append(model)
        return filtered

    def _score_models(
        self,
        models: Sequence[ModelConfig],
        complexity: float,
        constraints: RoutingConstraints,
    ) -> List[Tuple[ModelConfig, float]]:
        return [
            (model, self._strategy.score_model(model, complexity, constraints))
            for model in models
        ]

    def _create_decision(
        self,
        selected: ModelConfig,
        top_score: float,
        scored: Sequence[Tuple[ModelConfig, float]],
        constraints: RoutingConstraints,
    ) -> RoutingDecision:
        alternatives = tuple(model for model, _ in scored if model is not selected)
        reasoning = self._build_reasoning(selected, top_score, scored, constraints)
        return RoutingDecision(
            selected_model=selected,
            estimated_cost=selected.pricing,
            reasoning=reasoning,
            alternatives_considered=alternatives,
        )

    def _build_reasoning(
        self,
        selected: ModelConfig,
        top_score: float,
        scored: Sequence[Tuple[ModelConfig, float]],
        constraints: RoutingConstraints,
    ) -> str:
        summary = (
            f"Strategy {self._strategy.name()} selected {selected.model_name} "
            f"with score {top_score:.2f} under constraints"
        )
        parts = []
        if constraints.max_cost is not None:
            parts.append(f"max_cost={constraints.max_cost}")
        if constraints.max_latency is not None:
            parts.append(f"max_latency={constraints.max_latency}")
        if constraints.min_quality is not None:
            parts.append(f"min_quality={constraints.min_quality}")
        constraint_text = f" ({', '.join(parts)})" if parts else ""

        alt_names = [model.model_name for model, _ in scored[1:3]]
        alt_text = f". Considered: {', '.join(alt_names)}" if alt_names else ""
        return summary + constraint_text + alt_text

    def _quality_score(self, model: ModelConfig) -> float:
        if not model.capabilities:
            return 0.0
        total = sum(self.QUALITY_WEIGHTS.values())
        raw = sum(
            weight
            for cap, weight in self.QUALITY_WEIGHTS.items()
            if cap in model.capabilities
        )
        return raw / total

    def _supports_low_latency(self, model: ModelConfig) -> bool:
        return any(cap in self.LOW_LATENCY_TAGS for cap in model.capabilities)
