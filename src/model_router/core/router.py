"""Main router facade coordinating routing, providers, and middleware."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence

from model_router.core.config import RouterConfig
from model_router.core.middleware import IMiddleware, MiddlewareChain
from model_router.domain.interfaces import IUsageTracker
from model_router.domain.models import (
    Request,
    Response,
    RoutingConstraints,
    RoutingDecision,
    RoutingStrategy,
)
from model_router.providers.base import ProviderConfig
from model_router.providers.factory import ProviderFactory
from model_router.routing.engine import RoutingEngine


class Router:
    """High-level API for consumers to issue completions via routing engine."""

    def __init__(
        self,
        config: RouterConfig,
        provider_factory: ProviderFactory,
        routing_engine: RoutingEngine,
        provider_configs: Mapping[str, ProviderConfig],
        *,
        tracker: Optional[IUsageTracker] = None,
        middleware: Optional[MiddlewareChain] = None,
        middlewares: Optional[Sequence[IMiddleware]] = None,
    ) -> None:
        if middleware and middlewares:
            raise ValueError("Provide either 'middleware' or 'middlewares', not both")
        self._config = config
        self._provider_factory = provider_factory
        self._routing_engine = routing_engine
        self._provider_configs = dict(provider_configs)
        self._tracker = tracker
        self._middleware = middleware or MiddlewareChain(middlewares or [])
        self._forced_provider: Optional[str] = None

    def complete(
        self,
        prompt: str,
        *,
        max_cost: Optional[float] = None,
        max_latency: Optional[int] = None,
        min_quality: Optional[float] = None,
        strategy: Optional[str] = None,
        **llm_kwargs: Any,
    ) -> Response:
        request = Request(
            prompt=prompt,
            params=llm_kwargs,
            metadata={"entry_point": "complete"},
        )
        constraints = self._build_constraints(
            max_cost=max_cost,
            max_latency=max_latency,
            min_quality=min_quality,
            strategy=strategy,
        )

        def handler(processed_request: Request) -> Response:
            decision = self._routing_engine.route(processed_request, constraints)
            return self._execute_with_fallback(decision, processed_request)

        response = self._middleware.execute(request, handler)

        if self._tracker and self._config.enable_analytics:
            self._tracker.track(request, response)

        return response

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Response:
        prompt = self._messages_to_prompt(messages)
        return self.complete(prompt, **kwargs)

    def configure_fallback(self, models: List[str]) -> None:
        self._config = replace(self._config, fallback_models=list(models))

    @property
    def default_provider(self) -> Optional[str]:
        return self._forced_provider

    @default_provider.setter
    def default_provider(self, provider_name: Optional[str]) -> None:
        if provider_name is not None and provider_name not in self._provider_configs:
            raise ValueError(
                f"Unknown provider '{provider_name}'. "
                f"Available: {sorted(self._provider_configs)}"
            )
        self._forced_provider = provider_name

    @property
    def analytics(self) -> IUsageTracker:
        if not self._tracker:
            raise RuntimeError("Analytics tracker not configured")
        return self._tracker

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_constraints(
        self,
        *,
        max_cost: Optional[float],
        max_latency: Optional[int],
        min_quality: Optional[float],
        strategy: Optional[str],
    ) -> RoutingConstraints:
        strategy_name = strategy or self._config.default_strategy
        routed_strategy = self._map_strategy_to_enum(strategy_name)
        safe_max_cost = max_cost if max_cost is not None else 1_000.0
        safe_max_latency = max_latency if max_latency is not None else 30_000
        return RoutingConstraints(
            max_cost=safe_max_cost,
            max_latency=safe_max_latency,
            min_quality=min_quality,
            strategy=routed_strategy,
        )

    def _execute_with_fallback(
        self,
        decision: RoutingDecision,
        request: Request,
    ) -> Response:
        initial_candidates = [decision.selected_model, *decision.alternatives_considered]
        provider_attempts: list[tuple[str, str]] = [
            (candidate.model_name, candidate.provider.value)
            for candidate in initial_candidates
        ]

        seen_models = set(provider_attempts)
        for fallback_model in self._config.fallback_models:
            try:
                provider_key = self._provider_factory.infer_provider_key(fallback_model)
            except Exception:
                continue
            candidate = (fallback_model, provider_key)
            if candidate not in seen_models:
                provider_attempts.append(candidate)
                seen_models.add(candidate)

        if self._forced_provider:
            preferred = [
                attempt
                for attempt in provider_attempts
                if attempt[1] == self._forced_provider
            ]
            rest = [
                attempt
                for attempt in provider_attempts
                if attempt[1] != self._forced_provider
            ]
            provider_attempts = preferred + rest

        attempts = 0
        last_error: Optional[Exception] = None
        for model_name, provider_key in provider_attempts:
            if attempts >= self._config.max_retries:
                break
            provider_config = self._provider_configs.get(provider_key)
            if provider_config is None:
                continue
            provider = self._provider_factory.create(model_name, provider_config)
            try:
                return provider.complete(request)
            except Exception as exc:  # pragma: no cover - provider failures
                last_error = exc
                attempts += 1
                continue

        raise RuntimeError("All provider attempts failed") from last_error

    @staticmethod
    def _messages_to_prompt(messages: Sequence[Dict[str, str]]) -> str:
        return "\n".join(
            f"{message.get('role', 'user')}: {message.get('content', '')}"
            for message in messages
        )

    @staticmethod
    def _map_strategy_to_enum(name: str) -> RoutingStrategy:
        mapping = {
            "balanced": RoutingStrategy.BALANCED,
            "cost_optimized": RoutingStrategy.COST_BIASED,
            "quality_optimized": RoutingStrategy.QUALITY_BIASED,
            "latency_optimized": RoutingStrategy.LATENCY_BIASED,
        }
        try:
            return mapping[name.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported strategy '{name}'") from exc
