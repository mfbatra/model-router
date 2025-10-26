"""Dependency injection container for building fully-wired Router instances."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence

import httpx

from model_router.analytics.aggregator import AnalyticsAggregator
from model_router.analytics.sqlite_repository import SQLiteRepository
from model_router.analytics.tracker import UsageTracker
from model_router.core.config import RouterConfig
from model_router.core.middleware import (
    AnalyticsMiddleware,
    CachingMiddleware,
    LoggingMiddleware,
    MiddlewareChain,
    ValidationMiddleware,
    IMiddleware,
)
from model_router.core.router import Router
from model_router.domain.interfaces import IUsageTracker
from model_router.routing.strategies.base import IRoutingStrategy
from model_router.domain.models import ModelConfig
from model_router.providers.base import ProviderConfig
from model_router.providers.factory import ProviderFactory
from model_router.providers.openai_provider import OpenAIProvider
from model_router.providers.anthropic_provider import AnthropicProvider
from model_router.providers.google_provider import GoogleProvider
from model_router.routing.estimator import default_complexity_estimator
from model_router.routing.selector import ModelSelector
from model_router.routing.engine import RoutingEngine
from model_router.routing.strategies.balanced_strategy import BalancedStrategy
from model_router.routing.strategies.cost_strategy import CostOptimizedStrategy
from model_router.routing.strategies.latency_strategy import LatencyOptimizedStrategy
from model_router.routing.strategies.quality_strategy import QualityOptimizedStrategy


class DIContainer:
    """Factory helpers that assemble a Router with default wiring."""

    @staticmethod
    def create_router(
        *,
        openai_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        anthropic_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        google_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        config: Optional[RouterConfig] = None,
        analytics_db_path: str | Path = "analytics.db",
    ) -> Router:
        resolved_openai_key = DIContainer._normalize_api_key(
            "openai", openai_key, openai_api_key
        )
        resolved_anthropic_key = DIContainer._normalize_api_key(
            "anthropic", anthropic_key, anthropic_api_key
        )
        resolved_google_key = DIContainer._normalize_api_key(
            "google", google_key, google_api_key
        )

        cfg = config or RouterConfig.from_env()
        provider_configs = DIContainer._build_provider_configs(
            openai_key=resolved_openai_key,
            anthropic_key=resolved_anthropic_key,
            google_key=resolved_google_key,
        )
        if not provider_configs:
            raise ValueError("At least one provider API key must be supplied")

        http_client_factory = DIContainer._build_http_client_factory()
        provider_factory = ProviderFactory(http_client_factory)

        estimator = default_complexity_estimator()
        strategy = DIContainer._select_strategy(cfg.default_strategy)
        selector = ModelSelector(strategy)
        models = DIContainer._default_model_configs()
        routing_engine = RoutingEngine(estimator, selector, models)

        repository = SQLiteRepository(analytics_db_path)
        aggregator = AnalyticsAggregator()
        tracker: Optional[IUsageTracker] = (
            UsageTracker(repository, aggregator) if cfg.enable_analytics else None
        )

        middleware_chain = DIContainer._build_middleware_chain(cfg, tracker)

        return Router(
            config=cfg,
            provider_factory=provider_factory,
            routing_engine=routing_engine,
            provider_configs=provider_configs,
            tracker=tracker,
            middleware=middleware_chain,
        )

    @staticmethod
    def create_custom_router(
        *,
        config: RouterConfig,
        provider_factory: ProviderFactory,
        routing_engine: RoutingEngine,
        provider_configs: Mapping[str, ProviderConfig],
        tracker: Optional[IUsageTracker] = None,
        middleware: Optional[MiddlewareChain] = None,
        middlewares: Optional[Sequence[IMiddleware]] = None,
    ) -> Router:
        return Router(
            config=config,
            provider_factory=provider_factory,
            routing_engine=routing_engine,
            provider_configs=provider_configs,
            tracker=tracker,
            middleware=middleware,
            middlewares=middlewares,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_provider_configs(
        *,
        openai_key: Optional[str],
        anthropic_key: Optional[str],
        google_key: Optional[str],
    ) -> dict[str, ProviderConfig]:
        configs: dict[str, ProviderConfig] = {}
        if openai_key:
            configs["openai"] = ProviderConfig(
                api_key=openai_key,
                base_url="https://api.openai.com",
            )
        if anthropic_key:
            configs["anthropic"] = ProviderConfig(
                api_key=anthropic_key,
                base_url="https://api.anthropic.com",
            )
        if google_key:
            configs["google"] = ProviderConfig(
                api_key=google_key,
                base_url="https://generativelanguage.googleapis.com",
            )
        return configs

    @staticmethod
    def _build_http_client_factory() -> Callable[[ProviderConfig], httpx.Client]:
        def factory(provider_config: ProviderConfig) -> httpx.Client:
            return httpx.Client(timeout=provider_config.timeout)

        return factory

    @staticmethod
    def _select_strategy(name: str) -> IRoutingStrategy:
        normalized = name.lower()
        mapping: Dict[str, Callable[[], IRoutingStrategy]] = {
            "balanced": BalancedStrategy,
            "cost_optimized": CostOptimizedStrategy,
            "quality_optimized": QualityOptimizedStrategy,
            "latency_optimized": LatencyOptimizedStrategy,
        }
        try:
            return mapping[normalized]()
        except KeyError as exc:
            raise ValueError(f"Unknown strategy '{name}'") from exc

    @staticmethod
    def _default_model_configs() -> list[ModelConfig]:
        return [
            OpenAIProvider.DEFAULT_MODEL_CONFIG,
            AnthropicProvider.DEFAULT_MODEL_CONFIG,
            GoogleProvider.DEFAULT_MODEL_CONFIG,
        ]

    @staticmethod
    def _build_middleware_chain(
        config: RouterConfig, tracker: Optional[IUsageTracker]
    ) -> MiddlewareChain:
        middlewares = [
            ValidationMiddleware(),
            LoggingMiddleware(),
        ]
        if tracker and config.enable_analytics:
            middlewares.append(AnalyticsMiddleware(tracker))
        if config.enable_cache:
            middlewares.append(CachingMiddleware({}))
        return MiddlewareChain(middlewares)

    @staticmethod
    def _normalize_api_key(
        provider_name: str, *keys: Optional[str]
    ) -> Optional[str]:
        """Allow both *_key and *_api_key kwargs while preventing conflicts."""
        provided = [key for key in keys if key]
        if not provided:
            return None
        unique_values = set(provided)
        if len(unique_values) > 1:
            raise ValueError(
                f"Conflicting API keys supplied for {provider_name}: {provided}"
            )
        return provided[0]
