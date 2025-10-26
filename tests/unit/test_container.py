import httpx
import pytest

from model_router.core.config import RouterConfig
from model_router.core.container import DIContainer
from model_router.core.router import Router
from model_router.core.middleware import MiddlewareChain
from model_router.domain.interfaces import IUsageTracker
from model_router.domain.models import ModelConfig, Provider
from model_router.providers.base import ProviderConfig
from model_router.providers.factory import ProviderFactory
from model_router.routing.engine import RoutingEngine
from model_router.routing.estimator import default_complexity_estimator
from model_router.routing.selector import ModelSelector
from model_router.routing.strategies.balanced_strategy import BalancedStrategy
from model_router.providers.openai_provider import OpenAIProvider
from model_router.providers.anthropic_provider import AnthropicProvider
from model_router.providers.google_provider import GoogleProvider


class _TrackerStub(IUsageTracker):
    def track(self, request, response):
        pass

    def get_summary(self, period: str = "last_7_days"):
        raise NotImplementedError

    def to_dataframe(self):
        raise NotImplementedError


def test_create_router_requires_key():
    with pytest.raises(ValueError):
        DIContainer.create_router()


def test_create_router_accepts_api_key_alias(tmp_path):
    router = DIContainer.create_router(
        openai_api_key="sk-test",
        analytics_db_path=tmp_path / "analytics.db",
    )

    assert isinstance(router, Router)


def test_create_router_conflicting_key_values():
    with pytest.raises(ValueError):
        DIContainer.create_router(openai_key="one", openai_api_key="two")


def test_create_router_wires_dependencies(tmp_path):
    router = DIContainer.create_router(
        openai_key="sk-test",
        config=RouterConfig(enable_cache=True),
        analytics_db_path=tmp_path / "analytics.db",
    )

    assert isinstance(router, Router)
    assert "openai" in router._provider_configs  # type: ignore[attr-defined]
    assert router.analytics  # Should not raise


def test_create_custom_router_uses_supplied_components(tmp_path):
    config = RouterConfig()

    def http_client_factory(provider_config: ProviderConfig):
        return httpx.Client(timeout=provider_config.timeout)

    provider_factory = ProviderFactory(http_client_factory)
    models = [
        OpenAIProvider.DEFAULT_MODEL_CONFIG,
        AnthropicProvider.DEFAULT_MODEL_CONFIG,
        GoogleProvider.DEFAULT_MODEL_CONFIG,
    ]
    routing_engine = RoutingEngine(
        default_complexity_estimator(), ModelSelector(BalancedStrategy()), models
    )
    provider_configs = {
        "openai": ProviderConfig(api_key="sk-test", base_url="https://api.openai.com"),
    }
    tracker = _TrackerStub()
    middleware = MiddlewareChain([])

    router = DIContainer.create_custom_router(
        config=config,
        provider_factory=provider_factory,
        routing_engine=routing_engine,
        provider_configs=provider_configs,
        tracker=tracker,
        middleware=middleware,
    )

    assert router.analytics is tracker
