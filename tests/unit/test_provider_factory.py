import pytest

from model_router.domain.exceptions import ProviderError
from model_router.domain.models import ModelConfig, Provider
from model_router.providers.anthropic_provider import AnthropicProvider
from model_router.providers.base import BaseProvider, ProviderConfig
from model_router.providers.factory import ProviderFactory
from model_router.providers.google_provider import GoogleProvider
from model_router.providers.openai_provider import OpenAIProvider


class _DummyProvider(BaseProvider):
    DEFAULT_MODEL_CONFIG = ModelConfig(
        provider=Provider.CUSTOM,
        model_name="custom-model",
        pricing=0.1,
        capabilities=frozenset({"custom"}),
    )

    def __init__(self, http_client, config, *, model_config=None):
        self.http_client = http_client
        super().__init__(config, model_config or self.DEFAULT_MODEL_CONFIG)

    def _make_api_call(self, request):  # pragma: no cover - not exercised
        raise NotImplementedError


def _client_factory_stub(stats):
    def _factory(config):
        client = object()
        stats.append((config.api_key, client))
        return client

    return _factory


def test_factory_caches_by_provider_and_api_key():
    stats = []
    factory = ProviderFactory(_client_factory_stub(stats))

    config = ProviderConfig(api_key="key-a", base_url="https://api.openai.com")
    provider_a = factory.create("gpt-4", config)
    provider_b = factory.create("gpt-4o", config)

    assert isinstance(provider_a, OpenAIProvider)
    assert provider_a is provider_b
    assert len(stats) == 1  # client created once

    config_b = ProviderConfig(api_key="key-b", base_url="https://api.openai.com")
    provider_c = factory.create("gpt-4", config_b)
    assert provider_c is not provider_a
    assert len(stats) == 2


def test_factory_detects_multiple_builtin_providers():
    factory = ProviderFactory(_client_factory_stub([]))

    openai = factory.create(
        "gpt-4o", ProviderConfig(api_key="o", base_url="https://api.openai.com")
    )
    claude = factory.create(
        "claude-3", ProviderConfig(api_key="a", base_url="https://api.anthropic.com")
    )
    gemini = factory.create(
        "gemini-1.5-pro",
        ProviderConfig(
            api_key="g", base_url="https://generativelanguage.googleapis.com"
        ),
    )

    assert isinstance(openai, OpenAIProvider)
    assert isinstance(claude, AnthropicProvider)
    assert isinstance(gemini, GoogleProvider)


def test_factory_supports_custom_registration():
    factory = ProviderFactory(_client_factory_stub([]))
    factory.register_provider(r"^custom-", _DummyProvider)

    provider = factory.create(
        "custom-model-x", ProviderConfig(api_key="c", base_url="https://custom")
    )

    assert isinstance(provider, _DummyProvider)


def test_factory_raises_for_unknown_model():
    factory = ProviderFactory(_client_factory_stub([]))

    with pytest.raises(ProviderError):
        factory.create(
            "unknown-model", ProviderConfig(api_key="k", base_url="https://example.com")
        )
