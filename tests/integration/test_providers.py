import json

import httpx
import pytest

from model_router.domain.models import Request
from model_router.providers.base import ProviderConfig
from model_router.providers.openai_provider import OpenAIProvider
from model_router.providers.anthropic_provider import AnthropicProvider
from model_router.providers.google_provider import GoogleProvider
from model_router.domain.exceptions import (
    ProviderRateLimitError,
    ProviderUnavailableError,
    ProviderError,
)


def _build_client(handler):
    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport)


def test_openai_provider_maps_response_successfully(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["headers"] = dict(request.headers)
        data = {
            "choices": [
                {
                    "message": {"content": "Hello there"},
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }
        return httpx.Response(200, json=data)

    client = _build_client(handler)
    config = ProviderConfig(api_key="sk-test", base_url="https://api.openai.com")
    provider = OpenAIProvider(client, config)
    request = Request(prompt="Hello", params={"temperature": 0.7})

    response = provider.complete(request)

    assert response.content == "Hello there"
    assert response.tokens == 30
    assert response.cost == pytest.approx((30 / 1000) * provider.get_pricing().pricing)
    assert captured["body"]["messages"][0]["content"] == "Hello"
    assert captured["body"]["temperature"] == 0.7
    assert captured["headers"]["authorization"].startswith("Bearer ")


def test_openai_provider_raises_on_rate_limit():
    def handler(request: httpx.Request) -> httpx.Response:
        data = {"error": {"message": "Too many requests"}}
        return httpx.Response(429, json=data)

    client = _build_client(handler)
    config = ProviderConfig(api_key="sk-test", base_url="https://api.openai.com")
    provider = OpenAIProvider(client, config)

    with pytest.raises(ProviderRateLimitError):
        provider.complete(Request(prompt="Hi"))


def test_openai_provider_raises_on_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "Service down"}})

    client = _build_client(handler)
    config = ProviderConfig(api_key="sk-test", base_url="https://api.openai.com")
    provider = OpenAIProvider(client, config)

    with pytest.raises(ProviderUnavailableError):
        provider.complete(Request(prompt="Hi"))


def test_openai_provider_raises_on_client_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "Bad request"}})

    client = _build_client(handler)
    config = ProviderConfig(api_key="sk-test", base_url="https://api.openai.com")
    provider = OpenAIProvider(client, config)

    with pytest.raises(ProviderError):
        provider.complete(Request(prompt="Hi"))


def test_anthropic_provider_maps_response_successfully():
    def handler(request: httpx.Request) -> httpx.Response:
        data = {
            "content": [
                {"type": "text", "text": "Line one"},
                {"type": "text", "text": "Line two"},
            ],
            "usage": {"output_tokens": 200},
        }
        return httpx.Response(200, json=data)

    client = _build_client(handler)
    config = ProviderConfig(api_key="ak-test", base_url="https://api.anthropic.com")
    provider = AnthropicProvider(client, config)

    response = provider.complete(Request(prompt="Explain", params={"temperature": 0.2}))

    assert "Line one" in response.content
    assert response.tokens == 200
    assert response.cost == pytest.approx((200 / 1000) * provider.get_pricing().pricing)


def test_anthropic_provider_handles_rate_limit():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "Too many"}})

    client = _build_client(handler)
    config = ProviderConfig(api_key="ak-test", base_url="https://api.anthropic.com")
    provider = AnthropicProvider(client, config)

    with pytest.raises(ProviderRateLimitError):
        provider.complete(Request(prompt="Hi"))


def test_anthropic_provider_handles_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "Down"}})

    client = _build_client(handler)
    config = ProviderConfig(api_key="ak-test", base_url="https://api.anthropic.com")
    provider = AnthropicProvider(client, config)

    with pytest.raises(ProviderUnavailableError):
        provider.complete(Request(prompt="Hi"))


def test_anthropic_provider_handles_client_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "Bad"}})

    client = _build_client(handler)
    config = ProviderConfig(api_key="ak-test", base_url="https://api.anthropic.com")
    provider = AnthropicProvider(client, config)

    with pytest.raises(ProviderError):
        provider.complete(Request(prompt="Hi"))


def test_google_provider_maps_response_successfully():
    def handler(request: httpx.Request) -> httpx.Response:
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Gemini output line"},
                        ]
                    },
                }
            ],
            "usageMetadata": {"totalTokenCount": 150},
        }
        return httpx.Response(200, json=data)

    client = _build_client(handler)
    config = ProviderConfig(
        api_key="gk-test", base_url="https://generativelanguage.googleapis.com"
    )
    provider = GoogleProvider(client, config)

    response = provider.complete(Request(prompt="Draw", params={"temperature": 0.1}))

    assert "Gemini output line" in response.content
    assert response.tokens == 150
    assert response.cost == pytest.approx((150 / 1000) * provider.get_pricing().pricing)


def test_google_provider_handles_rate_limit():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "Too many"}})

    client = _build_client(handler)
    config = ProviderConfig(
        api_key="gk-test", base_url="https://generativelanguage.googleapis.com"
    )
    provider = GoogleProvider(client, config)

    with pytest.raises(ProviderRateLimitError):
        provider.complete(Request(prompt="Hi"))


def test_google_provider_handles_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "Down"}})

    client = _build_client(handler)
    config = ProviderConfig(
        api_key="gk-test", base_url="https://generativelanguage.googleapis.com"
    )
    provider = GoogleProvider(client, config)

    with pytest.raises(ProviderUnavailableError):
        provider.complete(Request(prompt="Hi"))


def test_google_provider_handles_client_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "Bad"}})

    client = _build_client(handler)
    config = ProviderConfig(
        api_key="gk-test", base_url="https://generativelanguage.googleapis.com"
    )
    provider = GoogleProvider(client, config)

    with pytest.raises(ProviderError):
        provider.complete(Request(prompt="Hi"))
