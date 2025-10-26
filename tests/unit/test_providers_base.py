import logging

import pytest

from model_router.domain.exceptions import (
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from model_router.domain.models import ModelConfig, Provider, Request, Response
from model_router.providers.base import BaseProvider, ProviderConfig


class _SuccessfulProvider(BaseProvider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0
        self.logged_requests = []
        self.logged_responses = []
        self.rate_limit_delays = []

    def _make_api_call(self, request: Request) -> Response:
        self.calls += 1
        if self.calls == 1:
            raise ProviderRateLimitError(context={"retry_after": 1})
        return Response(
            content="ok",
            model_used=self.get_pricing().model_name,
            cost=0.02,
            latency=120,
            tokens=256,
        )

    def log_request(self, request: Request, attempt: int) -> None:
        self.logged_requests.append((request.prompt, attempt))

    def log_response(self, response: Response) -> None:
        self.logged_responses.append(response.model_used)

    def handle_rate_limit(self, error: ProviderRateLimitError, delay: float) -> None:
        super().handle_rate_limit(error, delay)
        self.rate_limit_delays.append(delay)


class _FailingProvider(BaseProvider):
    def _make_api_call(
        self, request: Request
    ) -> Response:  # pragma: no cover - simple stub
        raise ProviderUnavailableError("down")


class _HelperProvider(BaseProvider):
    def _make_api_call(self, request: Request) -> Response:  # pragma: no cover - unused
        raise NotImplementedError


@pytest.fixture
def provider_config() -> ProviderConfig:
    return ProviderConfig(
        api_key="test-key",
        base_url="https://api.provider.test",
        timeout=5,
        max_retries=2,
        backoff_factor=0.1,
    )


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        provider=Provider.OPENAI,
        model_name="gpt-4",
        pricing=0.02,
        capabilities=frozenset({"chat"}),
    )


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(BaseProvider, "_sleep", lambda self, delay: None)


@pytest.fixture(autouse=True)
def no_logging(monkeypatch):
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass

    logger = logging.getLogger("model_router.providers.base.BaseProvider")
    logger.setLevel(logging.CRITICAL)
    logger.handlers = [NullHandler()]


def test_execute_request_retries_on_rate_limit(provider_config, model_config):
    provider = _SuccessfulProvider(provider_config, model_config)
    request = Request(prompt="hello world")

    response = provider.complete(request)

    assert response.model_used == "gpt-4"
    assert provider.calls == 2
    assert provider.rate_limit_delays[0] == provider_config.backoff_factor
    assert provider.logged_requests[0][1] == 0  # first attempt
    assert provider.logged_responses == ["gpt-4"]


def test_execute_request_raises_after_unavailable(provider_config, model_config):
    provider = _FailingProvider(provider_config, model_config)
    request = Request(prompt="failure expected")

    with pytest.raises(ProviderUnavailableError):
        provider.complete(request)


def test_token_helpers(provider_config, model_config):
    provider = _HelperProvider(provider_config, model_config)
    request = Request(prompt="hello world", params={"temperature": 0.2})
    response = Response(
        content="result",
        model_used=model_config.model_name,
        cost=0,
        latency=0,
        tokens=128,
    )

    assert provider.count_request_tokens(request) == 3
    assert provider.count_response_tokens(response) == 128
    assert provider.count_tokens("one two three") == 3
