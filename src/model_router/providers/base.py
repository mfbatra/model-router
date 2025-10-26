"""Provider abstractions and shared behavior implementations."""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from model_router.domain.exceptions import (
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from model_router.domain.models import ModelConfig, Provider, Request, Response


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration values shared by all provider adapters."""

    api_key: str
    base_url: str
    timeout: float = 30.0
    max_retries: int = 3
    backoff_factor: float = 0.5

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("api_key must be provided")
        if not self.base_url:
            raise ValueError("base_url must be provided")
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if self.backoff_factor <= 0:
            raise ValueError("backoff_factor must be greater than zero")


class BaseProvider(ABC):
    """Template-method base class that handles retries and logging."""

    PROVIDER_KEY = Provider.CUSTOM.value

    def __init__(
        self,
        config: ProviderConfig,
        model_config: ModelConfig,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self._model_config = model_config
        self.logger = logger or logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )

    def complete(self, request: Request) -> Response:
        """Public API that aligns with IProvider.complete."""

        return self.execute_request(request)

    def supports_streaming(self) -> bool:
        """Providers default to non-streaming output."""

        return False

    def get_pricing(self) -> ModelConfig:
        """Expose the immutable model configuration for routing decisions."""

        return self._model_config

    def execute_request(self, request: Request) -> Response:
        """Run request with retry/backoff when transient errors occur."""

        for attempt in range(self.config.max_retries + 1):
            self.log_request(request, attempt)
            try:
                response = self._make_api_call(request)
                self.log_response(response)
                return response
            except ProviderRateLimitError as exc:
                if attempt == self.config.max_retries:
                    self.logger.error(
                        "Rate limit exhausted after retries", exc_info=exc
                    )
                    raise
                delay = self._backoff_delay(attempt)
                self.handle_rate_limit(exc, delay)
                self._sleep(delay)
            except ProviderUnavailableError as exc:
                if attempt == self.config.max_retries:
                    self.logger.error(
                        "Provider unavailable after retries", exc_info=exc
                    )
                    raise
                delay = self._backoff_delay(attempt)
                self.logger.warning("Provider unavailable, backing off", exc_info=exc)
                self._sleep(delay)
            except ProviderError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.exception("Unexpected provider failure")
                raise ProviderError(
                    "Unexpected provider failure",
                    context={"provider": self.__class__.__name__},
                ) from exc

        raise ProviderError("Failed to execute request")

    @abstractmethod
    def _make_api_call(self, request: Request) -> Response:
        """Provider-specific HTTP/API interaction implemented by subclasses."""

    def log_request(self, request: Request, attempt: int) -> None:
        """Hook for request logging/analytics prior to execution."""

        self.logger.debug(
            "provider_request",
            extra={
                "prompt_tokens": self.count_tokens(request.prompt),
                "attempt": attempt,
                "provider": self.__class__.__name__,
            },
        )

    def log_response(self, response: Response) -> None:
        """Hook for logging successful responses."""

        self.logger.debug(
            "provider_response",
            extra={
                "model": response.model_used,
                "latency": response.latency,
                "tokens": response.tokens,
                "provider": self.__class__.__name__,
            },
        )

    def handle_rate_limit(self, error: ProviderRateLimitError, delay: float) -> None:
        """Allow subclasses to plug additional behavior when rate limited."""

        self.logger.warning(
            "Rate limited; backing off",
            extra={
                "delay": delay,
                "provider": self.__class__.__name__,
                "context": getattr(error, "context", {}),
            },
        )

    def _backoff_delay(self, attempt: int) -> float:
        return self.config.backoff_factor * math.pow(2, attempt)

    def _sleep(self, delay: float) -> None:
        time.sleep(delay)

    @staticmethod
    def count_tokens(text: Optional[str]) -> int:
        """Crude token estimation based on whitespace splitting."""

        if not text:
            return 0
        return len(text.strip().split())

    def count_request_tokens(self, request: Request) -> int:
        tokens = self.count_tokens(request.prompt)
        for value in request.params.values():
            tokens += self.count_tokens(str(value))
        return tokens

    def count_response_tokens(self, response: Response) -> int:
        return response.tokens
