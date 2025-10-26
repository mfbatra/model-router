"""OpenAI provider adapter built on top of ``BaseProvider``."""

from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from model_router.domain.exceptions import (
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from model_router.domain.models import ModelConfig, Provider, Request, Response

from .base import BaseProvider, ProviderConfig


OPENAI_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"


DEFAULT_CHAT_MODEL = ModelConfig(
    provider=Provider.OPENAI,
    model_name="gpt-4",
    pricing=0.06,  # assumed cost per 1K tokens for example purposes
    capabilities=frozenset({"chat", "reasoning", "code"}),
)


class OpenAIProvider(BaseProvider):
    """Concrete provider that speaks to OpenAI's chat completions API."""

    DEFAULT_MODEL_CONFIG = DEFAULT_CHAT_MODEL

    def __init__(
        self,
        http_client: httpx.Client,
        config: ProviderConfig,
        *,
        model_config: Optional[ModelConfig] = None,
    ) -> None:
        super().__init__(config, model_config or self.DEFAULT_MODEL_CONFIG)
        self._http = http_client
        self._endpoint = (
            f"{self.config.base_url.rstrip('/')}{OPENAI_CHAT_COMPLETIONS_PATH}"
        )

    def _make_api_call(self, request: Request) -> Response:
        payload = self._build_payload(request)
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        http_response = self._http.post(
            self._endpoint,
            json=payload,
            timeout=self.config.timeout,
            headers=headers,
        )

        return self._map_response(http_response)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_payload(self, request: Request) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self._model_config.model_name,
            "messages": [
                {"role": "user", "content": request.prompt},
            ],
        }
        payload.update(request.params)
        return payload

    def _map_response(self, http_response: httpx.Response) -> Response:
        status = http_response.status_code
        data = http_response.json()

        if status == 429:
            raise ProviderRateLimitError(
                "OpenAI rate limit exceeded",
                context={"status_code": status, "headers": dict(http_response.headers)},
            )
        if status >= 500 or status == 503:
            raise ProviderUnavailableError(
                "OpenAI service unavailable",
                context={"status_code": status},
            )
        if status >= 400:
            raise ProviderError(
                data.get("error", {}).get("message", "OpenAI request failed"),
                context={"status_code": status},
            )

        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:  # pragma: no cover - defensive
            raise ProviderError(
                "Malformed OpenAI response", context={"data": data}
            ) from exc

        usage = data.get("usage", {})
        latency = self._safe_elapsed(http_response)
        tokens = int(usage.get("total_tokens", usage.get("completion_tokens", 0)) or 0)
        cost = self._estimate_cost(tokens)

        return Response(
            content=content,
            model_used=self._model_config.model_name,
            cost=cost,
            latency=latency,
            tokens=tokens,
        )

    def _estimate_cost(self, total_tokens: int) -> float:
        return round((total_tokens / 1000) * self._model_config.pricing, 6)

    @staticmethod
    def _safe_elapsed(http_response: httpx.Response) -> float:
        try:
            elapsed = http_response.elapsed
        except RuntimeError:
            return 0.0
        return elapsed.total_seconds() if elapsed else 0.0
