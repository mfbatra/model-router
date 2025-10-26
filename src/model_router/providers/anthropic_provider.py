"""Anthropic provider adapter that reuses BaseProvider template hooks."""

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


ANTHROPIC_MESSAGES_PATH = "/v1/messages"


DEFAULT_ANTHROPIC_MODEL = ModelConfig(
    provider=Provider.ANTHROPIC,
    model_name="claude-3-opus",
    pricing=0.015,
    capabilities=frozenset({"chat", "reasoning", "code"}),
)


class AnthropicProvider(BaseProvider):
    """Concrete adapter around Anthropic's Messages API."""

    DEFAULT_MODEL_CONFIG = DEFAULT_ANTHROPIC_MODEL

    def __init__(
        self,
        http_client: httpx.Client,
        config: ProviderConfig,
        *,
        model_config: Optional[ModelConfig] = None,
    ) -> None:
        super().__init__(config, model_config or self.DEFAULT_MODEL_CONFIG)
        self._http = http_client
        self._endpoint = f"{self.config.base_url.rstrip('/')}{ANTHROPIC_MESSAGES_PATH}"

    def _make_api_call(self, request: Request) -> Response:
        payload = self._build_payload(request)
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
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
            "max_tokens": request.params.get("max_tokens", 1024),
        }
        extra_params = {k: v for k, v in request.params.items() if k != "max_tokens"}
        payload.update(extra_params)
        return payload

    def _map_response(self, http_response: httpx.Response) -> Response:
        status = http_response.status_code
        data = http_response.json()

        if status == 429:
            raise ProviderRateLimitError(
                "Anthropic rate limit exceeded",
                context={"status_code": status},
            )
        if status >= 500 or status == 503:
            raise ProviderUnavailableError(
                "Anthropic service unavailable",
                context={"status_code": status},
            )
        if status >= 400:
            raise ProviderError(
                data.get("error", {}).get("message", "Anthropic request failed"),
                context={"status_code": status},
            )

        content = self._extract_content(data)
        usage = data.get("usage", {})
        latency = self._safe_elapsed(http_response)
        tokens = int(usage.get("output_tokens", usage.get("output", 0)) or 0)
        cost = self._estimate_cost(tokens)

        return Response(
            content=content,
            model_used=self._model_config.model_name,
            cost=cost,
            latency=latency,
            tokens=tokens,
        )

    def _extract_content(self, data: Dict[str, Any]) -> str:
        try:
            content = data["content"]
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                return "\n".join(filter(None, parts))
            if isinstance(content, str):
                return content
        except KeyError as exc:  # pragma: no cover - defensive
            raise ProviderError(
                "Malformed Anthropic response", context={"data": data}
            ) from exc
        raise ProviderError("Malformed Anthropic response", context={"data": data})

    def _estimate_cost(self, total_tokens: int) -> float:
        return round((total_tokens / 1000) * self._model_config.pricing, 6)

    @staticmethod
    def _safe_elapsed(http_response: httpx.Response) -> float:
        try:
            elapsed = http_response.elapsed
        except RuntimeError:
            return 0.0
        return elapsed.total_seconds() if elapsed else 0.0
