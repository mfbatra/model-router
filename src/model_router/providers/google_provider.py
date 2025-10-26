"""Google Gemini provider adapter leveraging BaseProvider template."""

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


GEMINI_GENERATE_PATH_TEMPLATE = "/v1beta/models/{model}:generateContent"


DEFAULT_GEMINI_MODEL = ModelConfig(
    provider=Provider.GOOGLE,
    model_name="gemini-1.5-pro",
    pricing=0.01,
    capabilities=frozenset({"chat", "vision", "reasoning"}),
)


class GoogleProvider(BaseProvider):
    """Adapter for Google Gemini APIs with httpx transport injection."""

    DEFAULT_MODEL_CONFIG = DEFAULT_GEMINI_MODEL

    def __init__(
        self,
        http_client: httpx.Client,
        config: ProviderConfig,
        *,
        model_config: Optional[ModelConfig] = None,
    ) -> None:
        super().__init__(config, model_config or self.DEFAULT_MODEL_CONFIG)
        self._http = http_client

    def _make_api_call(self, request: Request) -> Response:
        payload = self._build_payload(request)
        endpoint = self._endpoint_for_model(self._model_config.model_name)
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.config.api_key,
        }

        http_response = self._http.post(
            endpoint,
            json=payload,
            timeout=self.config.timeout,
            headers=headers,
        )

        return self._map_response(http_response)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _endpoint_for_model(self, model_name: str) -> str:
        path = GEMINI_GENERATE_PATH_TEMPLATE.format(model=model_name)
        return f"{self.config.base_url.rstrip('/')}{path}"

    def _build_payload(self, request: Request) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": request.prompt},
                    ],
                }
            ],
        }
        payload.update(request.params)
        return payload

    def _map_response(self, http_response: httpx.Response) -> Response:
        status = http_response.status_code
        data = http_response.json()

        if status == 429:
            raise ProviderRateLimitError(
                "Google Gemini rate limit exceeded",
                context={"status_code": status},
            )
        if status >= 500 or status == 503:
            raise ProviderUnavailableError(
                "Google Gemini service unavailable",
                context={"status_code": status},
            )
        if status >= 400:
            message = data.get("error", {}).get(
                "message", "Google Gemini request failed"
            )
            raise ProviderError(message, context={"status_code": status})

        content = self._extract_content(data)
        usage = data.get("usageMetadata", {})
        latency = self._safe_elapsed(http_response)
        tokens = int(
            usage.get("totalTokenCount", usage.get("candidatesTokenCount", 0)) or 0
        )
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
            candidate = data["candidates"][0]
            content = candidate["content"]
            parts = content.get("parts", []) if isinstance(content, dict) else []
            texts = []
            for part in parts:
                if isinstance(part, dict) and "text" in part:
                    texts.append(part["text"])
            return "\n".join(filter(None, texts))
        except (KeyError, IndexError, TypeError) as exc:  # pragma: no cover - defensive
            raise ProviderError(
                "Malformed Google Gemini response", context={"data": data}
            ) from exc

    def _estimate_cost(self, total_tokens: int) -> float:
        return round((total_tokens / 1000) * self._model_config.pricing, 6)

    @staticmethod
    def _safe_elapsed(http_response: httpx.Response) -> float:
        try:
            elapsed = http_response.elapsed
        except RuntimeError:
            return 0.0
        return elapsed.total_seconds() if elapsed else 0.0
