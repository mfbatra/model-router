"""Provider factory that wires HTTP clients to concrete adapters."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Callable, Dict, List, Pattern, Tuple, Type, cast

import httpx

from model_router.domain.exceptions import ProviderError
from model_router.domain.interfaces import IProvider
from model_router.domain.models import ModelConfig

from .base import ProviderConfig
from .anthropic_provider import AnthropicProvider
from .google_provider import GoogleProvider
from .openai_provider import OpenAIProvider


class ProviderFactory:
    """Factory that caches provider instances per (provider, api_key)."""

    def __init__(self, http_client_factory: Callable[[ProviderConfig], httpx.Client]):
        self._http_client_factory = http_client_factory
        self._registry: List[Tuple[Pattern[str], Type[IProvider]]] = []
        self._cache: Dict[Tuple[Type[IProvider], str], IProvider] = {}
        self._register_defaults()

    def create(self, model_name: str, config: ProviderConfig) -> IProvider:
        provider_cls = self._detect_provider(model_name)
        cache_key = (provider_cls, config.api_key)
        if cache_key in self._cache:
            return self._cache[cache_key]

        http_client = self._http_client_factory(config)
        model_config = self._build_model_config(provider_cls, model_name)
        kwargs = {"model_config": model_config} if model_config is not None else {}
        constructor = cast(Callable[..., IProvider], provider_cls)
        instance = constructor(http_client, config, **kwargs)
        self._cache[cache_key] = instance
        return instance

    def register_provider(self, pattern: str, provider_class: Type[IProvider]) -> None:
        compiled = re.compile(pattern, re.IGNORECASE)
        self._registry.append((compiled, provider_class))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _register_defaults(self) -> None:
        self.register_provider(r"^(gpt|text-davinci|o1)", OpenAIProvider)
        self.register_provider(r"^(claude)", AnthropicProvider)
        self.register_provider(r"^(gemini|palm)", GoogleProvider)

    def _detect_provider(self, model_name: str) -> Type[IProvider]:
        for pattern, provider_cls in self._registry:
            if pattern.search(model_name):
                return provider_cls
        raise ProviderError(f"No provider registered for model '{model_name}'")

    def _build_model_config(
        self, provider_cls: Type[IProvider], model_name: str
    ) -> ModelConfig | None:
        base_config = cast(
            ModelConfig | None, getattr(provider_cls, "DEFAULT_MODEL_CONFIG", None)
        )
        if base_config is None:
            return None
        if base_config.model_name == model_name:
            return base_config
        return replace(base_config, model_name=model_name)
