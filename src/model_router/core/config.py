"""Router configuration management helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


def _str_to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _str_to_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid integer value: {value}") from exc


@dataclass(frozen=True)
class RouterConfig:
    """Immutable configuration object loaded from env or files."""

    default_strategy: str = "balanced"
    enable_analytics: bool = True
    enable_cache: bool = False
    fallback_models: List[str] = field(default_factory=list)
    max_retries: int = 3
    timeout_seconds: int = 30

    _ALLOWED_STRATEGIES = {
        "balanced",
        "cost_optimized",
        "quality_optimized",
        "latency_optimized",
    }

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_env(cls) -> "RouterConfig":
        defaults = cls()
        fallback_raw = os.getenv("ROUTER_FALLBACK_MODELS")
        fallback = (
            [model.strip() for model in fallback_raw.split(",") if model.strip()]
            if fallback_raw
            else defaults.fallback_models
        )
        config = cls(
            default_strategy=os.getenv(
                "ROUTER_DEFAULT_STRATEGY", defaults.default_strategy
            ),
            enable_analytics=_str_to_bool(
                os.getenv("ROUTER_ENABLE_ANALYTICS"), defaults.enable_analytics
            ),
            enable_cache=_str_to_bool(
                os.getenv("ROUTER_ENABLE_CACHE"), defaults.enable_cache
            ),
            fallback_models=fallback,
            max_retries=_str_to_int(
                os.getenv("ROUTER_MAX_RETRIES"), defaults.max_retries
            ),
            timeout_seconds=_str_to_int(
                os.getenv("ROUTER_TIMEOUT_SECONDS"), defaults.timeout_seconds
            ),
        )
        return config

    @classmethod
    def from_file(cls, path: str) -> "RouterConfig":
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        raw = file_path.read_text()
        data: Dict[str, Any]
        suffix = file_path.suffix.lower()
        if suffix == ".json":
            data = json.loads(raw)
        elif suffix in {".yaml", ".yml"}:
            data = cls._load_yaml(raw)
        else:
            raise ValueError("Unsupported config format. Use JSON or YAML.")
        return cls(**cls._merge_with_defaults(data))

    def validate(self) -> None:
        if self.default_strategy not in self._ALLOWED_STRATEGIES:
            raise ValueError(
                f"default_strategy must be one of {sorted(self._ALLOWED_STRATEGIES)}"
            )
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if not isinstance(self.fallback_models, list):
            raise ValueError("fallback_models must be a list")

    @classmethod
    def _merge_with_defaults(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        defaults = cls()
        merged = {
            "default_strategy": data.get("default_strategy", defaults.default_strategy),
            "enable_analytics": data.get("enable_analytics", defaults.enable_analytics),
            "enable_cache": data.get("enable_cache", defaults.enable_cache),
            "fallback_models": data.get("fallback_models", defaults.fallback_models),
            "max_retries": data.get("max_retries", defaults.max_retries),
            "timeout_seconds": data.get("timeout_seconds", defaults.timeout_seconds),
        }
        return merged

    @staticmethod
    def _load_yaml(raw: str) -> Dict[str, Any]:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PyYAML is required to parse YAML config files") from exc
        return yaml.safe_load(raw) or {}
