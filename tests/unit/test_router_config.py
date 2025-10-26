import json
import os
from pathlib import Path

import pytest

from model_router.core.config import RouterConfig


def test_router_config_defaults():
    config = RouterConfig()
    assert config.default_strategy == "balanced"
    assert config.enable_analytics is True
    assert config.enable_cache is False
    assert config.max_retries == 3


def test_router_config_from_env(monkeypatch):
    monkeypatch.setenv("ROUTER_DEFAULT_STRATEGY", "cost_optimized")
    monkeypatch.setenv("ROUTER_ENABLE_ANALYTICS", "false")
    monkeypatch.setenv("ROUTER_ENABLE_CACHE", "1")
    monkeypatch.setenv("ROUTER_FALLBACK_MODELS", "model-a,model-b")
    monkeypatch.setenv("ROUTER_MAX_RETRIES", "5")
    monkeypatch.setenv("ROUTER_TIMEOUT_SECONDS", "45")

    config = RouterConfig.from_env()

    assert config.default_strategy == "cost_optimized"
    assert config.enable_analytics is False
    assert config.enable_cache is True
    assert config.fallback_models == ["model-a", "model-b"]
    assert config.max_retries == 5
    assert config.timeout_seconds == 45


def test_router_config_from_file_json(tmp_path: Path):
    data = {
        "default_strategy": "quality_optimized",
        "enable_analytics": False,
        "fallback_models": ["m1"],
        "max_retries": 10,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))

    config = RouterConfig.from_file(str(path))

    assert config.default_strategy == "quality_optimized"
    assert config.enable_analytics is False
    assert config.fallback_models == ["m1"]
    assert config.max_retries == 10


def test_router_config_from_file_yaml(tmp_path: Path):
    yaml = pytest.importorskip("yaml")
    data = {
        "default_strategy": "latency_optimized",
        "enable_cache": True,
        "timeout_seconds": 60,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))

    config = RouterConfig.from_file(str(path))

    assert config.default_strategy == "latency_optimized"
    assert config.enable_cache is True
    assert config.timeout_seconds == 60


def test_router_config_validate_rejects_invalid_strategy():
    with pytest.raises(ValueError):
        RouterConfig(default_strategy="unknown")
