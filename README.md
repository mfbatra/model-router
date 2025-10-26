<div align="center">

# Model Router

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/mfbatra/model-router/actions)
[![Coverage](https://img.shields.io/badge/coverage-90%25-blue.svg)](#)
[![License](https://img.shields.io/badge/license-MIT-black.svg)](LICENSE)

</div>

## Quick Start

```python
from model_router.core.container import DIContainer
router = DIContainer.create_router(openai_key="sk-your-key")
print(router.complete("Summarize the request").content)
```

## Installation

```bash
git clone https://github.com/mfbatra/model-router
cd model-router
poetry install
```

## Basic Usage Examples

```python
router = DIContainer.create_router(openai_key="sk-test")

response = router.complete("Top 5 databases in 2024", max_cost=0.05)
print(response.content)

chat = router.chat(
    [
        {"role": "user", "content": "Help me deploy a service"},
        {"role": "assistant", "content": "What's the stack?"}
    ]
)
print(chat.content)
```

## Advanced Usage

- **Constraints**: `router.complete(prompt, max_cost=0.1, max_latency=500, min_quality=0.8)`
- **Fallback Chain**: Configure via `RouterConfig(fallback_models=["gpt-3.5", "claude-3"])`
- **Analytics**: Access `router.analytics.get_summary("last_7_days")`
- **Custom Middleware**: Inject `MiddlewareChain([...])` via `DIContainer.create_custom_router`

## Comparing LLMs & Benchmarking

### Multi-turn chat with guardrails

```python
response = router.chat(
    [
        {"role": "user", "content": "I need to deploy a microservice"},
        {"role": "assistant", "content": "What technology stack are you using?"},
        {"role": "user", "content": "Python FastAPI with PostgreSQL"}
    ],
    max_cost=0.03,
    min_quality=0.75,
)
print(response.content)
```

### Force specific models for side-by-side comparisons

```python
import pandas as pd
from model_router.core.container import DIContainer

router = DIContainer.create_router(
    openai_api_key="sk-...",
    anthropic_api_key="sk-ant-...",
    google_api_key="..."
)

def compare_providers(prompt: str, models: list[str]) -> pd.DataFrame:
    """Compare the same prompt across different models."""
    rows = []
    for model in models:
        response = router.complete(prompt, metadata={"force_model": model})
        rows.append(
            {
                "Model": model,
                "Response": response.content[:100] + "...",
                "Cost": f"${response.cost:.4f}",
                "Latency": f"{response.latency * 1000:.0f}ms",
                "Tokens": response.tokens,
            }
        )
    return pd.DataFrame(rows)

df = compare_providers(
    "Explain quantum computing in simple terms",
    ["gpt-3.5-turbo", "gpt-4-turbo", "claude-sonnet-3.5", "gemini-pro"],
)
print(df.to_markdown(index=False))
```

### Track cost vs latency across a test suite

```python
def benchmark(prompts: list[str], targets: list[tuple[str, str]]):
    results = {model: {"cost": 0.0, "latency": 0.0, "runs": 0} for _, model in targets}
    for prompt in prompts:
        for provider_key, model_name in targets:
            provider = factory.create(model_name, configs[provider_key])
            response = provider.complete(Request(prompt=prompt))
            stats = results[model_name]
            stats["cost"] += response.cost
            stats["latency"] += response.latency
            stats["runs"] += 1
    for model, stats in results.items():
        stats["avg_cost"] = stats["cost"] / stats["runs"]
        stats["avg_latency"] = stats["latency"] / stats["runs"]
    return results

bench = benchmark(
    [
        "Translate 'hello' to French",
        "Summarize the causes of World War II",
        "Write a Python function that reverses a list",
    ],
    [("openai", "gpt-4o"), ("anthropic", "claude-3")],
)
print(bench)
```

### Simple A/B harness

```python
def ab_test(router, prompt: str, model_a: str, model_b: str, provider_key="openai"):
    provider = router._provider_factory
    config = router._provider_configs[provider_key]
    response_a = provider.create(model_a, config).complete(Request(prompt=prompt))
    response_b = provider.create(model_b, config).complete(Request(prompt=prompt))
    return {
        model_a: {"cost": response_a.cost, "latency": response_a.latency},
        model_b: {"cost": response_b.cost, "latency": response_b.latency},
    }

print(ab_test(router, "Describe a scalable e-commerce platform", "gpt-4o", "gpt-4o-mini"))
```

Use these snippets to build richer reports (DataFrames, Matplotlib charts, or analytics exports via `router.analytics.to_dataframe()`).

## Configuration Options

| Option | Env | Description |
| --- | --- | --- |
| `default_strategy` | `ROUTER_DEFAULT_STRATEGY` | `balanced`, `cost_optimized`, etc. |
| `enable_analytics` | `ROUTER_ENABLE_ANALYTICS` | Toggle UsageTracker |
| `enable_cache` | `ROUTER_ENABLE_CACHE` | Enable middleware caching |
| `fallback_models` | `ROUTER_FALLBACK_MODELS` | Comma-separated list |
| `max_retries` | `ROUTER_MAX_RETRIES` | Provider retry attempts |
| `timeout_seconds` | `ROUTER_TIMEOUT_SECONDS` | Per-request timeout |

Load from env (`RouterConfig.from_env()`) or JSON/YAML file (`RouterConfig.from_file("config.yaml")`).

## Architecture Overview

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for clean architecture diagrams, SOLID mappings, and sequence flows.

## Contributing Guide

1. Fork the repo and create a feature branch.
2. `poetry install && poetry run pytest`
3. Follow the existing coding style (Black, Ruff, Mypy).
4. Add tests for new code and update docs if behavior changes.
5. Open a PR describing changes and testing steps.
