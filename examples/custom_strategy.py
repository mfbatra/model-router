"""Demonstrates creating and registering a custom routing strategy."""

from model_router.core.container import DIContainer
from model_router.domain.models import ModelConfig, RoutingConstraints
from model_router.routing.strategies.base import IRoutingStrategy


class MemoryOptimizedStrategy(IRoutingStrategy):
    """Scores models that advertise low latency and moderate pricing."""

    def name(self) -> str:
        return "memory_optimized"

    def score_model(self, model: ModelConfig, complexity: float, constraints: RoutingConstraints) -> float:
        has_low_latency = "low-latency" in model.capabilities
        base = 0.7 if has_low_latency else 0.4
        cost_penalty = min(model.pricing / 0.1, 1.0) * 0.3
        complexity_penalty = complexity * 0.2
        return max(0.0, min(1.0, base - cost_penalty - complexity_penalty))


def main() -> None:
    container = DIContainer
    router = container.create_router(openai_key="sk-demo")
    selector = router._routing_engine._selector  # type: ignore[attr-defined]
    selector._strategy = MemoryOptimizedStrategy()  # type: ignore[attr-defined]

    response = router.complete("Find an optimal GPU model for memory-intensive workloads")
    print(response.content)


if __name__ == "__main__":
    main()
