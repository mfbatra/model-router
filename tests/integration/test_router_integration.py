import pytest

from model_router.core.config import RouterConfig
from model_router.core.middleware import IMiddleware, MiddlewareChain
from model_router.core.router import Router
from model_router.domain.interfaces import IUsageTracker
from model_router.domain.models import (
    ModelConfig,
    Provider,
    Request,
    Response,
    RoutingConstraints,
    RoutingDecision,
)
from model_router.providers.base import ProviderConfig
from model_router.routing.engine import RoutingEngine

pytestmark = pytest.mark.integration


class MockProvider:
    def __init__(self, response: Response | None = None, should_fail: bool = False):
        self.response = response or Response(
            content="ok",
            model_used="gpt-4",
            cost=0.02,
            latency=0.2,
            tokens=20,
        )
        self.should_fail = should_fail
        self.calls = 0

    def complete(self, request: Request) -> Response:
        self.calls += 1
        if self.should_fail:
            raise RuntimeError("provider failure")
        return self.response

    def supports_streaming(self) -> bool:
        return False

    def get_pricing(self) -> ModelConfig:
        return ModelConfig(
            provider=Provider.OPENAI,
            model_name=self.response.model_used,
            pricing=0.02,
            capabilities=frozenset({"chat"}),
        )


class ProviderFactoryStub:
    def __init__(self, providers: dict[str, MockProvider]):
        self.providers = providers
        self.calls: list[tuple[str, ProviderConfig]] = []

    def create(self, model_name: str, config: ProviderConfig):
        self.calls.append((model_name, config))
        return self.providers[model_name]


class RoutingEngineStub:
    def __init__(self, decision: RoutingDecision):
        self.decision = decision
        self.calls: list[RoutingConstraints] = []

    def route(
        self, request: Request, constraints: RoutingConstraints
    ) -> RoutingDecision:
        self.calls.append(constraints)
        return self.decision


class TrackerStub(IUsageTracker):
    def __init__(self):
        self.tracks = 0

    def track(self, request: Request, response: Response) -> None:
        self.tracks += 1

    def get_summary(self, period: str = "last_7_days"):
        raise NotImplementedError

    def to_dataframe(self):
        raise NotImplementedError


class RecordingMiddleware(IMiddleware):
    def __init__(self, name: str, log: list[str]):
        self.name = name
        self.log = log

    def process_request(self, request: Request) -> Request:
        self.log.append(f"req:{self.name}")
        return request

    def process_response(self, response: Response) -> Response:
        self.log.append(f"res:{self.name}")
        return response


def _model(model_name: str, provider: Provider = Provider.OPENAI) -> ModelConfig:
    return ModelConfig(
        provider=provider,
        model_name=model_name,
        pricing=0.02,
        capabilities=frozenset({"chat"}),
    )


def _build_router(
    *,
    decision: RoutingDecision,
    providers: dict[str, MockProvider],
    config: RouterConfig | None = None,
    tracker: TrackerStub | None = None,
    middleware: MiddlewareChain | None = None,
) -> tuple[Router, ProviderFactoryStub, RoutingEngineStub, TrackerStub]:
    provider_config = ProviderConfig(api_key="key", base_url="https://api.mock")
    factory = ProviderFactoryStub(providers)
    engine = RoutingEngineStub(decision)
    tracker = tracker or TrackerStub()
    router = Router(
        config=config or RouterConfig(),
        provider_factory=factory,
        routing_engine=engine,
        provider_configs={decision.selected_model.provider.value: provider_config},
        tracker=tracker,
        middleware=middleware or MiddlewareChain([]),
    )
    return router, factory, engine, tracker


def test_complete_workflow_openai():
    decision = RoutingDecision(
        selected_model=_model("gpt-4"),
        estimated_cost=0.02,
        reasoning="ok",
        alternatives_considered=(),
    )
    provider = MockProvider()
    router, _, engine, tracker = _build_router(
        decision=decision, providers={"gpt-4": provider}
    )

    response = router.complete("Hello world")

    assert response.model_used == "gpt-4"
    assert provider.calls == 1
    assert len(engine.calls) == 1
    assert tracker.tracks == 1


def test_fallback_chain():
    decision = RoutingDecision(
        selected_model=_model("gpt-4"),
        estimated_cost=0.02,
        reasoning="ok",
        alternatives_considered=(),
    )
    primary = MockProvider(should_fail=True)
    fallback = MockProvider(
        response=Response(
            content="fallback", model_used="gpt-3.5", cost=0.01, latency=0.1, tokens=10
        )
    )
    config = RouterConfig(fallback_models=["gpt-3.5"], max_retries=2)
    router, factory, _, _ = _build_router(
        decision=decision,
        providers={"gpt-4": primary, "gpt-3.5": fallback},
        config=config,
    )

    response = router.complete("Hello world")

    assert response.model_used == "gpt-3.5"
    assert primary.calls == 1
    assert fallback.calls == 1
    assert factory.calls[0][0] == "gpt-4"


def test_analytics_tracking():
    decision = RoutingDecision(
        selected_model=_model("gpt-4"),
        estimated_cost=0.02,
        reasoning="ok",
        alternatives_considered=(),
    )
    tracker = TrackerStub()
    router, _, _, tracker = _build_router(
        decision=decision, providers={"gpt-4": MockProvider()}, tracker=tracker
    )

    router.complete("Track me")

    assert tracker.tracks == 1


def test_middleware_chain():
    decision = RoutingDecision(
        selected_model=_model("gpt-4"),
        estimated_cost=0.02,
        reasoning="ok",
        alternatives_considered=(),
    )
    log: list[str] = []
    middleware = MiddlewareChain(
        [
            RecordingMiddleware("a", log),
            RecordingMiddleware("b", log),
        ]
    )
    router, _, _, _ = _build_router(
        decision=decision,
        providers={"gpt-4": MockProvider()},
        middleware=middleware,
    )

    router.complete("Hello")

    assert log == ["req:a", "req:b", "res:b", "res:a"]


def test_cost_constraints_enforced():
    decision = RoutingDecision(
        selected_model=_model("gpt-4"),
        estimated_cost=0.02,
        reasoning="ok",
        alternatives_considered=(),
    )
    router, _, engine, _ = _build_router(
        decision=decision, providers={"gpt-4": MockProvider()}
    )

    router.complete("Budget", max_cost=0.05)

    assert engine.calls[0].max_cost == 0.05
    assert engine.calls[0].max_latency == 30000
