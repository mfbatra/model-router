import pytest

from model_router.core.config import RouterConfig
from model_router.core.middleware import MiddlewareChain
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


class _FakeProvider:
    def __init__(self, response: Response | None = None, should_fail: bool = False):
        self.response = response or Response(
            content="ok", model_used="gpt-4", cost=0.02, latency=0.2, tokens=10
        )
        self.should_fail = should_fail
        self.calls = 0

    def complete(self, request: Request) -> Response:
        self.calls += 1
        if self.should_fail:
            raise RuntimeError("provider failure")
        return self.response


class _ProviderFactoryStub:
    def __init__(self):
        self.providers = {}
        self.requests = []
        self.provider_keys = {}

    def register(self, model_name: str, provider, provider_key: str = "openai"):
        self.providers[model_name] = provider
        self.provider_keys[model_name] = provider_key

    def create(self, model_name: str, config: ProviderConfig):
        self.requests.append((model_name, config))
        return self.providers[model_name]

    def infer_provider_key(self, model_name: str) -> str:
        return self.provider_keys[model_name]


class _RoutingEngineStub:
    def __init__(self, decision: RoutingDecision):
        self.decision = decision
        self.calls = []

    def route(
        self, request: Request, constraints: RoutingConstraints
    ) -> RoutingDecision:
        self.calls.append((request, constraints))
        return self.decision


class _TrackerStub(IUsageTracker):
    def __init__(self):
        self.tracks = 0
        self.last = None

    def track(self, request: Request, response: Response) -> None:
        self.tracks += 1
        self.last = (request, response)

    def get_summary(self, period: str = "last_7_days"):
        raise NotImplementedError

    def to_dataframe(self):
        raise NotImplementedError


def _build_router(
    decision: RoutingDecision,
    tracker: IUsageTracker | None = None,
    config: RouterConfig | None = None,
):
    provider_config = ProviderConfig(api_key="key", base_url="https://api.openai.com")
    factory = _ProviderFactoryStub()
    provider = _FakeProvider(
        response=Response(
            content="ok",
            model_used=decision.selected_model.model_name,
            cost=0.02,
            latency=0.3,
            tokens=20,
        )
    )
    factory.register(decision.selected_model.model_name, provider)
    engine = _RoutingEngineStub(decision)
    config = config or RouterConfig()
    tracker = tracker or _TrackerStub()
    router = Router(
        config=config,
        provider_factory=factory,
        routing_engine=engine,
        provider_configs={decision.selected_model.provider.value: provider_config},
        tracker=tracker,
        middleware=MiddlewareChain([]),
    )
    return router, factory, provider, engine, tracker


def _model(name="gpt-4"):
    return ModelConfig(
        provider=Provider.OPENAI,
        model_name=name,
        pricing=0.02,
        capabilities=frozenset({"chat"}),
    )


def test_complete_routes_and_invokes_provider():
    decision = RoutingDecision(
        selected_model=_model(),
        estimated_cost=0.02,
        reasoning="ok",
        alternatives_considered=(),
    )
    router, factory, provider, engine, tracker = _build_router(decision)

    response = router.complete("Hello world")

    assert response.model_used == "gpt-4"
    assert provider.calls == 1
    assert len(engine.calls) == 1
    assert tracker.tracks == 1


def test_chat_converts_messages_to_prompt():
    decision = RoutingDecision(
        selected_model=_model(),
        estimated_cost=0.02,
        reasoning="ok",
        alternatives_considered=(),
    )
    router, *_ = _build_router(decision)

    response = router.chat(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]
    )

    assert response.content == "ok"


def test_configure_fallback_updates_config():
    decision = RoutingDecision(
        selected_model=_model(),
        estimated_cost=0.02,
        reasoning="ok",
        alternatives_considered=(),
    )
    router, *_ = _build_router(decision)

    router.configure_fallback(["gpt-3.5"])

    assert router._config.fallback_models == ["gpt-3.5"]


def test_fallback_attempts_other_models():
    model = _model()
    fallback_model = "gpt-3.5"
    decision = RoutingDecision(
        selected_model=model,
        estimated_cost=0.02,
        reasoning="ok",
        alternatives_considered=(),
    )
    provider_config = ProviderConfig(api_key="key", base_url="https://api.openai.com")
    factory = _ProviderFactoryStub()
    primary_provider = _FakeProvider(should_fail=True)
    fallback_provider = _FakeProvider(
        response=Response(
            content="fallback",
            model_used=fallback_model,
            cost=0.01,
            latency=0.1,
            tokens=5,
        )
    )
    factory.register(model.model_name, primary_provider)
    factory.register(fallback_model, fallback_provider)
    engine = _RoutingEngineStub(decision)
    config = RouterConfig(fallback_models=[fallback_model], max_retries=2)
    router = Router(
        config=config,
        provider_factory=factory,
        routing_engine=engine,
        provider_configs={model.provider.value: provider_config},
        tracker=_TrackerStub(),
        middleware=MiddlewareChain([]),
    )

    response = router.complete("hello")

    assert response.model_used == fallback_model
    assert primary_provider.calls == 1
    assert fallback_provider.calls == 1


def test_alternative_models_are_attempted_by_default():
    selected = _model("claude-3")
    alternative = _model("gpt-4")
    decision = RoutingDecision(
        selected_model=selected,
        estimated_cost=0.02,
        reasoning="ok",
        alternatives_considered=(alternative,),
    )
    factory = _ProviderFactoryStub()
    failing_provider = _FakeProvider(should_fail=True)
    alt_provider = _FakeProvider(
        response=Response(
            content="alt",
            model_used=alternative.model_name,
            cost=0.01,
            latency=0.1,
            tokens=5,
        )
    )
    factory.register(selected.model_name, failing_provider, provider_key=selected.provider.value)
    factory.register(alternative.model_name, alt_provider, provider_key=alternative.provider.value)
    router = Router(
        config=RouterConfig(max_retries=2),
        provider_factory=factory,
        routing_engine=_RoutingEngineStub(decision),
        provider_configs={
            selected.provider.value: ProviderConfig(
                api_key="anthropic", base_url="https://api.anthropic.com"
            ),
            alternative.provider.value: ProviderConfig(
                api_key="openai", base_url="https://api.openai.com"
            ),
        },
        tracker=_TrackerStub(),
        middleware=MiddlewareChain([]),
    )

    response = router.complete("Hello")

    assert response.model_used == alternative.model_name
    assert failing_provider.calls == 1
    assert alt_provider.calls == 1


def test_default_provider_forces_attempt_order():
    selected = ModelConfig(
        provider=Provider.ANTHROPIC,
        model_name="claude-3",
        pricing=0.02,
        capabilities=frozenset({"chat"}),
    )
    preferred = ModelConfig(
        provider=Provider.OPENAI,
        model_name="gpt-4",
        pricing=0.03,
        capabilities=frozenset({"chat"}),
    )
    decision = RoutingDecision(
        selected_model=selected,
        estimated_cost=0.02,
        reasoning="ok",
        alternatives_considered=(preferred,),
    )
    factory = _ProviderFactoryStub()
    anthropic_provider = _FakeProvider(should_fail=False)
    openai_provider = _FakeProvider(
        response=Response(
            content="preferred",
            model_used=preferred.model_name,
            cost=0.02,
            latency=0.1,
            tokens=5,
        )
    )
    factory.register(selected.model_name, anthropic_provider, provider_key="anthropic")
    factory.register(preferred.model_name, openai_provider, provider_key="openai")
    router = Router(
        config=RouterConfig(max_retries=2),
        provider_factory=factory,
        routing_engine=_RoutingEngineStub(decision),
        provider_configs={
            "anthropic": ProviderConfig(
                api_key="anthropic", base_url="https://api.anthropic.com"
            ),
            "openai": ProviderConfig(api_key="openai", base_url="https://api.openai.com"),
        },
        tracker=_TrackerStub(),
        middleware=MiddlewareChain([]),
    )
    router.default_provider = "openai"

    response = router.complete("Hello")

    assert response.model_used == preferred.model_name
    assert factory.requests[0][0] == preferred.model_name


def test_fallback_crosses_providers_when_available():
    primary_model = ModelConfig(
        provider=Provider.ANTHROPIC,
        model_name="claude-3",
        pricing=0.03,
        capabilities=frozenset({"chat"}),
    )
    fallback_model = "gpt-4"
    decision = RoutingDecision(
        selected_model=primary_model,
        estimated_cost=0.03,
        reasoning="ok",
        alternatives_considered=(),
    )
    anthropic_config = ProviderConfig(
        api_key="anthropic", base_url="https://api.anthropic.com"
    )
    openai_config = ProviderConfig(
        api_key="openai", base_url="https://api.openai.com"
    )
    factory = _ProviderFactoryStub()
    failing_provider = _FakeProvider(should_fail=True)
    fallback_provider = _FakeProvider(
        response=Response(
            content="fallback",
            model_used=fallback_model,
            cost=0.02,
            latency=0.1,
            tokens=15,
        )
    )
    factory.register(primary_model.model_name, failing_provider, provider_key="anthropic")
    factory.register(fallback_model, fallback_provider, provider_key="openai")
    engine = _RoutingEngineStub(decision)
    config = RouterConfig(fallback_models=[fallback_model], max_retries=2)
    router = Router(
        config=config,
        provider_factory=factory,
        routing_engine=engine,
        provider_configs={
            "anthropic": anthropic_config,
            "openai": openai_config,
        },
        tracker=_TrackerStub(),
        middleware=MiddlewareChain([]),
    )

    response = router.complete("Hello")

    assert response.model_used == fallback_model
    assert failing_provider.calls == 1
    assert fallback_provider.calls == 1


def test_analytics_property_requires_tracker():
    decision = RoutingDecision(
        selected_model=_model(),
        estimated_cost=0.02,
        reasoning="ok",
        alternatives_considered=(),
    )
    provider_config = ProviderConfig(api_key="key", base_url="https://api.openai.com")
    factory = _ProviderFactoryStub()
    factory.register("gpt-4", _FakeProvider())
    engine = _RoutingEngineStub(decision)
    router = Router(
        config=RouterConfig(),
        provider_factory=factory,
        routing_engine=engine,
        provider_configs={decision.selected_model.provider.value: provider_config},
        tracker=None,
        middleware=MiddlewareChain([]),
    )

    with pytest.raises(RuntimeError):
        _ = router.analytics
