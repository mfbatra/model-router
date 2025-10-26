# Model Router Architecture

## Clean Architecture Layers

```mermaid
graph TD
    Client[API Clients] --> RouterFacade[Router Facade]
    RouterFacade --> MiddlewareChain
    RouterFacade --> RoutingEngine
    RoutingEngine --> ModelSelector
    RoutingEngine --> ComplexityEstimator
    RouterFacade --> ProviderFactory
    RouterFacade --> UsageTracker
    ProviderFactory --> Providers
    UsageTracker --> AnalyticsRepo
```

## SOLID Principles

1. **Single Responsibility**: `RoutingEngine` only coordinates routing; `ProviderFactory` only creates providers; `UsageTracker` only tracks analytics.
2. **Open/Closed**: Strategies, providers, middleware, and estimators can be added or replaced via DI without modifying existing code.
3. **Liskov Substitution**: Any class implementing `IProvider` or `IRoutingStrategy` can replace another without breaking consumers.
4. **Interface Segregation**: Separate protocols (providers, analytics, routing, middleware) prevent clients from depending on unused methods.
5. **Dependency Inversion**: Router, engine, and selectors depend on interfaces injected by `DIContainer` rather than concrete classes.

## Design Patterns Used

- **Strategy**: `IRoutingStrategy` implementations drive scoring variations (cost, quality, latency, balanced).
- **Factory**: `ProviderFactory` detects provider types, caches instances, and supplies them to the router.
- **Facade**: `Router` exposes a single `complete/chat` API over complex internals (providers, engine, middleware).
- **Template Method**: `BaseProvider` unifies retry/backoff/logging while subclasses implement `_make_api_call`.
- **Chain of Responsibility**: `MiddlewareChain` composes cross-cutting behaviors around request handling.

## Dependency Flow

```mermaid
graph LR
    Domain --> Routing --> Core --> Providers
    Core --> Analytics
    Core --> Utils
```

## Sequence Diagram (Routing Flow)

```mermaid
sequenceDiagram
    actor Client
    participant Router
    participant Middleware
    participant Engine
    participant Selector
    participant Strategy
    participant ProviderFactory
    participant Provider

    Client->>Router: complete(prompt, constraints)
    Router->>Middleware: execute(request)
    Middleware->>Router: request'
    Router->>Engine: route(request', constraints)
    Engine->>Selector: select(models, request', constraints)
    Selector->>Strategy: score models
    Strategy-->>Selector: scores
    Selector-->>Engine: routing decision
    Engine-->>Router: routing decision
    Router->>ProviderFactory: create(selected_model)
    ProviderFactory->>Provider: complete(request')
    Provider-->>Router: response
    Router->>Middleware: process_response(response)
    Middleware-->>Router: response
    Router-->>Client: response
```

## Extensibility

- **Add Providers**: Inherit from `BaseProvider`, implement `_make_api_call`, register a regex pattern + class in `ProviderFactory`.
- **Add Strategies**: Implement `IRoutingStrategy`, wire into `DIContainer` and `ModelSelector`.
- **Add Middlewares**: Implement `IMiddleware` pipes and include them in `MiddlewareChain`.
- **Add Analytics**: Implement `IAnalyticsRepository` or `IAnalyticsAggregator` variations and inject them via the container.

## Testing Strategy

- **Unit Tests**: Cover domain models, utilities, providers, strategies, selector, router, middleware, analytics, and DI wiring.
- **Integration Tests**: Validate router flows end-to-end with stubbed providers (integration marker), covering fallback, middleware, analytics, and constraint propagation.
