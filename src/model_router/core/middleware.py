"""Middleware system for routing cross-cutting concerns."""

from __future__ import annotations

import logging
from typing import Callable, MutableMapping, Protocol, Sequence

from model_router.domain.interfaces import IUsageTracker
from model_router.domain.models import Request, Response


class IMiddleware(Protocol):
    """Protocol describing middleware hooks."""

    def process_request(self, request: Request) -> Request: ...

    def process_response(self, response: Response) -> Response: ...


class LoggingMiddleware(IMiddleware):
    """Logs inbound requests and outbound responses."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(__name__)

    def process_request(self, request: Request) -> Request:
        self._logger.info(
            "routing_request",
            extra={
                "prompt_preview": request.prompt[:100],
                "metadata": request.metadata,
            },
        )
        return request

    def process_response(self, response: Response) -> Response:
        self._logger.info(
            "routing_response",
            extra={
                "model": response.model_used,
                "latency": response.latency,
                "tokens": response.tokens,
            },
        )
        return response


class ValidationMiddleware(IMiddleware):
    """Ensures incoming requests meet minimal criteria."""

    def process_request(self, request: Request) -> Request:
        if not request.prompt or not request.prompt.strip():
            raise ValueError("Request prompt must be non-empty")
        if len(request.prompt) > 20000:
            raise ValueError("Request prompt exceeds maximum length")
        return request

    def process_response(self, response: Response) -> Response:
        return response


class AnalyticsMiddleware(IMiddleware):
    """Forwards usage events to the analytics tracker."""

    def __init__(self, tracker: IUsageTracker) -> None:
        self._tracker = tracker
        self._last_request: Request | None = None

    def process_request(self, request: Request) -> Request:
        self._last_request = request
        return request

    def process_response(self, response: Response) -> Response:
        if self._last_request is not None:
            self._tracker.track(self._last_request, response)
            self._last_request = None
        return response


class CachingMiddleware(IMiddleware):
    """Caches responses keyed by a deterministic prompt signature."""

    def __init__(
        self,
        cache: MutableMapping[str, Response],
        *,
        key_fn: Callable[[Request], str] | None = None,
    ) -> None:
        self._cache = cache
        self._key_fn = key_fn or (lambda req: req.prompt)
        self._current_key: str | None = None

    def process_request(self, request: Request) -> Request:
        key = self._key_fn(request)
        self._current_key = key
        if key in self._cache:
            cached = self._cache[key]
            metadata = dict(request.metadata)
            metadata["cache_hit"] = True
            metadata["cached_response"] = cached.model_dump()
            return request.model_copy(update={"metadata": metadata})
        return request

    def process_response(self, response: Response) -> Response:
        if self._current_key is not None:
            self._cache[self._current_key] = response
            self._current_key = None
        return response


class MiddlewareChain:
    """Applies middleware around a handler using chain of responsibility."""

    def __init__(self, middlewares: Sequence[IMiddleware]) -> None:
        self._middlewares = list(middlewares)

    def execute(
        self, request: Request, handler: Callable[[Request], Response]
    ) -> Response:
        for middleware in self._middlewares:
            request = middleware.process_request(request)

        response = handler(request)

        for middleware in reversed(self._middlewares):
            response = middleware.process_response(response)

        return response
