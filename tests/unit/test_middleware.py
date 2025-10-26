import logging
from typing import List

import pytest

from model_router.core.middleware import (
    AnalyticsMiddleware,
    CachingMiddleware,
    IMiddleware,
    LoggingMiddleware,
    MiddlewareChain,
    ValidationMiddleware,
)
from model_router.domain.models import Request, Response


class _FakeLogger:
    def __init__(self):
        self.messages: List[str] = []

    def info(self, msg, *_, **__):
        self.messages.append(msg)


class _StubTracker:
    def __init__(self):
        self.records = []

    def track(self, request: Request, response: Response) -> None:
        self.records.append((request, response))

    def get_summary(self, period: str = "last_7_days"):
        raise NotImplementedError

    def to_dataframe(self):
        raise NotImplementedError


def _request(prompt: str = "hello") -> Request:
    return Request(prompt=prompt)


def _response(model: str = "gpt-4") -> Response:
    return Response(content="ok", model_used=model, cost=0.01, latency=0.2, tokens=10)


def test_logging_middleware_logs_messages():
    fake_logger = _FakeLogger()
    middleware = LoggingMiddleware(logger=fake_logger)  # type: ignore[arg-type]

    middleware.process_request(_request("hi"))
    middleware.process_response(_response())

    assert "routing_request" in fake_logger.messages
    assert "routing_response" in fake_logger.messages


def test_validation_middleware_rejects_empty_prompt():
    middleware = ValidationMiddleware()
    with pytest.raises(ValueError):
        middleware.process_request(_request("    "))


def test_analytics_middleware_tracks_response():
    tracker = _StubTracker()
    middleware = AnalyticsMiddleware(tracker)
    req = _request("complex prompt")
    res = _response()

    middleware.process_request(req)
    middleware.process_response(res)

    assert tracker.records[0] == (req, res)


def test_caching_middleware_marks_cache_hit_and_saves_response():
    cache = {}
    middleware = CachingMiddleware(cache)
    req = _request("cache me")
    res = _response("cached-model")

    cached_request = middleware.process_request(req)
    assert (
        cached_request.metadata.get("cache_hit") is False
        or "cache_hit" not in cached_request.metadata
    )

    middleware.process_response(res)
    assert cache["cache me"].model_used == "cached-model"

    # second pass hits cache
    cache_request = middleware.process_request(req)
    assert cache_request.metadata.get("cache_hit") is True


def test_middleware_chain_runs_in_order():
    class _RecordingMiddleware:
        def __init__(self, name: str, log: List[str]):
            self.name = name
            self.log = log

        def process_request(self, request: Request) -> Request:
            self.log.append(f"req:{self.name}")
            return request

        def process_response(self, response: Response) -> Response:
            self.log.append(f"res:{self.name}")
            return response

    log: List[str] = []
    chain = MiddlewareChain(
        [
            _RecordingMiddleware("a", log),
            _RecordingMiddleware("b", log),
        ]
    )

    def handler(request: Request) -> Response:
        log.append("handler")
        return _response()

    chain.execute(_request(), handler)

    assert log == ["req:a", "req:b", "handler", "res:b", "res:a"]
