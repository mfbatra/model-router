import sys
import builtins
from datetime import datetime, timedelta, timezone

import pytest

from model_router.analytics.interfaces import RequestRecord
from model_router.analytics.tracker import UsageTracker
from model_router.domain.models import Request, Response


class _InMemoryRepo:
    def __init__(self):
        self.records: list[RequestRecord] = []

    def save(self, record: RequestRecord) -> None:
        self.records.append(record)

    def find_by_date(self, start: datetime, end: datetime):
        return [r for r in self.records if start <= r.timestamp <= end]

    def find_by_model(self, model: str):
        raise NotImplementedError


class _StubAggregator:
    def __init__(self):
        self.calculate_total_cost_calls = []

    def calculate_total_cost(self, records):
        self.calculate_total_cost_calls.append(len(records))
        return round(sum(r.cost for r in records), 6)

    def calculate_savings(self, records, baseline):
        raise NotImplementedError

    def group_by_model(self, records):
        raise NotImplementedError

    def calculate_percentiles(self, records, metric):
        raise NotImplementedError


def _response(model: str, cost: float, latency: float) -> Response:
    return Response(
        content="ok", model_used=model, cost=cost, latency=latency, tokens=10
    )


def _record(model: str, minutes_ago: int) -> RequestRecord:
    timestamp = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return RequestRecord(
        id=f"{model}-{minutes_ago}",
        timestamp=timestamp,
        model=model,
        cost=0.02 + minutes_ago * 0.001,
        latency=0.2,
        success=True,
    )


def test_track_persists_record():
    repo = _InMemoryRepo()
    tracker = UsageTracker(repo)
    request = Request(prompt="hello", metadata={"request_id": "req-123"})
    response = _response("gpt-4", 0.05, 0.3)

    tracker.track(request, response)

    assert repo.records[0].id == "req-123"
    assert repo.records[0].model == "gpt-4"
    assert repo.records[0].cost == 0.05


def test_get_summary_uses_aggregator(monkeypatch):
    repo = _InMemoryRepo()
    repo.records = [_record("gpt-4", 1), _record("claude", 2)]
    agg = _StubAggregator()
    tracker = UsageTracker(repo, aggregator=agg)

    summary = tracker.get_summary("last_24_hours")

    expected_cost = sum(r.cost for r in repo.records)
    assert summary.total_requests == 2
    assert summary.total_cost == pytest.approx(expected_cost)
    assert summary.average_latency > 0
    assert agg.calculate_total_cost_calls == [2]


def test_to_dataframe_returns_rows(monkeypatch):
    repo = _InMemoryRepo()
    repo.records = [_record("gpt-4", 5)]
    tracker = UsageTracker(repo)

    class FakePandasModule:
        def __init__(self):
            self.data = None

        def DataFrame(self, data):
            self.data = data
            return data

    fake_pd = FakePandasModule()
    monkeypatch.setitem(sys.modules, "pandas", fake_pd)

    df = tracker.to_dataframe()

    assert df[0]["model"] == "gpt-4"
    assert fake_pd.data == df


def test_to_dataframe_raises_without_pandas(monkeypatch):
    repo = _InMemoryRepo()
    tracker = UsageTracker(repo)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("no pandas")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError):
        tracker.to_dataframe()
