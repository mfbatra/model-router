from datetime import datetime, timedelta

import pytest

from model_router.analytics.aggregator import AnalyticsAggregator
from model_router.analytics.interfaces import RequestRecord


def _record(
    idx: int, model: str, cost: float, latency: float, success: bool = True
) -> RequestRecord:
    return RequestRecord(
        id=f"{model}-{idx}",
        timestamp=datetime(2024, 1, 1) + timedelta(minutes=idx),
        model=model,
        cost=cost,
        latency=latency,
        success=success,
    )


def test_calculate_total_cost():
    agg = AnalyticsAggregator()
    records = [_record(1, "gpt-4", 0.02, 0.5), _record(2, "claude", 0.03, 0.6)]
    assert agg.calculate_total_cost(records) == 0.05


def test_calculate_savings_compares_to_baseline():
    agg = AnalyticsAggregator()
    records = [
        _record(1, "gpt-4", 0.02, 0.5),
        _record(2, "claude", 0.05, 0.6),
        _record(3, "claude", 0.05, 0.7),
    ]
    savings = agg.calculate_savings(records, baseline=0.05)
    # Baseline cost per request = 0.05 so hypothetical 0.15 vs actual 0.12
    assert savings == pytest.approx(0.03)


def test_calculate_savings_zero_records_returns_zero():
    agg = AnalyticsAggregator()
    assert agg.calculate_savings([], baseline=0.05) == 0.0


def test_group_by_model():
    agg = AnalyticsAggregator()
    records = [
        _record(i, "gpt-4" if i % 2 == 0 else "claude", 0.01, 0.4) for i in range(4)
    ]
    grouped = agg.group_by_model(records)
    assert set(grouped.keys()) == {"gpt-4", "claude"}
    assert len(grouped["gpt-4"]) == 2


def test_calculate_percentiles_for_latency():
    agg = AnalyticsAggregator()
    records = [_record(i, "gpt-4", 0.02, latency=i * 0.1) for i in range(1, 6)]
    percentiles = agg.calculate_percentiles(records, metric="latency")
    assert percentiles["p50"] == pytest.approx(0.3)
    assert percentiles["p95"] == pytest.approx(0.48)
    assert percentiles["p99"] == pytest.approx(0.496)


def test_calculate_percentiles_empty_records():
    agg = AnalyticsAggregator()
    assert agg.calculate_percentiles([], metric="cost") == {
        "p50": 0.0,
        "p95": 0.0,
        "p99": 0.0,
    }


def test_calculate_percentiles_rejects_unknown_metric():
    agg = AnalyticsAggregator()
    with pytest.raises(ValueError):
        agg.calculate_percentiles([], metric="tokens")
