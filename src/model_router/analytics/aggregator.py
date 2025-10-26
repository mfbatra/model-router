"""Pure business-logic helpers for analytics aggregation."""

from __future__ import annotations

from typing import Dict, List, Sequence

from model_router.analytics.interfaces import IAnalyticsAggregator, RequestRecord


class AnalyticsAggregator(IAnalyticsAggregator):
    """Performs read-only calculations on analytics records."""

    def calculate_total_cost(self, records: Sequence[RequestRecord]) -> float:
        return round(sum(record.cost for record in records), 6)

    def calculate_savings(
        self, records: Sequence[RequestRecord], baseline: float
    ) -> float:
        if not records or baseline <= 0:
            return 0.0
        actual_cost = sum(record.cost for record in records)
        hypothetical = baseline * len(records)
        return round(max(hypothetical - actual_cost, 0.0), 6)

    def group_by_model(
        self, records: Sequence[RequestRecord]
    ) -> Dict[str, List[RequestRecord]]:
        grouped: Dict[str, List[RequestRecord]] = {}
        for record in records:
            grouped.setdefault(record.model, []).append(record)
        return grouped

    def calculate_percentiles(
        self, records: Sequence[RequestRecord], metric: str
    ) -> Dict[str, float]:
        if metric not in {"cost", "latency"}:
            raise ValueError("metric must be 'cost' or 'latency'")
        if not records:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        values = sorted(getattr(record, metric) for record in records)
        return {
            "p50": self._percentile(values, 0.5),
            "p95": self._percentile(values, 0.95),
            "p99": self._percentile(values, 0.99),
        }

    @staticmethod
    def _percentile(values: Sequence[float], quantile: float) -> float:
        if not values:
            return 0.0
        index = (len(values) - 1) * quantile
        lower = int(index)
        upper = min(lower + 1, len(values) - 1)
        weight = index - lower
        return round(values[lower] * (1 - weight) + values[upper] * weight, 6)
