"""Usage tracking facade that coordinates repository + aggregator."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Sequence
from uuid import uuid4

from model_router.analytics.aggregator import AnalyticsAggregator
from model_router.analytics.interfaces import (
    IAnalyticsAggregator,
    IAnalyticsRepository,
    RequestRecord,
)
from model_router.domain.interfaces import IUsageTracker
from model_router.domain.models import Request, Response
from model_router.domain.interfaces import UsageSummary


class UsageTracker(IUsageTracker):
    """High-level facade for recording usage data and producing summaries."""

    PERIOD_WINDOWS = {
        "last_24_hours": timedelta(days=1),
        "last_7_days": timedelta(days=7),
        "last_30_days": timedelta(days=30),
    }

    def __init__(
        self,
        repository: IAnalyticsRepository,
        aggregator: IAnalyticsAggregator | None = None,
    ) -> None:
        self._repository = repository
        self._aggregator = aggregator or AnalyticsAggregator()

    def track(self, request: Request, response: Response) -> None:
        """Persist the normalized request/response analytics record."""

        record = RequestRecord(
            id=str(request.metadata.get("request_id", uuid4())),
            timestamp=datetime.now(timezone.utc),
            model=response.model_used,
            cost=response.cost,
            latency=response.latency,
            success=True,
        )
        self._repository.save(record)

    def get_summary(self, period: str = "last_7_days") -> UsageSummary:
        """Return aggregate metrics for the requested period."""

        start, end = self._period_window(period)
        records = self._repository.find_by_date(start, end)
        total_cost = self._aggregator.calculate_total_cost(records)
        avg_latency = self._average_latency(records)
        return UsageSummary(
            period=period,
            total_requests=len(records),
            total_cost=total_cost,
            average_latency=avg_latency,
        )

    def to_dataframe(self) -> Any:
        """Export recent analytics to a pandas DataFrame."""

        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("pandas is required for dataframe export") from exc

        start, end = self._period_window("last_30_days")
        records = self._repository.find_by_date(start, end)
        data = [record.model_dump() for record in records]
        return pd.DataFrame(data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _period_window(self, period: str) -> tuple[datetime, datetime]:
        delta = self.PERIOD_WINDOWS.get(period)
        if delta is None:
            raise ValueError(f"Unsupported period '{period}'")
        end = datetime.now(timezone.utc)
        start = end - delta
        return start, end

    @staticmethod
    def _average_latency(records: Sequence[RequestRecord]) -> float:
        if not records:
            return 0.0
        total = sum(record.latency for record in records)
        return round(total / len(records), 6)
