"""Analytics contracts that separate persistence from aggregation logic."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field


class RequestRecord(BaseModel):
    """Immutable record summarizing a routed request outcome."""

    id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model: str
    cost: float
    latency: float
    success: bool = True

    model_config = ConfigDict(frozen=True)


class IAnalyticsRepository(Protocol):
    """Persistence contract for analytics storage layers."""

    def save(self, record: RequestRecord) -> None:
        """Persist the provided request record."""

    def find_by_date(self, start: datetime, end: datetime) -> List[RequestRecord]:
        """Return records whose timestamps fall within the inclusive window."""

    def find_by_model(self, model: str) -> List[RequestRecord]:
        """Return records routed through the specified model id."""


class IAnalyticsAggregator(Protocol):
    """Business-logic layer that derives metrics from persisted records."""

    def calculate_total_cost(self, records: Sequence[RequestRecord]) -> float:
        """Sum cost across the supplied records."""

    def calculate_savings(
        self, records: Sequence[RequestRecord], baseline: float
    ) -> float:
        """Compare spend with a baseline value and return the savings (baseline - actual)."""

    def group_by_model(
        self, records: Sequence[RequestRecord]
    ) -> Dict[str, List[RequestRecord]]:
        """Bucket records by model identifier for downstream aggregations."""
