"""Domain-level interfaces defining contracts for routing collaborators."""

from __future__ import annotations

from datetime import datetime
from typing import List, Mapping, Protocol, Sequence

from pydantic import BaseModel, Field

from .models import ModelConfig, Request, Response, RoutingConstraints, RoutingDecision


class RequestRecord(BaseModel):
    """Historic snapshot of a model invocation for analytics and auditing."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request: Request
    response: Response
    tags: Mapping[str, str] = Field(default_factory=dict)


class UsageSummary(BaseModel):
    """Aggregated usage metrics across a reporting window."""

    period: str
    total_requests: int
    total_cost: float
    average_latency: float


class IProvider(Protocol):
    """Contract every provider adapter must satisfy."""

    def complete(self, request: Request) -> Response:
        """Submit a request to the provider and return a normalized response."""

    def supports_streaming(self) -> bool:
        """Return True when the provider can stream partial outputs."""

    def get_pricing(self) -> ModelConfig:
        """Expose immutable pricing metadata for routing comparisons."""


class IComplexityEstimator(Protocol):
    """Produces a normalized 0-1 complexity score for prompts."""

    def estimate(self, prompt: str) -> float:
        """Return a float between 0 and 1 indicating prompt complexity."""


class IRoutingStrategy(Protocol):
    """Encapsulates model-selection heuristics based on constraints."""

    def select_model(
        self,
        models: Sequence[ModelConfig],
        complexity: float,
        constraints: RoutingConstraints,
    ) -> ModelConfig:
        """Choose the best model candidate for the supplied complexity and constraints."""

    def name(self) -> str:
        """Return the strategy identifier for observability tagging."""


class IModelSelector(Protocol):
    """Coordinates providers, strategies, and constraints to make routing decisions."""

    def select(
        self,
        models: Sequence[ModelConfig],
        request: Request,
        constraints: RoutingConstraints,
    ) -> RoutingDecision:
        """Return a routing decision describing the chosen model and reasoning."""


class IAnalyticsRepository(Protocol):
    """Persists and queries request/response analytics records."""

    def save_request(self, request: Request, response: Response) -> None:
        """Persist the fully materialized request lifecycle for later analysis."""

    def query(self, filters: Mapping[str, object]) -> List[RequestRecord]:
        """Return request records that match the provided filter criteria."""


class IUsageTracker(Protocol):
    """Collects usage signals for cost control and observability."""

    def track(self, request: Request, response: Response) -> None:
        """Record usage metrics derived from the completed request."""

    def get_summary(self, period: str) -> UsageSummary:
        """Retrieve pre-aggregated usage data for the requested period label."""
