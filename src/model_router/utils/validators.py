"""Input validation helpers used across the router."""

from __future__ import annotations

from model_router.domain.models import RoutingConstraints

MAX_PROMPT_LENGTH = 20000


def validate_prompt(prompt: str) -> None:
    if not prompt or not prompt.strip():
        raise ValueError("Prompt must be non-empty")
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise ValueError("Prompt exceeds maximum supported length")


def validate_constraints(constraints: RoutingConstraints) -> None:
    if constraints.max_cost is not None and constraints.max_cost <= 0:
        raise ValueError("max_cost must be greater than zero")
    if constraints.max_latency is not None and constraints.max_latency <= 0:
        raise ValueError("max_latency must be greater than zero")
    if constraints.min_quality is not None and not 0 <= constraints.min_quality <= 1:
        raise ValueError("min_quality must be between 0 and 1")
