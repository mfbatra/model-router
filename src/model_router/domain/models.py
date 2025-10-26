"""Domain value objects representing model routing concepts."""

from __future__ import annotations

from enum import Enum
from typing import Any, FrozenSet, Mapping, Optional, Sequence, Tuple

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic.dataclasses import dataclass as pydantic_dataclass


class Provider(str, Enum):
    """Supported model providers inside the domain boundary."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    CUSTOM = "custom"


@pydantic_dataclass(frozen=True)
class ModelConfig:
    """Immutable configuration for an individual model option."""

    provider: Provider
    model_name: str
    pricing: float = Field(..., gt=0)
    capabilities: FrozenSet[str] = Field(default_factory=frozenset)

    @field_validator("pricing")
    @classmethod
    def validate_pricing(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("pricing must be greater than zero")
        return value

    @field_validator("capabilities")
    @classmethod
    def validate_capabilities(
        cls, value: Sequence[str] | FrozenSet[str]
    ) -> FrozenSet[str]:
        return frozenset(value)


class Request(BaseModel):
    """Immutable user request passed through routing."""

    model_config = ConfigDict(frozen=True)

    prompt: str
    params: Mapping[str, Any] = Field(default_factory=dict)
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_prompt(self) -> "Request":
        if not self.prompt or not self.prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        return self


class ProviderResponsePayload(BaseModel):
    """Normalized provider response used by Response factory."""

    content: Any
    model_used: str
    cost: float
    latency: float
    tokens: int

    model_config = ConfigDict(frozen=True)


class Response(BaseModel):
    """Immutable response returned by routing layer."""

    model_config = ConfigDict(frozen=True)

    content: Any
    model_used: str
    cost: float = Field(..., ge=0)
    latency: float = Field(..., ge=0)
    tokens: int = Field(..., ge=0)

    @classmethod
    def from_provider_response(
        cls,
        payload: ProviderResponsePayload | Mapping[str, Any],
    ) -> "Response":
        if isinstance(payload, Mapping):
            try:
                payload = ProviderResponsePayload(**payload)
            except ValidationError as exc:
                raise ValueError("invalid provider response payload") from exc
        return cls(**payload.model_dump())


class RoutingStrategy(str, Enum):
    """Supported routing strategies."""

    COST_BIASED = "cost_biased"
    QUALITY_BIASED = "quality_biased"
    BALANCED = "balanced"
    LATENCY_BIASED = "latency_biased"


class RoutingConstraints(BaseModel):
    """Routing guardrails that influence decision making."""

    model_config = ConfigDict(frozen=True)

    max_cost: Optional[float] = Field(default=None, gt=0)
    max_latency: Optional[float] = Field(default=None, gt=0)
    min_quality: Optional[float] = Field(default=None)
    strategy: RoutingStrategy = RoutingStrategy.BALANCED

    @model_validator(mode="after")
    def validate_constraints(self) -> "RoutingConstraints":
        if self.min_quality is not None and not 0 <= self.min_quality <= 1:
            raise ValueError("min_quality must be between 0 and 1")
        if self.max_cost is None and self.max_latency is None:
            raise ValueError("at least one constraint must be specified")
        return self


class RoutingDecision(BaseModel):
    """Result of evaluating constraints against available models."""

    model_config = ConfigDict(frozen=True)

    selected_model: ModelConfig
    estimated_cost: float = Field(..., ge=0)
    reasoning: str
    alternatives_considered: Tuple[ModelConfig, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def ensure_reasoning(self) -> "RoutingDecision":
        if not self.reasoning.strip():
            raise ValueError("reasoning must not be empty")
        if self.selected_model in self.alternatives_considered:
            raise ValueError("selected model cannot be listed as an alternative")
        return self
