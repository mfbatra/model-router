"""Exception hierarchy for domain-specific model routing failures."""

from __future__ import annotations

from typing import Any, Mapping


class ModelRouterError(Exception):
    """Base class for all domain-level errors in the model router."""

    default_message = "Model router error occurred"

    def __init__(
        self, message: str | None = None, *, context: Mapping[str, Any] | None = None
    ):
        self.message = message or self.default_message
        self.context: Mapping[str, Any] = dict(context or {})
        formatted = self._format_message()
        super().__init__(formatted)

    def _format_message(self) -> str:
        if self.context:
            return f"{self.message} | context={self.context}"
        return self.message


class ProviderError(ModelRouterError):
    """Generic provider-related issues (availability, auth, rate limits)."""

    default_message = "Provider error"


class ProviderUnavailableError(ProviderError):
    """Provider service is down or unreachable."""

    default_message = "Provider is unavailable"


class ProviderRateLimitError(ProviderError):
    """Provider refuses request due to rate limiting."""

    default_message = "Provider rate limit exceeded"


class ProviderAuthError(ProviderError):
    """Authentication or authorization with provider failed."""

    default_message = "Provider authentication failed"


class RoutingError(ModelRouterError):
    """Failures while evaluating routing logic."""

    default_message = "Routing error"


class NoSuitableModelError(RoutingError):
    """Raised when no model satisfies the requested constraints."""

    default_message = "No suitable model found for constraints"


class InvalidConstraintsError(RoutingError):
    """Raised when provided routing constraints conflict or are invalid."""

    default_message = "Invalid routing constraints"


class ValidationError(ModelRouterError):
    """Raised when domain validation fails."""

    default_message = "Domain validation failed"
