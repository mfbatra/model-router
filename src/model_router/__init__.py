"""Model Router package following Clean Architecture layering."""

from .core.router import Router
from .core.container import DIContainer

__all__ = [
    "Router",
    "DIContainer",
    "domain",
    "routing",
    "core",
    "providers",
    "analytics",
    "utils",
]
