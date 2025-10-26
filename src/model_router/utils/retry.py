"""Retry decorator utilities."""

from __future__ import annotations

import time
from functools import wraps
from typing import Callable, Tuple, Type, TypeVar, ParamSpec


P = ParamSpec("P")
R = TypeVar("R")


def retry(
    *,
    attempts: int = 3,
    delay: float = 0.1,
    backoff: float = 2.0,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Retry decorator with exponential backoff."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            current_delay = delay
            last_error: BaseException | None = None
            for attempt in range(attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_error = exc
                    if attempt == attempts - 1:
                        raise
                    time.sleep(current_delay)
                    current_delay *= backoff
            raise last_error if last_error else RuntimeError("retry failed")

        return wrapper

    return decorator
