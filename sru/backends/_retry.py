"""Shared retry helper for API backends."""
from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")

#: Substrings that mark an error worth retrying rather than aborting on.
TRANSIENT = ("rate limit", "rate_limit", "429", "overloaded", "500", "502",
             "503", "504", "timeout", "timed out", "connection",
             "temporarily unavailable")


def with_retries(fn: Callable[[], T], attempts: int = 6,
                 base_delay: float = 2.0) -> T:
    """Call fn, backing off on transient API failures.

    A non-transient error (a bad model name, a rejected parameter, an auth
    failure) is raised immediately: retrying it only wastes the user's time
    and quota.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - provider SDKs vary
            message = str(exc).lower()
            if not any(t in message for t in TRANSIENT):
                raise
            last = exc
            if attempt == attempts - 1:
                break
            time.sleep(base_delay * (2 ** attempt) * (0.5 + random.random()))
    raise RuntimeError(f"API call failed after {attempts} attempts: {last}")
