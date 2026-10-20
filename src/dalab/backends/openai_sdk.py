"""Shared bits for the two backends that go through the official OpenAI SDK (Decisions API, Responses API)."""
from __future__ import annotations

from typing import Any

from dalab.util.http import RETRY_STATUSES, BackendError


def make_openai_client(secret: str | None, timeout_s: float) -> Any:
    """SDK client with its own retries switched off: Decider retries instead, so latency_ms covers one attempt."""
    from openai import OpenAI  # imported lazily so unit tests never need the SDK configured

    return OpenAI(api_key=secret, timeout=timeout_s, max_retries=0)


def to_backend_error(e: Exception) -> BackendError:
    """Map an SDK exception to BackendError: status from .status_code; connection/timeout errors are retryable."""
    if isinstance(e, BackendError):
        return e
    status = getattr(e, "status_code", None)
    try:
        from openai import APIConnectionError

        transport = isinstance(e, APIConnectionError)
    except ImportError:  # pragma: no cover - openai is a hard dependency
        transport = False
    return BackendError(status, str(e)[:300], retryable=transport or status in RETRY_STATUSES)
