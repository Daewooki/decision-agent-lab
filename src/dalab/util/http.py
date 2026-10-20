"""httpx helpers: a configured client, a bounded retry loop, and a JSON POST built on it."""
from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import Any

import httpx

# 408 (request timeout) and 429 ask for a repeat; 5xx and 529 are transient server trouble.
RETRY_STATUSES = {408, 429, 500, 502, 503, 504, 529}


class BackendError(Exception):
    """A failed backend call.

    retryable: whether with_retries() may try again (default: status in RETRY_STATUSES).
    attempts:  set by with_retries() when it gives up, so error rows record how many tries were made.
    usage:     (input_tokens, output_tokens) when the provider billed the call although it failed
               (an OpenAI refusal, for example), so the error row still carries tokens and cost.
    """

    def __init__(self, status: int | None, message: str, body: Any = None, *, retryable: bool | None = None):
        super().__init__(f"HTTP {status}: {message}" if status is not None else message)
        self.status = status
        self.body = body
        self.retryable = (status in RETRY_STATUSES) if retryable is None else retryable
        self.attempts: int | None = None
        self.usage: tuple[int | None, int | None] | None = None


def make_client(timeout_s: float) -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(timeout_s, connect=10.0))


def _error_message(body: Any, fallback: str) -> str:
    if isinstance(body, dict):
        for key in ("detail", "error", "message"):
            v = body.get(key)
            if isinstance(v, dict):
                v = v.get("message") or v.get("type")
            if isinstance(v, str) and v:
                return v
    return fallback


def with_retries(
    attempt: Callable[[], Any],
    *,
    retries: int = 3,
    backoff_s: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Any, float, int]:
    """Call `attempt` up to retries+1 times, sleeping with exponential backoff after a retryable BackendError.

    Returns (result, wall-clock ms of the attempt that produced it, attempts made). Only the final attempt is
    timed, never the earlier tries or the sleeps between them, so latencies stay comparable across backends.
    Raises the last BackendError, with .attempts filled in, when it is not retryable or retries run out.
    """
    last: BackendError | None = None
    for i in range(retries + 1):
        t0 = time.perf_counter()
        try:
            result = attempt()
        except BackendError as e:
            e.attempts = i + 1
            last = e
            if not e.retryable:
                raise
        else:
            return result, (time.perf_counter() - t0) * 1000.0, i + 1
        if i < retries:
            sleep(backoff_s * (2**i) + random.uniform(0, 0.1))
    assert last is not None
    raise last


def post_json(
    client: httpx.Client,
    url: str,
    *,
    headers: dict[str, str],
    json: dict[str, Any],
    retries: int = 3,
    backoff_s: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any], float, int]:
    """POST JSON through with_retries(); transport errors and RETRY_STATUSES are retried.

    Returns (parsed body, wall-clock ms of the final attempt, attempts made).
    Raises BackendError on a non-retryable status or after `retries` failed retries.
    """

    def attempt() -> dict[str, Any]:
        try:
            r = client.post(url, headers=headers, json=json)
        except httpx.TransportError as e:
            raise BackendError(None, f"transport error: {e}", retryable=True) from e
        if r.status_code < 400:
            try:
                return r.json()
            except ValueError:
                raise BackendError(r.status_code, "non-JSON response body", r.text[:500], retryable=False) from None
        try:
            body: Any = r.json()
        except ValueError:
            body = r.text[:500]
        raise BackendError(r.status_code, _error_message(body, r.text[:300]), body)

    return with_retries(attempt, retries=retries, backoff_s=backoff_s, sleep=sleep)
