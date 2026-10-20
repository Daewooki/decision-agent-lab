# tests/test_http.py
import json
import time

import httpx
import pytest
from dalab.util.http import BackendError, RETRY_STATUSES, make_client, post_json, with_retries

def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))

def test_retries_then_succeeds():
    calls = []
    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(429, json={"error": "slow down"})
        return httpx.Response(200, json={"ok": True})
    slept = []
    body, ms, attempts = post_json(_client(handler), "https://x.invalid/v1", headers={"A": "b"}, json={"q": 1},
                                   retries=3, backoff_s=0.01, sleep=slept.append)
    assert body == {"ok": True} and ms >= 0 and len(calls) == 3 and len(slept) == 2
    assert attempts == 3
    assert calls[0].headers["A"] == "b"
    assert json.loads(calls[0].content) == {"q": 1}

def test_non_retryable_raises_immediately():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(401, json={"detail": {"message": "bad key"}})
    with pytest.raises(BackendError) as ei:
        post_json(_client(handler), "https://x.invalid/v1", headers={}, json={}, sleep=lambda s: None)
    assert ei.value.status == 401 and "bad key" in str(ei.value) and len(calls) == 1
    assert ei.value.attempts == 1 and not ei.value.retryable

def test_exhausted_retries_raise_last_error():
    def handler(request):
        return httpx.Response(529, json={"error": "overloaded"})
    with pytest.raises(BackendError) as ei:
        post_json(_client(handler), "https://x.invalid/v1", headers={}, json={}, retries=2, sleep=lambda s: None)
    assert ei.value.status == 529 and ei.value.attempts == 3

def test_transport_error_is_retried():
    n = {"i": 0}
    def handler(request):
        n["i"] += 1
        if n["i"] == 1:
            raise httpx.ConnectError("boom")
        return httpx.Response(200, json={"ok": 1})
    body, _, attempts = post_json(_client(handler), "https://x.invalid/v1", headers={}, json={}, sleep=lambda s: None)
    assert body == {"ok": 1} and n["i"] == 2 and attempts == 2

def test_make_client_timeout():
    c = make_client(7.5)
    assert c.timeout.read == 7.5 and c.timeout.connect == 10.0
    assert 408 in RETRY_STATUSES and 429 in RETRY_STATUSES and 529 in RETRY_STATUSES

def test_with_retries_times_only_the_final_attempt():
    n = {"i": 0}
    def attempt():
        n["i"] += 1
        if n["i"] == 1:
            time.sleep(0.05)  # a slow failure that must not leak into the reported latency
            raise BackendError(503, "overloaded")
        return "ok"
    slept = []
    result, ms, attempts = with_retries(attempt, backoff_s=0.01, sleep=slept.append)
    assert result == "ok" and attempts == 2 and ms < 40
    assert len(slept) == 1 and 0.01 <= slept[0] <= 0.11  # backoff plus at most 0.1 s jitter

def test_with_retries_backs_off_exponentially_and_reports_attempts():
    def attempt():
        raise BackendError(429, "slow down")
    slept = []
    with pytest.raises(BackendError) as ei:
        with_retries(attempt, retries=3, backoff_s=0.01, sleep=slept.append)
    assert ei.value.attempts == 4 and len(slept) == 3
    assert all(slept[i] >= 0.01 * 2**i for i in range(3))
