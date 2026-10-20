# tests/conftest.py
import httpx
import pytest
from dalab.backends.base import BackendConfig
from dalab.questions import Choice, Noul

@pytest.fixture
def gate_questions():
    return {
        "action_class": Choice("What kind of action is the proposed command?", {
            "read_only": "Only reads files or state; changes nothing",
            "local_write": "Writes or deletes inside the workspace",
            "outward": "Sends data or changes outside the workspace",
            "unclear": "Cannot tell from the information given"}),
        "is_destructive": Noul("Would running this command irreversibly delete or overwrite data?"),
    }

@pytest.fixture
def state():
    return "User asked: 'show me the git log'. Agent proposes to run: git log -n 3 --oneline"

# Recorded 2026-10-20 (values abbreviated; shapes exact).
UPSTAGE_RESPONSE = {
    "model": "solar-decide",
    "answers": {
        "action_class": {"type": "choice", "choice": "read_only",
                         "probabilities": {"local_write": 0.000709, "outward": 0.000296, "read_only": 0.998765, "unclear": 0.00023},
                         "confidence": 0.992275},
        "is_destructive": {"type": "noul", "noul": 0.012187},
    },
    "usage": {"input_tokens": 829, "output_tokens": 2},
}
JEVSTATION_RESPONSE = {
    "code": 0, "message": "ok",
    "data": {"model": "jev-1.13.0",
             "answers": {"action_class": {"type": "choice", "choice": "read_only",
                                          "probabilities": {"read_only": 1, "local_write": 0, "outward": 0, "unclear": 0}, "confidence": 1},
                         "is_destructive": {"type": "noul", "noul": 0.01}},
             "usage": {"input_tokens": 418, "output_tokens": 71},
             "credits": {"charged": 1, "remaining": 48}, "latency_ms": 326},
}
TYPESAFE_RESPONSE = {  # shape from docs.typesafe.ai/api; score answer included
    "model": "jev-1.14.0",
    "answers": {
        "action_class": {"type": "choice", "choice": "read_only",
                         "probabilities": {"read_only": 0.97, "local_write": 0.02, "outward": 0.005, "unclear": 0.005}, "confidence": 0.88},
        "is_destructive": {"type": "noul", "noul": 0.03},
        "severity": {"type": "score", "score": 0.4, "legend": {"0": "low", "1": "mid", "2": "high"},
                     "probabilities": {"0": 0.65, "1": 0.3, "2": 0.05}, "confidence": 0.5},
    },
    "usage": {"input_tokens": 410, "output_tokens": 3},
}

def mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))

def json_handler(payload, status=200, capture=None):
    def handler(request):
        if capture is not None:
            capture.append(request)
        return httpx.Response(status, json=payload)
    return handler

@pytest.fixture
def cfg_factory():
    def make(name="typesafe", model="jev-1.14.0", endpoint="https://api.typesafe.ai/v1/systemone", secret="sk-test-0000",
             price_in=0.042, price_out=0.0, max_options=255, timeout_s=10.0, extra=None):
        return BackendConfig(name=name, model=model, endpoint=endpoint, secret=secret, price_input_per_m=price_in,
                             price_output_per_m=price_out, max_options=max_options, timeout_s=timeout_s, extra=extra or {})
    return make
