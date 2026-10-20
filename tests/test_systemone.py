# tests/test_systemone.py
import copy
import json
import httpx
import pytest
from dalab.backends.systemone import JevstationDecider, SystemOneDecider, parse_systemone_answers
from dalab.decision import Decision
from dalab.questions import Choice
from tests.conftest import JEVSTATION_RESPONSE, TYPESAFE_RESPONSE, UPSTAGE_RESPONSE, json_handler, mock_client

def test_parse_answers_choice_noul_score():
    a = parse_systemone_answers(TYPESAFE_RESPONSE["answers"])
    assert a["action_class"].choice == "read_only" and a["action_class"].confidence == 0.88
    assert a["action_class"].confidence_source == "reported"
    assert a["is_destructive"].noul == 0.03
    assert a["severity"].score == 0.4 and a["severity"].probabilities == {"0": 0.65, "1": 0.3, "2": 0.05}

def test_parse_answers_computes_confidence_when_missing():
    a = parse_systemone_answers({"q": {"type": "choice", "choice": "a", "probabilities": {"a": 1.0, "b": 0.0}}})
    assert a["q"].confidence == 1.0 and a["q"].confidence_source == "computed"

def test_parse_answers_without_probabilities_has_no_confidence():
    a = parse_systemone_answers({"q": {"type": "choice", "choice": "a"}})
    assert a["q"].choice == "a" and a["q"].probabilities is None
    assert a["q"].confidence is None and a["q"].confidence_source is None  # never a fabricated 1.0

def test_typesafe_request_and_decision(cfg_factory, gate_questions, state):
    captured = []
    d = SystemOneDecider(cfg_factory(), client=mock_client(json_handler(TYPESAFE_RESPONSE, capture=captured))).decide(state, gate_questions)
    req = json.loads(captured[0].content)
    assert captured[0].headers["authorization"] == "Bearer sk-test-0000"
    assert req["model"] == "jev-1.14.0" and req["state"] == state
    assert list(req["questions"]["action_class"]["criteria"]) == ["read_only", "local_write", "outward", "unclear"]
    assert isinstance(d, Decision) and d.error is None
    assert d.backend == "typesafe" and d.model_requested == "jev-1.14.0" and d.model_reported == "jev-1.14.0"
    assert d.usage_input_tokens == 410 and abs(d.cost_usd - 410 / 1e6 * 0.042) < 1e-12
    assert d.latency_ms is not None and d.latency_ms >= 0 and d.server_latency_ms is None
    assert d.option_order == {"action_class": ["read_only", "local_write", "outward", "unclear"]}
    assert d.ts_utc.endswith("+00:00") and d.location == "Seoul, KR"
    assert d.raw == TYPESAFE_RESPONSE
    assert d.extra["attempts"] == 1 and d.extra["wall_ms"] >= d.latency_ms

def test_order_seed_permutes_wire_order(cfg_factory, gate_questions, state):
    captured = []
    d = SystemOneDecider(cfg_factory(), client=mock_client(json_handler(TYPESAFE_RESPONSE, capture=captured))).decide(state, gate_questions, order_seed=3)
    sent = list(json.loads(captured[0].content)["questions"]["action_class"]["criteria"])
    assert sent == d.option_order["action_class"] and sorted(sent) == sorted(gate_questions["action_class"].criteria)

def test_upstage_response(cfg_factory, gate_questions, state):
    cfg = cfg_factory(name="upstage", model="solar-decide", endpoint="https://api.upstage.ai/v1/systemone", price_in=0.10, max_options=26)
    d = SystemOneDecider(cfg, client=mock_client(json_handler(UPSTAGE_RESPONSE))).decide(state, gate_questions)
    assert d.model_reported == "solar-decide" and d.answers["action_class"].choice == "read_only"
    assert abs(d.cost_usd - 829 / 1e6 * 0.10) < 1e-12

def test_too_many_options_raises_before_calling(cfg_factory, state):
    cfg = cfg_factory(name="upstage", model="solar-decide", max_options=26)
    qs = {"big": Choice("Pick", {f"o{i}": f"option {i}" for i in range(27)})}
    with pytest.raises(ValueError, match="27 options"):
        SystemOneDecider(cfg, client=mock_client(json_handler({}))).decide(state, qs)

def test_http_error_becomes_decision_error(cfg_factory, gate_questions, state):
    d = SystemOneDecider(cfg_factory(), client=mock_client(json_handler({"detail": {"message": "Cannot authenticate"}}, status=401))).decide(state, gate_questions)
    assert d.error is not None and "401" in d.error and d.answers == {} and d.cost_usd is None and d.model_reported is None
    assert d.extra["attempts"] == 1

def test_http_error_echoing_the_key_is_masked_in_the_row(cfg_factory, gate_questions, state):
    body = {"detail": {"message": "key sk-test-0000 is not valid"}, "echo": "Bearer sk-test-0000"}
    d = SystemOneDecider(cfg_factory(), client=mock_client(json_handler(body, status=401))).decide(state, gate_questions)
    assert "sk-test-0000" not in d.error and "***" in d.error
    assert "sk-test-0000" not in json.dumps(d.raw) and d.raw["error_body"]["echo"] == "Bearer ***"

def test_retryable_status_is_retried_and_only_the_last_attempt_is_timed(cfg_factory, gate_questions, state):
    n = {"i": 0}
    def handler(request):
        n["i"] += 1
        return httpx.Response(429, json={"error": "slow down"}) if n["i"] < 3 else httpx.Response(200, json=TYPESAFE_RESPONSE)
    dec = SystemOneDecider(cfg_factory(), client=mock_client(handler))
    slept = []
    dec._sleep = slept.append
    d = dec.decide(state, gate_questions)
    assert d.error is None and d.extra["attempts"] == 3 and len(slept) == 2 and n["i"] == 3
    assert d.latency_ms is not None and d.extra["wall_ms"] >= d.latency_ms

def test_exhausted_retries_leave_an_error_row_with_attempts(cfg_factory, gate_questions, state):
    dec = SystemOneDecider(cfg_factory(), client=mock_client(json_handler({"error": "overloaded"}, status=529)))
    dec._sleep = lambda s: None
    d = dec.decide(state, gate_questions)
    assert d.error is not None and "529" in d.error and d.extra["attempts"] == 4

@pytest.mark.parametrize("payload", [
    {"model": "jev-1.14.0", "answers": {"action_class": {"type": "choice", "choice": "read_only", "probabilities": {"read_only": 1}},
                                        "is_destructive": {"type": "noul"}}},                                  # noul without value
    {"model": "jev-1.14.0", "answers": {"action_class": {"type": "choice", "choice": "read_only", "probabilities": {"read_only": 1}},
                                        "is_destructive": {"type": "noul", "noul": None}}},                    # noul null
    {"model": "jev-1.14.0", "answers": [{"type": "noul", "noul": 0.1}]},                                       # answers as a list
    {"model": "jev-1.14.0", "answers": {"action_class": {"type": "choice", "choice": "read_only", "probabilities": {"read_only": "all"}},
                                        "is_destructive": {"type": "noul", "noul": 0.1}}},                     # non-numeric probability
    {"model": "jev-1.14.0", "answers": {"action_class": "read_only", "is_destructive": 0.1}},                  # answers not objects
    ["a", "list", "body"],
])
def test_malformed_bodies_become_error_rows(cfg_factory, gate_questions, state, payload):
    d = SystemOneDecider(cfg_factory(), client=mock_client(json_handler(payload))).decide(state, gate_questions)
    assert d.error is not None and d.answers == {} and d.cost_usd is None

def test_missing_answer_is_an_error_row(cfg_factory, gate_questions, state):
    partial = copy.deepcopy(TYPESAFE_RESPONSE)
    del partial["answers"]["is_destructive"]
    d = SystemOneDecider(cfg_factory(), client=mock_client(json_handler(partial))).decide(state, gate_questions)
    assert d.error == "missing answers for ['is_destructive']" and d.answers == {} and d.raw["error_body"] == partial

def test_ollama_without_secret_sends_no_auth_header(cfg_factory, gate_questions, state):
    captured = []
    cfg = cfg_factory(name="ollama", model="nimble", endpoint="http://localhost:11434/v1/systemone", secret=None, price_in=0.0, max_options=26)
    SystemOneDecider(cfg, client=mock_client(json_handler(UPSTAGE_RESPONSE, capture=captured))).decide(state, gate_questions)
    assert "authorization" not in captured[0].headers

def test_jevstation_unwraps_and_charges_credits(cfg_factory, gate_questions, state):
    cfg = cfg_factory(name="jevstation", model="jev-1.13.0", endpoint="https://jevstation.com/api/v1/systemone", price_in=0.0, extra={"price_per_credit": 0.00198})
    d = JevstationDecider(cfg, client=mock_client(json_handler(JEVSTATION_RESPONSE))).decide(state, gate_questions)
    assert d.error is None and d.model_reported == "jev-1.13.0" and d.answers["is_destructive"].noul == 0.01
    assert d.server_latency_ms == 326.0
    assert {k: d.extra[k] for k in ("credits_charged", "credits_remaining", "relay_latency_ms")} == {"credits_charged": 1, "credits_remaining": 48, "relay_latency_ms": 326}
    assert d.extra["attempts"] == 1 and "wall_ms" in d.extra
    assert abs(d.cost_usd - 0.00198) < 1e-12

def test_jevstation_nonzero_code_is_error(cfg_factory, gate_questions, state):
    cfg = cfg_factory(name="jevstation", model="jev-1.13.0", extra={"price_per_credit": 0.00198})
    d = JevstationDecider(cfg, client=mock_client(json_handler({"code": 42, "message": "no credits"}))).decide(state, gate_questions)
    assert d.error is not None and "no credits" in d.error

@pytest.mark.parametrize("payload", [["list", "body"], {"code": 0, "data": ["list", "data"]}, {"code": 0, "data": {"model": "jev-1.13.0", "latency_ms": "fast"}}])
def test_jevstation_malformed_bodies_become_error_rows(cfg_factory, gate_questions, state, payload):
    cfg = cfg_factory(name="jevstation", model="jev-1.13.0", extra={"price_per_credit": 0.00198})
    d = JevstationDecider(cfg, client=mock_client(json_handler(payload))).decide(state, gate_questions)
    assert d.error is not None and d.answers == {}
