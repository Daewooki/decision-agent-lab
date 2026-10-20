# tests/test_openai_decisions.py
import time
import httpx
import pytest
from dalab.backends.openai_decisions import OpenAIDecisionsDecider, from_openai_answers, to_openai_questions
from dalab.questions import Score
from dalab.util.http import BackendError

# Recorded 2026-10-20 from client.decisions.create(model="gpt-6-luna", ...)
OPENAI_RESPONSE = {
    "answers": [
        {"choice": "read_only", "confidence": 1.0, "name": "action_class",
         "probabilities": [{"probability": 1.0, "value": "read_only"}, {"probability": 0.0, "value": "local_write"},
                           {"probability": 0.0, "value": "outward"}, {"probability": 0.0, "value": "unclear"}], "type": "choice"},
        {"name": "is_destructive", "probability": 0.0, "type": "predicate"},
    ],
    "model": "gpt-6-luna",
    "usage": {"input_tokens": 344, "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
              "output_tokens": 0, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 344},
}
# Shape of openai 3.35.1 types/decision.py AnswerAnswerResourceScoreProbability: {label, probability, value:int}
SCORE_ANSWER = {"type": "score", "name": "sev", "score": 0.3, "confidence": 0.12,
                "probabilities": [{"label": "low", "probability": 0.7, "value": 0}, {"label": "high", "probability": 0.3, "value": 1}]}
REFUSAL_RESPONSE = {"answers": [OPENAI_RESPONSE["answers"][0], {"name": "is_destructive", "type": "refusal"}],
                    "model": "gpt-6-luna", "usage": OPENAI_RESPONSE["usage"]}

class _Resp:
    def __init__(self, d): self._d = d
    def model_dump(self): return self._d

class FakeClient:
    """`exc` is an exception raised on every call, or a list of exceptions/payloads consumed one call at a time."""
    def __init__(self, payload=OPENAI_RESPONSE, exc=None):
        self.calls, self._payload = [], payload
        self._always = exc if isinstance(exc, BaseException) else None
        self._script = list(exc) if isinstance(exc, list) else []
        self.decisions = self
    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._always is not None:
            raise self._always
        if self._script:
            step = self._script.pop(0)
            if isinstance(step, BaseException):
                raise step
            return _Resp(step)
        return _Resp(self._payload)

class Boom(Exception):
    def __init__(self, msg, status_code):
        super().__init__(msg); self.status_code = status_code

def _decider(cfg_factory, client):
    cfg = cfg_factory(name="openai_decisions", model="gpt-6-luna", endpoint=None, price_in=0.10)
    dec = OpenAIDecisionsDecider(cfg, client=client)
    dec._sleep = lambda s: None
    return dec

def test_question_translation(gate_questions):
    qs = dict(gate_questions); qs["sev"] = Score("How severe?", ("low", "high"))
    out = to_openai_questions(qs)
    assert out[0] == {"type": "choice", "name": "action_class", "instructions": gate_questions["action_class"].instructions,
                      "choices": [{"value": k, "description": v} for k, v in gate_questions["action_class"].criteria.items()]}
    assert out[1] == {"type": "predicate", "name": "is_destructive", "instructions": gate_questions["is_destructive"].instructions}
    assert out[2] == {"type": "score", "name": "sev", "instructions": "How severe?",
                      "levels": [{"label": "low", "description": "low"}, {"label": "high", "description": "high"}]}

def test_answer_translation():
    a = from_openai_answers(OPENAI_RESPONSE["answers"])
    assert a["action_class"].choice == "read_only" and a["action_class"].probabilities["read_only"] == 1.0
    assert a["action_class"].confidence == 1.0 and a["action_class"].confidence_source == "reported"
    assert a["is_destructive"].type == "noul" and a["is_destructive"].noul == 0.0

def test_score_keys_match_system_one_levels():
    a = from_openai_answers([SCORE_ANSWER])
    assert a["sev"].score == 0.3 and a["sev"].probabilities == {"0": 0.7, "1": 0.3}  # level index, like System One's "0".."n-1"
    assert a["sev"].confidence == 0.12 and a["sev"].confidence_source == "reported"
    unreported = from_openai_answers([{**SCORE_ANSWER, "confidence": None}])
    assert unreported["sev"].confidence_source == "computed" and 0.11 < unreported["sev"].confidence < 0.13

def test_refusal_is_an_error_without_a_made_up_reason():
    with pytest.raises(BackendError) as ei:
        from_openai_answers([{"type": "refusal", "name": "x"}])  # the SDK refusal object has no reason field
    assert str(ei.value) == "model refused question 'x'" and ei.value.body == {"type": "refusal", "name": "x"}

def test_decide_roundtrip(cfg_factory, gate_questions, state):
    fake = FakeClient()
    d = _decider(cfg_factory, fake).decide(state, gate_questions, order_seed=1)
    assert fake.calls[0]["model"] == "gpt-6-luna" and fake.calls[0]["input"] == state
    sent = [c["value"] for c in fake.calls[0]["questions"][0]["choices"]]
    assert sent == d.option_order["action_class"]
    assert d.error is None and d.model_reported == "gpt-6-luna" and d.usage_input_tokens == 344
    assert abs(d.cost_usd - 344 / 1e6 * 0.10) < 1e-12 and d.raw["model"] == "gpt-6-luna"
    assert d.extra["attempts"] == 1 and d.extra["wall_ms"] >= d.latency_ms

def test_refused_call_is_an_error_row_that_keeps_its_cost(cfg_factory, gate_questions, state):
    d = _decider(cfg_factory, FakeClient(REFUSAL_RESPONSE)).decide(state, gate_questions)
    assert d.error == "model refused question 'is_destructive'" and d.answers == {}
    assert d.usage_input_tokens == 344 and abs(d.cost_usd - 344 / 1e6 * 0.10) < 1e-12  # billed although refused

def test_sdk_exception_becomes_error_after_bounded_retries(cfg_factory, gate_questions, state):
    fake = FakeClient(exc=Boom("rate limited", 429))
    dec = _decider(cfg_factory, fake)
    slept = []
    dec._sleep = slept.append
    d = dec.decide(state, gate_questions)
    assert d.error is not None and "429" in d.error and d.cost_usd is None
    assert d.extra["attempts"] == 4 and len(fake.calls) == 4 and len(slept) == 3

def test_non_retryable_sdk_error_is_not_retried(cfg_factory, gate_questions, state):
    fake = FakeClient(exc=Boom("bad key", 401))
    d = _decider(cfg_factory, fake).decide(state, gate_questions)
    assert "401" in d.error and d.extra["attempts"] == 1 and len(fake.calls) == 1

def test_connection_error_is_retried(cfg_factory, gate_questions, state):
    import openai
    conn = openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.invalid/v1/decisions"))
    fake = FakeClient(exc=[conn, OPENAI_RESPONSE])
    d = _decider(cfg_factory, fake).decide(state, gate_questions)
    assert d.error is None and d.extra["attempts"] == 2 and len(fake.calls) == 2

def test_latency_covers_only_the_final_attempt(cfg_factory, gate_questions, state):
    class SlowThenFast(FakeClient):
        def create(self, **kwargs):
            if not self.calls:
                self.calls.append(kwargs)
                time.sleep(0.2)
                raise Boom("overloaded", 503)
            return super().create(**kwargs)
    d = _decider(cfg_factory, SlowThenFast()).decide(state, gate_questions)
    assert d.error is None and d.extra["attempts"] == 2
    assert d.latency_ms < 100 and d.extra["wall_ms"] >= 200

def test_sdk_client_is_built_without_its_own_retries(cfg_factory, monkeypatch):
    import openai
    seen = {}
    class FakeOpenAI:
        def __init__(self, **kwargs): seen.update(kwargs)
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    cfg = cfg_factory(name="openai_decisions", model="gpt-6-luna", endpoint=None, price_in=0.10, timeout_s=30.0)
    OpenAIDecisionsDecider(cfg)
    assert seen == {"api_key": "sk-test-0000", "timeout": 30.0, "max_retries": 0}

def test_partial_or_malformed_answers_become_error_rows(cfg_factory, gate_questions, state):
    partial = {**OPENAI_RESPONSE, "answers": OPENAI_RESPONSE["answers"][:1]}
    d = _decider(cfg_factory, FakeClient(partial)).decide(state, gate_questions)
    assert d.error == "missing answers for ['is_destructive']"
    broken = {**OPENAI_RESPONSE, "answers": [{"type": "predicate", "name": "is_destructive"}]}  # no probability
    d = _decider(cfg_factory, FakeClient(broken)).decide(state, gate_questions)
    assert d.error.startswith("malformed response: KeyError") and d.raw == broken
    d = _decider(cfg_factory, FakeClient({**OPENAI_RESPONSE, "answers": {"not": "a list"}})).decide(state, gate_questions)
    assert d.error is not None and d.answers == {}
