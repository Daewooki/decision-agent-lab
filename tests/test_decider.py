# tests/test_decider.py -- behaviour of the Decider template itself, with a minimal in-memory backend
import json
import pytest
from dalab.backends.base import BackendConfig, Decider
from dalab.decision import Answer
from dalab.questions import Noul
from dalab.util.http import BackendError

class ScriptedDecider(Decider):
    """_call returns the scripted raw body (or raises it); _normalize does a naive parse so shape drift surfaces."""
    def __init__(self, raw, *, secret="sk-test-0000", call_extra=None):
        super().__init__(BackendConfig("scripted", "m1", None, secret, 1.0))
        self.raw, self.call_extra = raw, call_extra
    def _call(self, state, questions):
        if isinstance(self.raw, Exception):
            raise self.raw
        return (self.raw, 1.5, None) if self.call_extra is None else (self.raw, 1.5, None, self.call_extra)
    def _normalize(self, raw, questions):
        answers = {n: Answer("noul", noul=float(raw["answers"][n]["noul"])) for n in raw["answers"]}
        return raw["model"], answers, raw["usage"]["input_tokens"], 0, {}

QS = {"a": Noul("a?"), "b": Noul("b?")}
GOOD = {"model": "m1", "answers": {"a": {"noul": 0.1}, "b": {"noul": 0.9}}, "usage": {"input_tokens": 10}}

def test_config_repr_hides_secret():
    cfg = BackendConfig("typesafe", "jev-1.14.0", "https://api.typesafe.ai/v1/systemone", "sk-live-0123456789", 0.042)
    assert "sk-live-0123456789" not in repr(cfg) and "sk-live-0123456789" not in repr(ScriptedDecider(GOOD).cfg)
    assert cfg.secret == "sk-live-0123456789"  # still usable, just not printed

def test_three_tuple_call_is_accepted_and_extras_are_recorded():
    d = ScriptedDecider(GOOD).decide("s", QS)
    assert d.error is None and d.answers["b"].noul == 0.9 and d.cost_usd == 10 / 1e6
    assert set(d.extra) == {"wall_ms"} and d.extra["wall_ms"] >= 0
    d = ScriptedDecider(GOOD, call_extra={"attempts": 2}).decide("s", QS)
    assert d.extra["attempts"] == 2 and "wall_ms" in d.extra

@pytest.mark.parametrize("raw", [
    {"model": "m1", "answers": {"a": {}, "b": {"noul": 0.9}}, "usage": {"input_tokens": 1}},          # noul without value
    {"model": "m1", "answers": {"a": {"noul": None}, "b": {"noul": 0.9}}, "usage": {"input_tokens": 1}},  # noul null
    {"model": "m1", "answers": [{"noul": 0.1}], "usage": {"input_tokens": 1}},                         # answers as a list
    {"model": "m1", "answers": {"a": {"noul": "high"}, "b": {"noul": 0.9}}, "usage": {"input_tokens": 1}},  # non-numeric
    ["not", "an", "object"],
    {"model": "m1", "answers": {"a": {"noul": 0.1}, "b": {"noul": 0.9}}, "usage": None},               # usage null
])
def test_malformed_bodies_become_error_rows_not_exceptions(raw):
    d = ScriptedDecider(raw).decide("s", QS)
    assert d.error is not None and d.error.startswith("malformed response:") and d.answers == {} and d.cost_usd is None
    assert d.raw == (raw if isinstance(raw, dict) else {"response": raw})  # the body is kept for inspection

def test_partial_answer_set_is_an_error_row():
    d = ScriptedDecider({"model": "m1", "answers": {"a": {"noul": 0.1}}, "usage": {"input_tokens": 10}}).decide("s", QS)
    assert d.error == "missing answers for ['b']" and d.answers == {} and d.raw["error_body"]["model"] == "m1"

def test_error_rows_are_masked():
    leak = BackendError(401, "invalid key sk-test-0000 for OPENAI_API_KEY", {"detail": "token sk-test-0000 rejected"})
    d = ScriptedDecider(leak).decide("s", QS)
    assert "sk-test-0000" not in d.error and "***" in d.error and "401" in d.error
    assert "sk-test-0000" not in json.dumps(d.raw) and d.raw["error_body"]["detail"] == "token *** rejected"

def test_error_rows_mask_env_secrets_too(monkeypatch):
    monkeypatch.setenv("OTHER_API_KEY", "env-secret-value-1")
    d = ScriptedDecider(BackendError(None, "echo env-secret-value-1"), secret=None).decide("s", QS)
    assert d.error == "echo ***"

def test_billed_failure_keeps_usage_and_cost():
    e = BackendError(None, "model refused question 'a'", {"type": "refusal"})
    e.usage, e.attempts = (344, 0), 1
    d = ScriptedDecider(e).decide("s", QS)
    assert d.error is not None and d.usage_input_tokens == 344 and d.cost_usd == 344 / 1e6 and d.extra["attempts"] == 1

def test_option_limit_still_raises_before_any_call():
    from dalab.questions import Choice
    dec = ScriptedDecider(GOOD)
    dec.cfg = BackendConfig("scripted", "m1", None, None, 1.0, max_options=2)
    with pytest.raises(ValueError, match="3 options"):
        dec.decide("s", {"c": Choice("?", {"x": "1", "y": "2", "z": "3"})})
