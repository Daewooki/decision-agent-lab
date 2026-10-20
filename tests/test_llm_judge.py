# tests/test_llm_judge.py
import json
import pytest
from dalab.backends.llm_judge import LLMJudgeDecider, judge_prompt, judge_schema, parse_judge
from dalab.questions import Score

class _Usage:
    def __init__(self, i, o): self.input_tokens, self.output_tokens = i, o

class _Resp:
    def __init__(self, text, model="gpt-6-luna", usage=(500, 40)):
        self.output_text, self.model = text, model
        self.usage = _Usage(*usage) if usage is not None else None

class FakeResponses:
    def __init__(self, text, exc=None, usage=(500, 40)):
        self.calls, self._text, self._exc, self._usage = [], text, exc, usage
        self.responses = self
    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._exc: raise self._exc
        return _Resp(self._text, usage=self._usage)

GOOD = json.dumps({"action_class": {"read_only": 0.9, "local_write": 0.1, "outward": 0, "unclear": 0}, "is_destructive": 0.05})

def _decider(cfg_factory, client):
    cfg = cfg_factory(name="llm_judge", model="gpt-6-luna", endpoint=None, price_in=0.10, price_out=0.50)
    dec = LLMJudgeDecider(cfg, client=client)
    dec._sleep = lambda s: None
    return dec

def test_schema_is_strict(gate_questions):
    s = judge_schema(gate_questions)
    assert s["additionalProperties"] is False and s["required"] == ["action_class", "is_destructive"]
    ac = s["properties"]["action_class"]
    assert ac["additionalProperties"] is False and ac["required"] == ["read_only", "local_write", "outward", "unclear"]
    assert s["properties"]["is_destructive"] == {"type": "number", "minimum": 0, "maximum": 1}

def test_prompt_lists_options(gate_questions, state):
    p = judge_prompt(state, gate_questions)
    assert state in p and "- action_class" in p and "    - read_only: Only reads" in p and "- is_destructive" in p

def test_parse_normalizes_and_picks_argmax(gate_questions):
    obj = {"action_class": {"read_only": 0.6, "local_write": 0.6, "outward": 0.0, "unclear": 0.0}, "is_destructive": 1.4}
    a = parse_judge(obj, gate_questions)
    assert a["action_class"].choice == "read_only" and abs(sum(a["action_class"].probabilities.values()) - 1.0) < 1e-9
    assert a["action_class"].probabilities["read_only"] == 0.5 and a["action_class"].confidence_source == "computed"
    assert a["is_destructive"].noul == 1.0

def test_parse_score():
    qs = {"sev": Score("How severe?", ("low", "mid", "high"))}
    a = parse_judge({"sev": {"0": 0.5, "1": 0.5, "2": 0.0}}, qs)
    assert a["sev"].score == 0.5 and a["sev"].probabilities == {"0": 0.5, "1": 0.5, "2": 0.0}

def test_decide_roundtrip_with_cost(cfg_factory, gate_questions, state):
    fake = FakeResponses(GOOD)
    d = _decider(cfg_factory, fake).decide(state, gate_questions)
    assert fake.calls[0]["model"] == "gpt-6-luna" and fake.calls[0]["text"]["format"]["type"] == "json_schema"
    assert fake.calls[0]["text"]["format"]["strict"] is True
    assert d.error is None and d.answers["action_class"].choice == "read_only" and d.model_reported == "gpt-6-luna"
    assert abs(d.cost_usd - (500 / 1e6 * 0.10 + 40 / 1e6 * 0.50)) < 1e-12
    assert d.extra["attempts"] == 1 and "wall_ms" in d.extra

def test_missing_usage_leaves_cost_unknown(cfg_factory, gate_questions, state):
    d = _decider(cfg_factory, FakeResponses(GOOD, usage=None)).decide(state, gate_questions)
    assert d.error is None and d.answers["action_class"].choice == "read_only"
    assert d.usage_input_tokens is None and d.cost_usd is None

def test_non_json_output_is_error(cfg_factory, gate_questions, state):
    d = _decider(cfg_factory, FakeResponses("not json")).decide(state, gate_questions)
    assert d.error is not None and "non-JSON" in d.error

def test_output_missing_a_question_is_an_error_row(cfg_factory, gate_questions, state):
    d = _decider(cfg_factory, FakeResponses(json.dumps({"action_class": {"read_only": 1.0}}))).decide(state, gate_questions)
    assert d.error.startswith("malformed response: KeyError") and d.answers == {}

def test_sdk_errors_are_retried_with_bounded_attempts(cfg_factory, gate_questions, state):
    class Boom(Exception):
        status_code = 503
    fake = FakeResponses(GOOD, exc=Boom("overloaded"))
    d = _decider(cfg_factory, fake).decide(state, gate_questions)
    assert "503" in d.error and d.extra["attempts"] == 4 and len(fake.calls) == 4

def test_sdk_client_is_built_without_its_own_retries(cfg_factory, monkeypatch):
    import openai
    seen = {}
    class FakeOpenAI:
        def __init__(self, **kwargs): seen.update(kwargs)
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    LLMJudgeDecider(cfg_factory(name="llm_judge", model="gpt-6-luna", endpoint=None, price_in=0.10, price_out=0.50, timeout_s=60.0))
    assert seen == {"api_key": "sk-test-0000", "timeout": 60.0, "max_retries": 0}
