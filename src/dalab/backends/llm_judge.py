"""Baseline: a generative LLM answering the same questions with self-reported probabilities (OpenAI Responses API, strict JSON)."""
from __future__ import annotations

import json
from typing import Any

from dalab.backends.base import BackendConfig, Decider
from dalab.backends.openai_sdk import make_openai_client, to_backend_error
from dalab.decision import Answer, concentration
from dalab.questions import Choice, Noul, Question, Score
from dalab.util.http import BackendError, with_retries

JUDGE_SYSTEM = (
    "You are a careful, calibrated reviewer. You receive a STATE and several QUESTIONS about it. "
    "Answer every question with probabilities between 0 and 1 that reflect how likely each answer is. "
    "For yes/no questions give the probability of 'yes'. For choice and level questions give a probability "
    "for every option; they should sum to 1. Respond only with JSON matching the schema."
)

_UNIT = {"type": "number", "minimum": 0, "maximum": 1}


def judge_schema(questions: dict[str, Question]) -> dict[str, Any]:
    props: dict[str, Any] = {}
    for name, q in questions.items():
        if isinstance(q, Noul):
            props[name] = dict(_UNIT)
        elif isinstance(q, Choice):
            props[name] = {"type": "object", "properties": {k: dict(_UNIT) for k in q.criteria},
                           "required": list(q.criteria), "additionalProperties": False}
        elif isinstance(q, Score):
            keys = [str(i) for i in range(len(q.criteria))]
            props[name] = {"type": "object", "properties": {k: dict(_UNIT) for k in keys},
                           "required": keys, "additionalProperties": False}
        else:
            raise TypeError(f"unsupported question type: {type(q)!r}")
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def judge_prompt(state: str, questions: dict[str, Question]) -> str:
    lines = ["STATE:", state, "", "QUESTIONS:"]
    for name, q in questions.items():
        if isinstance(q, Noul):
            lines.append(f"- {name} (probability of yes): {q.instructions}")
        elif isinstance(q, Choice):
            lines.append(f"- {name} (one option; probability per option): {q.instructions}")
            lines.extend(f"    - {k}: {v}" for k, v in q.criteria.items())
        elif isinstance(q, Score):
            lines.append(f"- {name} (levels 0..{len(q.criteria) - 1}; probability per level): {q.instructions}")
            lines.extend(f"    - {i}: {c}" for i, c in enumerate(q.criteria))
    return "\n".join(lines)


def _clip(x: Any) -> float:
    return max(0.0, min(1.0, float(x)))


def _normalize(probs: dict[str, float]) -> dict[str, float]:
    total = sum(probs.values())
    n = len(probs)
    if total <= 0:
        return {k: 1.0 / n for k in probs}
    return {k: v / total for k, v in probs.items()}


def parse_judge(obj: dict[str, Any], questions: dict[str, Question]) -> dict[str, Answer]:
    out: dict[str, Answer] = {}
    for name, q in questions.items():
        v = obj[name]
        if isinstance(q, Noul):
            out[name] = Answer("noul", noul=_clip(v))
        elif isinstance(q, Choice):
            probs = _normalize({k: _clip(v.get(k, 0.0)) for k in q.criteria})
            out[name] = Answer("choice", choice=max(probs, key=probs.get), probabilities=probs,
                               confidence=concentration(probs), confidence_source="computed")
        elif isinstance(q, Score):
            probs = _normalize({str(i): _clip(v.get(str(i), 0.0)) for i in range(len(q.criteria))})
            out[name] = Answer("score", score=sum(int(k) * p for k, p in probs.items()), probabilities=probs,
                               confidence=concentration(probs), confidence_source="computed")
    return out


class LLMJudgeDecider(Decider):
    def __init__(self, cfg: BackendConfig, location: str = "Seoul, KR", client: Any = None):
        super().__init__(cfg, location)
        self.client = client if client is not None else make_openai_client(cfg.secret, cfg.timeout_s)

    def _call(self, state, questions):
        payload = {
            "model": self.cfg.model,
            "input": [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": judge_prompt(state, questions)}],
            "text": {"format": {"type": "json_schema", "name": "judge", "schema": judge_schema(questions), "strict": True}},
        }

        def attempt():
            try:
                return self.client.responses.create(**payload)
            except BackendError:
                raise
            except Exception as e:
                raise to_backend_error(e) from e

        resp, ms, attempts = with_retries(attempt, retries=self.retries, backoff_s=self.backoff_s, sleep=self._sleep)
        usage = getattr(resp, "usage", None)  # may be None; the row then carries no tokens and no cost
        raw = {"model": getattr(resp, "model", None), "output_text": getattr(resp, "output_text", None),
               "usage": {"input_tokens": getattr(usage, "input_tokens", None), "output_tokens": getattr(usage, "output_tokens", None)}}
        return raw, ms, None, {"attempts": attempts}

    def _normalize(self, raw, questions):
        try:
            obj = json.loads(raw["output_text"])
        except (ValueError, TypeError):
            raise BackendError(None, "judge returned non-JSON output", raw, retryable=False) from None
        usage = raw.get("usage") or {}
        return raw.get("model"), parse_judge(obj, questions), usage.get("input_tokens"), usage.get("output_tokens"), {}
