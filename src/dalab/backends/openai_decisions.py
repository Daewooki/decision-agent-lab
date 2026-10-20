"""OpenAI Decisions API (POST /v1/decisions) through the official SDK, translated to/from System One shapes."""
from __future__ import annotations

from typing import Any

from dalab.backends.base import BackendConfig, Decider
from dalab.backends.openai_sdk import make_openai_client, to_backend_error
from dalab.decision import Answer, answer_with_confidence
from dalab.questions import Choice, Noul, Question, Score
from dalab.util.http import BackendError, with_retries


def to_openai_questions(questions: dict[str, Question]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, q in questions.items():
        if isinstance(q, Choice):
            out.append({"type": "choice", "name": name, "instructions": q.instructions,
                        "choices": [{"value": k, "description": v} for k, v in q.criteria.items()]})
        elif isinstance(q, Noul):
            out.append({"type": "predicate", "name": name, "instructions": q.instructions})
        elif isinstance(q, Score):
            out.append({"type": "score", "name": name, "instructions": q.instructions,
                        "levels": [{"label": c, "description": c} for c in q.criteria]})
        else:
            raise TypeError(f"unsupported question type: {type(q)!r}")
    return out


def _probs(items: list[dict[str, Any]]) -> dict[str, float]:
    """Choice items carry {value, probability}; score items {label, probability, value:int}. Keys follow `value`
    so score keys are "0".."n-1" exactly like System One's."""
    out: dict[str, float] = {}
    for i, p in enumerate(items or []):
        key = p.get("value", p.get("label", i))
        out[str(key)] = float(p["probability"])
    return out


def from_openai_answers(answers: list[dict[str, Any]]) -> dict[str, Answer]:
    out: dict[str, Answer] = {}
    for a in answers:
        kind, name = a.get("type"), a.get("name")
        if kind == "refusal":  # the SDK refusal object carries only `type` and `name` (spec section 4: refusal -> error)
            raise BackendError(None, f"model refused question {name!r}", a, retryable=False)
        if kind == "choice":
            out[name] = answer_with_confidence("choice", _probs(a.get("probabilities", [])), a.get("confidence"), choice=a.get("choice"))
        elif kind == "predicate":
            out[name] = Answer("noul", noul=float(a["probability"]))
        elif kind == "score":
            out[name] = answer_with_confidence("score", _probs(a.get("probabilities", [])), a.get("confidence"), score=float(a["score"]))
        else:
            raise BackendError(None, f"unknown answer type {kind!r} for {name!r}", a, retryable=False)
    return out


class OpenAIDecisionsDecider(Decider):
    def __init__(self, cfg: BackendConfig, location: str = "Seoul, KR", client: Any = None):
        super().__init__(cfg, location)
        self.client = client if client is not None else make_openai_client(cfg.secret, cfg.timeout_s)

    def _call(self, state, questions):
        payload = {"model": self.cfg.model, "input": state, "questions": to_openai_questions(questions)}

        def attempt():
            try:
                return self.client.decisions.create(**payload)
            except BackendError:
                raise
            except Exception as e:  # openai.APIStatusError and friends carry status_code
                raise to_backend_error(e) from e

        resp, ms, attempts = with_retries(attempt, retries=self.retries, backoff_s=self.backoff_s, sleep=self._sleep)
        raw = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
        return raw, ms, None, {"attempts": attempts}

    def _normalize(self, raw, questions):
        usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        in_tok, out_tok = usage.get("input_tokens"), usage.get("output_tokens")
        try:
            answers = from_openai_answers(raw.get("answers") or [])
        except BackendError as e:
            e.usage = (in_tok, out_tok)  # a refused call is still billed: the error row keeps its tokens and cost
            raise
        return raw.get("model"), answers, in_tok, out_tok, {}
