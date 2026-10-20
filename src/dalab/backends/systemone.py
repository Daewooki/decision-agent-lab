"""Backends that speak TypeSafe's /v1/systemone wire format: typesafe, openrouter, upstage, ollama, and the jevstation relay."""
from __future__ import annotations

from typing import Any

import httpx

from dalab.backends.base import BackendConfig, Decider
from dalab.decision import Answer, answer_with_confidence
from dalab.questions import Question, to_systemone
from dalab.util.http import BackendError, make_client, post_json


def _probs(a: dict[str, Any]) -> dict[str, float]:
    return {str(k): float(v) for k, v in (a.get("probabilities") or {}).items()}


def parse_systemone_answers(answers: dict[str, Any]) -> dict[str, Answer]:
    out: dict[str, Answer] = {}
    for name, a in answers.items():
        if not isinstance(a, dict):
            raise BackendError(None, f"answer {name!r} is not an object", a, retryable=False)
        kind = a.get("type") or ("choice" if "choice" in a else "noul" if "noul" in a else "score" if "score" in a else None)
        if kind == "choice":
            out[name] = answer_with_confidence("choice", _probs(a), a.get("confidence"), choice=a.get("choice"))
        elif kind == "noul":
            out[name] = Answer("noul", noul=float(a["noul"]))
        elif kind == "score":
            out[name] = answer_with_confidence("score", _probs(a), a.get("confidence"), score=float(a["score"]))
        else:
            raise BackendError(None, f"unknown answer shape for {name!r}", a, retryable=False)
    return out


class SystemOneDecider(Decider):
    def __init__(self, cfg: BackendConfig, location: str = "Seoul, KR", client: httpx.Client | None = None):
        super().__init__(cfg, location)
        self.client = client or make_client(cfg.timeout_s)

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.cfg.secret:
            h["Authorization"] = f"Bearer {self.cfg.secret}"
        return h

    def _body(self, state: str, questions: dict[str, Question]) -> dict[str, Any]:
        return {"model": self.cfg.model, "state": state, "questions": to_systemone(questions)}

    def _call(self, state, questions):
        if not self.cfg.endpoint:
            raise BackendError(None, f"{self.name}: no endpoint configured", retryable=False)
        raw, ms, attempts = post_json(self.client, self.cfg.endpoint, headers=self._headers(), json=self._body(state, questions),
                                      retries=self.retries, backoff_s=self.backoff_s, sleep=self._sleep)
        return raw, ms, None, {"attempts": attempts}

    def _normalize(self, raw, questions):
        if not isinstance(raw, dict) or not isinstance(raw.get("answers"), dict):
            raise BackendError(None, "response has no 'answers' object", raw, retryable=False)
        usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        return raw.get("model"), parse_systemone_answers(raw["answers"]), usage.get("input_tokens"), usage.get("output_tokens"), {}


class JevstationDecider(SystemOneDecider):
    """jevstation.com wraps System One responses as {code, message, data} and bills credits per call."""

    def _call(self, state, questions):
        raw, ms, _, call_extra = super()._call(state, questions)
        data = raw.get("data") if isinstance(raw, dict) else None
        server = data.get("latency_ms") if isinstance(data, dict) else None
        return raw, ms, (float(server) if server is not None else None), call_extra

    def _normalize(self, raw, questions):
        if not isinstance(raw, dict):
            raise BackendError(None, "jevstation response is not a JSON object", raw, retryable=False)
        if raw.get("code", 0) != 0:
            raise BackendError(None, f"jevstation error: {raw.get('message')}", raw, retryable=False)
        data = raw.get("data") or {}
        model, answers, in_tok, out_tok, _ = super()._normalize(data, questions)
        credits = data.get("credits") or {}
        extra = {"credits_charged": credits.get("charged"), "credits_remaining": credits.get("remaining"),
                 "relay_latency_ms": data.get("latency_ms")}
        return model, answers, in_tok, out_tok, extra

    def cost_usd(self, in_tok, out_tok, extra):
        per_credit = self.cfg.extra.get("price_per_credit")
        charged = (extra or {}).get("credits_charged")
        if per_credit is None or charged is None:
            return None
        return float(charged) * float(per_credit)
