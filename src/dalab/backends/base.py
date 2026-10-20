"""Decider: the one interface every decision backend implements."""
from __future__ import annotations

import datetime as dt
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from dalab.decision import Answer, Decision
from dalab.questions import Choice, Question, permute_choices
from dalab.util.env import mask_secrets
from dalab.util.http import BackendError

# A response whose shape drifted must end as an error row, never as an exception that aborts a run (spec section 4).
MALFORMED = (KeyError, TypeError, ValueError, AttributeError, IndexError)


@dataclass(frozen=True)
class BackendConfig:
    name: str
    model: str
    endpoint: str | None
    secret: str | None = field(repr=False)  # keep keys out of tracebacks, --showlocals, notebooks and logs
    price_input_per_m: float
    price_output_per_m: float = 0.0
    max_options: int = 255
    timeout_s: float = 10.0
    extra: dict[str, Any] = field(default_factory=dict)


class Decider(ABC):
    """Template: permute options -> _call (bounded retries) -> _normalize -> coverage check -> cost.

    Errors become Decision.error, never exceptions: HTTP/SDK failures (BackendError) and malformed bodies alike.
    Only _check_limits raises, before any call is made.
    """

    retries: int = 3  # extra attempts after the first (spec section 4: exponential backoff, at most 3)
    backoff_s: float = 0.5
    _sleep = staticmethod(time.sleep)  # injectable, so tests never wait

    def __init__(self, cfg: BackendConfig, location: str = "Seoul, KR"):
        self.cfg = cfg
        self.location = location

    @property
    def name(self) -> str:
        return self.cfg.name

    @property
    def model_requested(self) -> str:
        return self.cfg.model

    def decide(self, state: str, questions: dict[str, Question], *, order_seed: int | None = None) -> Decision:
        qs, order = permute_choices(questions, order_seed)
        self._check_limits(qs)
        ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")
        t0 = time.perf_counter()
        raw: Any = {}
        call_extra: dict[str, Any] = {}
        try:
            raw, latency_ms, server_ms, *rest = self._call(state, qs)
            call_extra = dict(rest[0]) if rest and rest[0] else {}
            model_reported, answers, in_tok, out_tok, extra = self._normalize(raw, qs)
            missing = set(qs) - set(answers)
            if missing:  # a partial answer set must not pass as a clean row: the gate policy could not tell
                raise BackendError(None, f"missing answers for {sorted(missing)}", raw, retryable=False)
            return Decision(
                backend=self.name, model_requested=self.cfg.model, model_reported=model_reported, ts_utc=ts,
                location=self.location, latency_ms=latency_ms, server_latency_ms=server_ms, answers=answers,
                usage_input_tokens=in_tok, usage_output_tokens=out_tok, cost_usd=self.cost_usd(in_tok, out_tok, extra),
                option_order=order, error=None, raw=raw, extra={**call_extra, **extra, "wall_ms": self._wall_ms(t0)},
            )
        except BackendError as e:
            in_tok, out_tok = e.usage or (None, None)
            extra = {**call_extra, **({"attempts": e.attempts} if e.attempts is not None else {}), "wall_ms": self._wall_ms(t0)}
            return self._error_decision(ts, order, self._mask(str(e)), {"error_body": self._mask_json(e.body)}, in_tok, out_tok, extra)
        except MALFORMED as e:
            raw = raw if isinstance(raw, dict) else {"response": raw}  # keep the body so the drift can be inspected
            extra = {**call_extra, "wall_ms": self._wall_ms(t0)}
            return self._error_decision(ts, order, self._mask(f"malformed response: {type(e).__name__}: {e}"), raw, None, None, extra)

    def _error_decision(self, ts: str, order: dict[str, list[str]], error: str, raw: dict[str, Any],
                        in_tok: int | None, out_tok: int | None, extra: dict[str, Any]) -> Decision:
        return Decision(
            backend=self.name, model_requested=self.cfg.model, model_reported=None, ts_utc=ts, location=self.location,
            latency_ms=None, server_latency_ms=None, answers={}, usage_input_tokens=in_tok, usage_output_tokens=out_tok,
            cost_usd=self.cost_usd(in_tok, out_tok, {}) if in_tok is not None else None,
            option_order=order, error=error, raw=raw, extra=extra,
        )

    @staticmethod
    def _wall_ms(t0: float) -> float:
        return (time.perf_counter() - t0) * 1000.0

    def _mask(self, text: str) -> str:
        """Strip this backend's key and every secret-looking env value: error rows are committed artifacts."""
        text = mask_secrets(text)
        return mask_secrets(text, [self.cfg.secret]) if self.cfg.secret else text

    def _mask_json(self, body: Any) -> Any:
        if body is None:
            return None
        try:
            return json.loads(self._mask(json.dumps(body, ensure_ascii=False, default=str)))
        except (TypeError, ValueError):
            return self._mask(str(body))

    def _check_limits(self, questions: dict[str, Question]) -> None:
        for name, q in questions.items():
            if isinstance(q, Choice) and len(q.criteria) > self.cfg.max_options:
                raise ValueError(f"{self.name}: question {name!r} has {len(q.criteria)} options > max {self.cfg.max_options}")

    def cost_usd(self, in_tok: int | None, out_tok: int | None, extra: dict[str, Any]) -> float | None:
        if in_tok is None:
            return None
        return in_tok / 1e6 * self.cfg.price_input_per_m + (out_tok or 0) / 1e6 * self.cfg.price_output_per_m

    @abstractmethod
    def _call(self, state: str, questions: dict[str, Question]) -> tuple[dict[str, Any], float, float | None, dict[str, Any]]:
        """Return (raw response, client latency ms of the final attempt, server-reported latency ms or None,
        call extras such as {"attempts": n}). A 3-tuple without extras is accepted. Raise BackendError on failure."""

    @abstractmethod
    def _normalize(
        self, raw: dict[str, Any], questions: dict[str, Question]
    ) -> tuple[str | None, dict[str, Answer], int | None, int | None, dict[str, Any]]:
        """Return (model_reported, answers, input_tokens, output_tokens, extra). Raise BackendError if malformed."""
