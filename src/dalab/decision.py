"""Normalized decision record shared by every backend."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Answer:
    type: str  # "choice" | "noul" | "score"
    choice: str | None = None
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    confidence_source: str | None = None  # "reported" (backend gave it) | "computed" (concentration()) | None (unknown)
    noul: float | None = None
    score: float | None = None


def concentration(probs: dict[str, float]) -> float:
    """1 - normalized Shannon entropy. 1.0 = all mass on one option, 0.0 = uniform.

    Matches the `confidence` that System One backends report for choice answers.
    Raises ValueError on an empty distribution: there is nothing to be confident about.
    """
    n = len(probs)
    if n == 0:
        raise ValueError("concentration() needs at least one probability")
    if n == 1:
        return 1.0
    h = -sum(p * math.log(p) for p in probs.values() if p > 0)
    return max(0.0, min(1.0, 1.0 - h / math.log(n)))


def answer_with_confidence(kind: str, probabilities: dict[str, float] | None, confidence: Any, **fields: Any) -> Answer:
    """Build a choice/score Answer whose confidence is the backend's when reported, else derived from the
    distribution, else None. A missing distribution never turns into a fabricated full-confidence value."""
    probs = dict(probabilities) if probabilities else None
    if confidence is not None:
        conf, source = float(confidence), "reported"
    elif probs:
        conf, source = concentration(probs), "computed"
    else:
        conf, source = None, None
    return Answer(kind, probabilities=probs, confidence=conf, confidence_source=source, **fields)


@dataclass
class Decision:
    backend: str
    model_requested: str
    model_reported: str | None
    ts_utc: str
    location: str
    latency_ms: float | None  # client round-trip of the final HTTP attempt
    server_latency_ms: float | None  # only if the backend reports one
    answers: dict[str, Answer]
    usage_input_tokens: int | None
    usage_output_tokens: int | None
    cost_usd: float | None  # from config/models.toml prices; None if unknown
    option_order: dict[str, list[str]]
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)  # backend-specific (credits, route, ...) + attempts, wall_ms

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Decision":
        d = dict(d)
        d["answers"] = {k: Answer(**v) for k, v in d["answers"].items()}
        return cls(**d)
