"""Typed questions in TypeSafe's System One vocabulary (choice / noul / score)."""
from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass(frozen=True, eq=False)
class Choice:
    instructions: str
    criteria: dict[str, str]  # insertion order == option order sent to the model
    type: str = field(default="choice", init=False)


@dataclass(frozen=True, eq=False)
class Noul:
    instructions: str
    type: str = field(default="noul", init=False)


@dataclass(frozen=True, eq=False)
class Score:
    instructions: str
    criteria: tuple[str, ...]  # level descriptions, index 0..n-1
    type: str = field(default="score", init=False)


Question = Choice | Noul | Score


def permute_choices(
    questions: dict[str, Question], seed: int | None
) -> tuple[dict[str, Question], dict[str, list[str]]]:
    """Return a copy with every Choice's options reordered by `seed`, plus the order used.

    seed=None keeps the original order. The same seed always yields the same order.
    """
    rng = random.Random(seed) if seed is not None else None
    out: dict[str, Question] = {}
    order: dict[str, list[str]] = {}
    for name, q in questions.items():
        if isinstance(q, Choice):
            keys = list(q.criteria)
            if rng is not None:
                rng.shuffle(keys)
            order[name] = keys
            out[name] = Choice(q.instructions, {k: q.criteria[k] for k in keys})
        else:
            out[name] = q
    return out, order


def to_systemone(questions: dict[str, Question]) -> dict[str, dict]:
    """Serialize to the `questions` object of POST /v1/systemone."""
    wire: dict[str, dict] = {}
    for name, q in questions.items():
        if isinstance(q, Choice):
            wire[name] = {"type": "choice", "instructions": q.instructions, "criteria": dict(q.criteria)}
        elif isinstance(q, Noul):
            wire[name] = {"type": "noul", "instructions": q.instructions}
        elif isinstance(q, Score):
            wire[name] = {"type": "score", "instructions": q.instructions, "criteria": list(q.criteria)}
        else:
            raise TypeError(f"unsupported question type: {type(q)!r}")
    return wire
