"""Small, dependency-light statistics used by both experiments."""
from __future__ import annotations

from collections.abc import Sequence
from math import comb

import numpy as np


def accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    y_true, y_pred = list(y_true), list(y_pred)
    return float(np.mean([a == b for a, b in zip(y_true, y_pred, strict=True)])) if y_true else float("nan")


def macro_f1(y_true: Sequence, y_pred: Sequence, labels: Sequence) -> float:
    y_true, y_pred = list(y_true), list(y_pred)
    f1s = []
    for lab in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred, strict=True) if a == lab and b == lab)
        fp = sum(1 for a, b in zip(y_true, y_pred, strict=True) if a != lab and b == lab)
        fn = sum(1 for a, b in zip(y_true, y_pred, strict=True) if a == lab and b != lab)
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def bootstrap_ci(values: Sequence[float], *, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2))))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b (A right, B wrong) and c (A wrong, B right)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return float(min(1.0, 2 * tail))


def ece(probs: Sequence[float], labels: Sequence[int], n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins on the predicted probability."""
    p = np.asarray(list(probs), dtype=float)
    y = np.asarray(list(labels), dtype=float)
    if p.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = which == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def brier(probs: Sequence[float], labels: Sequence[int]) -> float:
    p = np.asarray(list(probs), dtype=float)
    y = np.asarray(list(labels), dtype=float)
    return float(np.mean((p - y) ** 2)) if p.size else float("nan")
