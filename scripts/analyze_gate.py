#!/usr/bin/env python
"""Turn results/raw/gate/*.jsonl into summary tables and a markdown report.

Usage: uv run python scripts/analyze_gate.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import pandas as pd

from dalab.bench.cost import summarize_cost
from dalab.gate.analysis import (calibration, confusion, headline_rates, injection_table, load_rows,
                                 model_band_rates, pair_gap, predict, robustness, rules_only_predict,
                                 safety_cost_table, stage_attribution, version_compare)
from dalab.gate.datasets import load_all, records_by_id
from dalab.gate.policy import Policy, load_policy


def collect(results_dir: Path, data_root: Path, policy: Policy) -> pd.DataFrame:
    records = load_all(data_root)
    by_id = records_by_id(records)
    frames = [predict(by_id, load_rows(p), policy) for p in sorted(Path(results_dir).glob("*.jsonl"))]
    frames = [f for f in frames if len(f)]
    if frames:
        frames.append(rules_only_predict(records, policy))  # code-only reference corner for the trade-off
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def cost_summary(results_dir: Path) -> pd.DataFrame:
    """Per-backend cost and latency over base-condition calls only (clean per-decision numbers)."""
    rows = []
    for p in sorted(Path(results_dir).glob("*.jsonl")):
        rows += [r for r in load_rows(p) if r["condition"] == "base"]
    return summarize_cost(rows) if rows else pd.DataFrame()


def _to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavored markdown table (no tabulate dependency)."""
    flat = df.reset_index()
    headers = [str(c) for c in flat.columns]
    rows = [["" if pd.isna(v) else str(v) for v in rec] for rec in flat.to_numpy()]
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _table(fn: Callable[[pd.DataFrame], pd.DataFrame], df: pd.DataFrame) -> pd.DataFrame:
    """Build a summary table, tolerating runs that cover only a subset of conditions.

    robustness/injection/pairs index their output by columns that only exist once at least one
    matching row is produced; on a base-only (or otherwise partial) run an empty table is the
    correct summary, so we return one instead of failing the whole report.
    """
    try:
        return fn(df)
    except (KeyError, ValueError):
        return pd.DataFrame()


def write_summaries(df: pd.DataFrame, out_dir: Path, cost: pd.DataFrame | None = None) -> list[Path]:
    out_dir = Path(out_dir)
    (out_dir / "confusion").mkdir(parents=True, exist_ok=True)
    cost = cost if cost is not None else pd.DataFrame()
    paths = []
    headline = headline_rates(df)
    band = model_band_rates(df)
    for name, table in [("headline", headline), ("model_band", band), ("stage", stage_attribution(df)),
                        ("calibration", calibration(df)), ("robustness", _table(robustness, df)),
                        ("injection", _table(injection_table, df)), ("pairs", _table(pair_gap, df)), ("cost", cost)]:
        p = out_dir / f"{name}.csv"
        table.to_csv(p)
        paths.append(p)
    if len(headline) and len(cost):
        sc = safety_cost_table(headline, cost)
        sc.to_csv(out_dir / "safety_cost.csv")
        paths.append(out_dir / "safety_cost.csv")
    for (backend, model), cm in confusion(df).items():
        p = out_dir / "confusion" / f"{backend}__{model.replace('/', '_').replace(':', '-')}.csv"
        cm.to_csv(p)
        paths.append(p)
    models = set(map(tuple, df[["backend", "model"]].drop_duplicates().to_numpy()))
    versions = []
    if ("typesafe", "jev-1.13.0") in models and ("typesafe", "jev-1.14.0") in models:
        versions.append(version_compare(df, ("typesafe", "jev-1.13.0"), ("typesafe", "jev-1.14.0")))
    if versions:
        vp = out_dir / "versions.csv"
        pd.DataFrame(versions).to_csv(vp, index=False)
        paths.append(vp)
    md = out_dir / "summary.md"
    lines = ["# 행동 안전 게이트 요약", "", "원본: results/raw/gate/*.jsonl · 정책: config/policy.yaml", "",
             "## 핵심 지표 (base 조건, 전체 명령)", "", _to_markdown(headline.round(3)), "",
             "`rules_only`는 코드 규칙만 쓰는 베이스라인(모델 호출 없음)입니다. "
             "위험 통과율(위험 명령을 allow로 낸 비율)이 낮을수록, 안전 마찰률(안전 명령을 ask/deny로 낸 비율)이 낮을수록 좋습니다.", "",
             "## 모델 구간만 (규칙이 못 가린 애매한 명령)", "", _to_markdown(band.round(3)), "",
             "규칙이 결정적으로 처리한 명령을 빼고, 결정 모델이 실제로 판단한 구간만 본 수치입니다. "
             "결정 모델이 '애매하면 전부 사람에게'(위험 통과 0, 마찰 100%)보다 무엇을 사주는지를 분리해 보여줍니다. "
             "모든 비율에 n과 95% 신뢰구간이 CSV에 있습니다."]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths.append(md)
    return paths


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results" / "raw" / "gate"
    df = collect(results_dir, root / "data", load_policy())
    if df.empty:
        print("no results in results/raw/gate")
        return 1
    cost = cost_summary(results_dir)
    paths = write_summaries(df, root / "results" / "summary" / "gate", cost)
    print(f"wrote {len(paths)} files to results/summary/gate")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
