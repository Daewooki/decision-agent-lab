#!/usr/bin/env python
"""Render experiment figures from the summary tables under results/summary.

Minimal phase-2 dispatcher: `--gate` draws the gate figures that the Task 9 summary
tables already provide (F2 confusion, F3 pass-vs-friction, F4 injection, F9 Korean gap).
Figures whose inputs are not emitted by scripts/analyze_gate.py yet (F5 calibration points,
F6 latency, F7 cost) are rendered only when their source file is present, so this stays a
thin dispatcher over dalab.plots.gate_figures and is extended in phase 3.

Usage: uv run python scripts/make_figures.py --gate
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from dalab.plots import gate_figures


def _read_indexed(path: Path, index_col: list[int]) -> pd.DataFrame:
    """Read a summary CSV with a multi-index, tolerating the empty-table placeholder.

    analyze_gate writes an empty DataFrame (a lone `""` line) for tables a partial run
    can't fill; pandas raises IndexError on that with index_col set, so treat it as no data.
    """
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, index_col=index_col)
    except (IndexError, ValueError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def make_gate_figures(summary_dir: Path, out_dir: Path) -> list[Path]:
    summary_dir, out_dir = Path(summary_dir), Path(out_dir)
    written: list[Path] = []

    headline = _read_indexed(summary_dir / "headline.csv", [0, 1])
    if len(headline):
        written += gate_figures.fig_pass_vs_friction(headline, out_dir, "f3_pass_vs_friction")

    injection = _read_indexed(summary_dir / "injection.csv", [0, 1, 2])
    if len(injection):
        written += gate_figures.fig_injection_heatmap(injection, out_dir, "f4_injection")

    pairs = _read_indexed(summary_dir / "pairs.csv", [0, 1])
    if len(pairs):
        written += gate_figures.fig_korean_gap(pairs, out_dir, "f9_korean_gap")

    conf_dir = summary_dir / "confusion"
    if conf_dir.is_dir():
        for p in sorted(conf_dir.glob("*.csv")):
            cm = pd.read_csv(p, index_col=0)
            if len(cm):
                written += gate_figures.fig_confusion(cm, p.stem.replace("__", " · "), out_dir, f"f2_cm_{p.stem}")

    # Trade-off frontiers (the headline insight figures) + cost bar, from the joined safety/cost table.
    sc = _read_indexed(summary_dir / "safety_cost.csv", [0, 1])
    if len(sc):
        written += gate_figures.fig_safety_vs_cost(sc, out_dir, "f10_safety_vs_cost")
        written += gate_figures.fig_safety_vs_latency(sc, out_dir, "f11_safety_vs_latency")

    cost = _read_indexed(summary_dir / "cost.csv", [0, 1])
    if len(cost) and "cost_per_1k" in cost.columns:
        written += gate_figures.fig_cost_bar(cost, out_dir, "f7_cost")

    # Optional: calibration reliability points if a future analyze step emits them.
    cal_points = summary_dir / "calibration_points.csv"
    if cal_points.exists():
        df_points = pd.read_csv(cal_points)
        if len(df_points):
            written += gate_figures.fig_calibration(df_points, out_dir, "f5_calibration")

    return written


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Render decision-agent-lab figures from results/summary.")
    parser.add_argument("--gate", action="store_true",
                        help="render the gate experiment figures from results/summary/gate into figures/")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if not args.gate:
        parser.error("nothing to do: pass --gate")
    written = make_gate_figures(root / "results" / "summary" / "gate", root / "figures")
    print(f"wrote {len(written)} figure files to figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
