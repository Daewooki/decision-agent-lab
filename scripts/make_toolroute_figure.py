#!/usr/bin/env python3
"""Render the tool-routing figure from results/summary/toolroute/summary.csv."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "summary" / "toolroute" / "summary.csv"
FIG = ROOT / "figures"

LABEL = {
    "typesafe:jev-1.13.0": "Jev 1.13\n(decision model)",
    "openai_decisions:gpt-6-luna": "OpenAI Decisions",
    "llm_native:gpt-6-luna": "LLM native\n(structured output)",
}
COLOR = {
    "typesafe:jev-1.13.0": "#1f77b4",
    "openai_decisions:gpt-6-luna": "#2ca02c",
    "llm_native:gpt-6-luna": "#d62728",
}


def main() -> None:
    rows = list(csv.DictReader(SUMMARY.open()))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for r in rows:
        x = float(r["p50_ms"])
        y = float(r["acc_overall"])
        over = float(r["overfire_rate"])
        c = COLOR.get(r["router"], "#555")
        ax.scatter(x, y, s=260, color=c, zorder=3, edgecolor="white", linewidth=1.5)
        note = LABEL.get(r["router"], r["router"])
        if over > 0:
            note += f"\noverfire {over*100:.0f}%"
        ax.annotate(note, (x, y), textcoords="offset points", xytext=(10, -6),
                    fontsize=9, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("p50 latency per routing call (ms, log scale)")
    ax.set_ylabel("routing accuracy (exact, n=40)")
    ax.set_ylim(0.78, 1.03)
    ax.set_title("Tool routing: accuracy vs. latency (upper-left is best)")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"f_toolroute.{ext}", dpi=140)
    print(f"wrote {FIG / 'f_toolroute.png'} (+svg)")


if __name__ == "__main__":
    main()
