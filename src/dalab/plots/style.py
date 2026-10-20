# src/dalab/plots/style.py
"""Matplotlib paper style with Korean (Noto Sans CJK KR) support. Import before plotting."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; set before pyplot is imported anywhere
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

_CJK_CANDIDATES = ["Noto Sans CJK KR", "NotoSansCJKkr", "Noto Sans KR", "NanumGothic", "Malgun Gothic"]


def _available_cjk() -> list[str]:
    installed = {f.name for f in fm.fontManager.ttflist}
    return [c for c in _CJK_CANDIDATES if c in installed]


def apply_style() -> None:
    cjk = _available_cjk()
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 130, "figure.autolayout": True,
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.3, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "font.sans-serif": [*cjk, "DejaVu Sans"], "axes.unicode_minus": False,
    })


def caption(ax, text: str) -> None:
    ax.figure.text(0.01, 0.005, text, fontsize=7, color="#555", ha="left", va="bottom", wrap=True)


def save(fig, out_dir, name: str) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "svg"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths
