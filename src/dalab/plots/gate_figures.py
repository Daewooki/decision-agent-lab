# src/dalab/plots/gate_figures.py
"""Paper-style figures for the gate experiment. Every function saves PNG+SVG and returns the paths."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from dalab.plots import style

style.apply_style()
import matplotlib.pyplot as plt  # noqa: E402  (after apply_style)

SRC = "출처: decision-agent-lab 직접 측정 (서울, 2026-10). 비율에는 95% 부트스트랩 신뢰구간."


def fig_confusion(cm, title: str, out_dir, name: str) -> list[Path]:
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    data = cm.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(len(cm.columns)), cm.columns)
    ax.set_yticks(range(len(cm.index)), cm.index)
    ax.set_xlabel("판정(pred)")
    ax.set_ylabel("정답(gold)")
    ax.set_title(f"행동 게이트 혼동행렬 · {title}")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, int(data[i, j]), ha="center", va="center",
                    color="white" if data[i, j] > data.max() / 2 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    style.caption(ax, SRC)
    return style.save(fig, out_dir, name)


def fig_pass_vs_friction(headline_df, out_dir, name: str) -> list[Path]:
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    labels = [f"{b}\n{m}" for b, m in headline_df.index]
    ax.scatter(headline_df["safe_friction"] * 100, headline_df["dangerous_pass"] * 100, s=60)
    for (x, y, lab) in zip(headline_df["safe_friction"] * 100, headline_df["dangerous_pass"] * 100, labels):
        ax.annotate(lab, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("안전 명령 마찰률 (%) — 낮을수록 좋음")
    ax.set_ylabel("위험 명령 통과율 (%) — 낮을수록 좋음")
    ax.set_title("안전성 대 편의성")
    style.caption(ax, SRC)
    return style.save(fig, out_dir, name)


def fig_injection_heatmap(injection_df, out_dir, name: str) -> list[Path]:
    piv = injection_df["dangerous_pass"].unstack("backend") * 100
    fig, ax = plt.subplots(figsize=(1.6 + 0.9 * len(piv.columns), 0.6 + 0.5 * len(piv.index)))
    data = piv.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="Reds", vmin=0)
    ax.set_xticks(range(len(piv.columns)), piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(piv.index)), [f"{a}·{b}" for a, b in piv.index])
    ax.set_title("주입 조건별 위험 통과율 (%)")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    style.caption(ax, SRC)
    return style.save(fig, out_dir, name)


def fig_calibration(df_points, out_dir, name: str) -> list[Path]:
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot([0, 1], [0, 1], "--", color="#999", linewidth=1)
    for label, grp in df_points.groupby("label"):
        ax.plot(grp["conf"], grp["acc"], marker="o", label=label)
    ax.set_xlabel("예측 확률")
    ax.set_ylabel("실제 비율")
    ax.set_title("신뢰도 보정 (destructive)")
    ax.legend()
    style.caption(ax, SRC)
    return style.save(fig, out_dir, name)


def fig_latency_box(raw_df, out_dir, name: str) -> list[Path]:
    groups, labels = [], []
    if "latency_ms" in raw_df.columns:
        for (b, m), g in raw_df.dropna(subset=["latency_ms"]).groupby(["backend", "model"], sort=True):
            groups.append(g["latency_ms"].to_numpy())
            labels.append(f"{b}\n{m}")
    if not groups:  # empty or all-NaN latency input: nothing to draw (matches the other figures)
        return []
    fig, ax = plt.subplots(figsize=(1.4 + 0.9 * len(groups), 4.0))
    ax.boxplot(groups, tick_labels=labels, showfliers=False)
    ax.set_ylabel("지연 (ms, 서울)")
    ax.set_title("백엔드별 지연 분포")
    style.caption(ax, SRC + " 첫 호출(연결 수립 포함)은 제외.")
    return style.save(fig, out_dir, name)


def fig_cost_bar(cost_df, out_dir, name: str) -> list[Path]:
    labels = [f"{b}\n{m}" for b, m in cost_df.index]
    fig, ax = plt.subplots(figsize=(1.4 + 0.9 * len(labels), 4.0))
    vals = cost_df["cost_per_1k"].to_numpy(dtype=float)
    ax.bar(range(len(vals)), np.where(vals <= 0, 1e-6, vals))
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("1,000 결정당 비용 (USD, 로그)")
    ax.set_title("비용")
    style.caption(ax, "단가표: config/models.toml (2026-10-20). 로컬 모델은 0.")
    return style.save(fig, out_dir, name)


def fig_korean_gap(pair_df, out_dir, name: str) -> list[Path]:
    labels = [f"{b}\n{m}" for b, m in pair_df.index]
    fig, ax = plt.subplots(figsize=(1.4 + 0.9 * len(labels), 4.0))
    ax.bar(range(len(labels)), pair_df["en_minus_ko_acc"] * 100)
    ax.axhline(0, color="#333", linewidth=0.8)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("영어 − 한국어 정확도 (%p)")
    ax.set_title("한국어 격차 (자체 ko/en 쌍)")
    style.caption(ax, SRC)
    return style.save(fig, out_dir, name)
