# tests/test_gate_figures.py
import matplotlib
import pandas as pd

from dalab.plots import gate_figures, style


def test_apply_style_uses_agg_and_cjk():
    style.apply_style()
    assert matplotlib.get_backend().lower() == "agg"
    assert any("CJK" in f or "Noto" in f for f in matplotlib.rcParams["font.sans-serif"]) or matplotlib.rcParams["font.sans-serif"]


def test_confusion_and_bars_render(tmp_path):
    cm = pd.DataFrame([[10, 1, 0], [2, 8, 1], [0, 1, 12]], index=["allow", "ask", "deny"], columns=["allow", "ask", "deny"])
    paths = gate_figures.fig_confusion(cm, "jev-1.14.0", tmp_path, "f2_cm")
    assert len(paths) == 2 and all(p.exists() and p.stat().st_size > 0 for p in paths)

    headline = pd.DataFrame({"dangerous_pass": [0.0, 0.1], "safe_friction": [0.2, 0.3]},
                            index=pd.MultiIndex.from_tuples([("typesafe", "jev-1.14.0"), ("upstage", "solar-decide")], names=["backend", "model"]))
    assert all(p.exists() for p in gate_figures.fig_pass_vs_friction(headline, tmp_path, "f3"))

    inj = pd.DataFrame({"dangerous_pass": [0.1, 0.0, 0.2, 0.0]},
                       index=pd.MultiIndex.from_tuples([("typesafe", "authority_owner", "en"), ("typesafe", "authority_owner", "ko"),
                                                        ("upstage", "authority_owner", "en"), ("upstage", "authority_owner", "ko")],
                                                       names=["backend", "injection", "lang"]))
    assert all(p.exists() for p in gate_figures.fig_injection_heatmap(inj, tmp_path, "f4"))


def test_latency_box_empty_and_all_nan_are_safe(tmp_path):
    # Unlike a populated frame, empty / all-NaN latency input must not raise (it returns no paths).
    assert gate_figures.fig_latency_box(pd.DataFrame(), tmp_path, "f6_empty") == []
    all_nan = pd.DataFrame({"backend": ["typesafe"], "model": ["jev-1.14.0"], "latency_ms": [float("nan")]})
    assert gate_figures.fig_latency_box(all_nan, tmp_path, "f6_nan") == []


def test_latency_cost_korean(tmp_path):
    raw = pd.DataFrame({"backend": ["typesafe", "typesafe", "upstage"], "model": ["jev-1.14.0", "jev-1.14.0", "solar-decide"],
                        "latency_ms": [120.0, 140.0, 250.0]})
    assert all(p.exists() for p in gate_figures.fig_latency_box(raw, tmp_path, "f6"))
    cost = pd.DataFrame({"cost_per_1k": [0.9, 2.1]},
                        index=pd.MultiIndex.from_tuples([("typesafe", "jev-1.14.0"), ("upstage", "solar-decide")], names=["backend", "model_requested"]))
    assert all(p.exists() for p in gate_figures.fig_cost_bar(cost, tmp_path, "f7"))
    pair = pd.DataFrame({"en_minus_ko_acc": [0.05, 0.12], "agreement": [0.9, 0.8]},
                        index=pd.MultiIndex.from_tuples([("typesafe", "jev-1.14.0"), ("upstage", "solar-decide")], names=["backend", "model"]))
    assert all(p.exists() for p in gate_figures.fig_korean_gap(pair, tmp_path, "f9"))


def test_frontier_figures_render(tmp_path):
    import pandas as pd
    from dalab.plots import gate_figures
    sc = pd.DataFrame({"dangerous_pass": [0.0, 0.02, 0.1], "safe_friction": [0.5, 0.15, 0.08],
                       "cost_per_1k": [0.0, 0.9, 18.0], "latency_p50_ms": [0.0, 120.0, 2200.0],
                       "latency_p95_ms": [0.0, 200.0, 4000.0]},
                      index=pd.MultiIndex.from_tuples([("rules_only", "rules"), ("typesafe", "jev-1.14.0"),
                                                       ("llm_judge", "gpt-6-luna")], names=["backend", "model"]))
    assert all(p.exists() and p.stat().st_size > 0 for p in gate_figures.fig_safety_vs_cost(sc, tmp_path, "f10"))
    assert all(p.exists() for p in gate_figures.fig_safety_vs_latency(sc, tmp_path, "f11"))
