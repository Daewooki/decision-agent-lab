# tests/test_cost.py
import math
from dalab.bench.cost import COLUMNS, summarize_cost

def row(backend, model, cond, lat, cost, in_tok, err=None):
    return {"key": "k", "run_id": "r", "item_id": "i", "condition": cond, "repeat": 0, "order_seed": None, "meta": {},
            "decision": {"backend": backend, "model_requested": model, "model_reported": model, "ts_utc": "t", "location": "Seoul, KR",
                         "latency_ms": lat, "server_latency_ms": None, "answers": {}, "usage_input_tokens": in_tok,
                         "usage_output_tokens": 0, "cost_usd": cost, "option_order": {}, "error": err, "raw": {}, "extra": {}}}

def test_summary_per_backend():
    rows = [row("typesafe", "jev-1.14.0", "warmup", 900.0, 0.00001, 100),
            row("typesafe", "jev-1.14.0", "warmup", 300.0, 0.00001, 100)]
    rows += [row("typesafe", "jev-1.14.0", "base", float(100 + i), 0.00002, 400) for i in range(20)]
    rows += [row("typesafe", "jev-1.14.0", "base", None, None, None, err="HTTP 529: overloaded")]
    rows += [row("upstage", "solar-decide", "base", 250.0, 0.0001, 800)]
    df = summarize_cost(rows)
    ts = df.loc[("typesafe", "jev-1.14.0")]
    assert ts.n_calls == 21 and ts.n_errors == 1 and ts.n_unpriced == 0 and ts.input_tokens == 8000 and abs(ts.cost_usd - 0.0004) < 1e-12
    assert abs(ts.cost_per_1k - 0.0004 / 20 * 1000) < 1e-12
    assert ts.latency_p50_ms == 109.5 and ts.latency_p95_ms == 118.05 and ts.first_call_ms == 900.0
    up = df.loc[("upstage", "solar-decide")]
    assert up.n_calls == 1 and up.first_call_ms != up.first_call_ms  # NaN when no warmup rows

def test_unpriced_successful_rows_do_not_deflate_cost_per_1k():
    rows = [row("ollama", "nimble", "base", 50.0, 0.001, 100) for _ in range(10)]
    rows += [row("ollama", "nimble", "base", 50.0, None, None) for _ in range(10)]  # ok calls without usage
    s = summarize_cost(rows).loc[("ollama", "nimble")]
    assert s.n_calls == 20 and s.n_errors == 0 and s.n_unpriced == 10
    assert abs(s.cost_usd - 0.01) < 1e-12 and abs(s.cost_per_1k - 1.0) < 1e-12  # per priced call, not per ok call

def test_all_unpriced_gives_nan_not_zero():
    s = summarize_cost([row("ollama", "nimble", "base", 50.0, None, None)]).loc[("ollama", "nimble")]
    assert s.n_unpriced == 1 and math.isnan(s.cost_per_1k) and s.cost_usd == 0.0

def test_empty_rows_give_an_empty_frame():
    df = summarize_cost([])
    assert len(df) == 0 and list(df.columns) == COLUMNS and df.index.names == ["backend", "model_requested"]
