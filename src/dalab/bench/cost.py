"""Per-backend cost and latency summary from raw rows."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

INDEX = ["backend", "model_requested"]
COLUMNS = ["n_calls", "n_errors", "n_unpriced", "input_tokens", "output_tokens", "cost_usd", "cost_per_1k",
           "latency_p50_ms", "latency_p95_ms", "first_call_ms"]


def summarize_cost(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """One line per (backend, model). Warmup rows feed first_call_ms only; errors are counted, not averaged.

    cost_per_1k divides the summed cost by the number of *priced* rows (cost_usd not null). Successful rows
    without usage are counted in n_unpriced and excluded, as spec section 4 asks; footnote them when reporting.
    """
    if not rows:
        return pd.DataFrame(columns=[*INDEX, *COLUMNS]).set_index(INDEX)
    recs = []
    for r in rows:
        d = r["decision"]
        recs.append({"backend": d["backend"], "model": d["model_requested"], "condition": r["condition"],
                     "latency_ms": d["latency_ms"], "cost_usd": d["cost_usd"], "input_tokens": d["usage_input_tokens"],
                     "output_tokens": d["usage_output_tokens"], "error": d["error"] is not None, "ts": d["ts_utc"]})
    df = pd.DataFrame(recs)
    out = []
    for (backend, model), g in df.groupby(["backend", "model"], sort=True):
        warm = g[g.condition == "warmup"].sort_values("ts")
        main = g[g.condition != "warmup"]
        ok = main[~main.error & main.latency_ms.notna()]
        priced = main.cost_usd.notna()
        n_priced = int(priced.sum())
        cost = float(main.cost_usd[priced].sum())
        out.append({"backend": backend, "model_requested": model,
                    "n_calls": int(len(main)), "n_errors": int(main.error.sum()),
                    "n_unpriced": int((~main.error & ~priced).sum()),
                    "input_tokens": int(main.input_tokens.fillna(0).sum()), "output_tokens": int(main.output_tokens.fillna(0).sum()),
                    "cost_usd": cost, "cost_per_1k": (cost / n_priced * 1000.0) if n_priced else np.nan,
                    "latency_p50_ms": float(np.percentile(ok.latency_ms, 50)) if len(ok) else np.nan,
                    "latency_p95_ms": float(np.percentile(ok.latency_ms, 95)) if len(ok) else np.nan,
                    "first_call_ms": float(warm.latency_ms.iloc[0]) if len(warm) else np.nan})
    return pd.DataFrame(out, columns=[*INDEX, *COLUMNS]).set_index(INDEX)
