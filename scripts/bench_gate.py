#!/usr/bin/env python
"""Run the gate decision matrix for one backend over chosen conditions and sources.

Usage:
  SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt uv run python scripts/bench_gate.py \
      --spec typesafe:jev-1.14.0 --conditions base,perm1,perm2 --sources themsquared,godspede,laya_guard,own --warmup 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dalab.backends.registry import make_decider
from dalab.bench.runner import Item, RawStore, run_items
from dalab.gate.datasets import load_all
from dalab.gate.injections import expand_injection_suite, expand_obfuscation_suite
from dalab.gate.questions import format_state, questions_for
from dalab.util.env import load_dotenv

CONDITIONS = ("base", "perm1", "perm2", "repeat1", "repeat2", "polarity", "ko_instr")


def condition_params(condition: str) -> dict:
    return {"perm1": {"order_seed": 1, "repeat": 0}, "perm2": {"order_seed": 2, "repeat": 0},
            "repeat1": {"order_seed": None, "repeat": 1}, "repeat2": {"order_seed": None, "repeat": 2},
            }.get(condition, {"order_seed": None, "repeat": 0})


def build_items(records, condition: str) -> list[Item]:
    questions = questions_for(condition)
    return [Item(id=r.id, state=format_state(r), questions=questions, meta={"source": r.source, "gold": r.gold}) for r in records]


def run(spec, records, *, conditions, out_dir, data_root, concurrency, warmup, limit=None):
    decider = make_decider(spec)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{decider.name}__{decider.model_requested.replace('/', '_').replace(':', '-')}.jsonl"
    store = RawStore(out)
    run_id = f"gate-{decider.name}-{decider.model_requested}"
    recs = records[:limit] if limit else records
    inj = expand_injection_suite([r for r in recs if r.meta.get("injection_seed")])
    obf = expand_obfuscation_suite([r for r in recs if r.meta.get("injection_seed")])
    for cond in conditions:
        params = condition_params(cond)
        items = build_items(recs, cond)
        run_items(decider, items, store=store, run_id=run_id, condition=cond, concurrency=concurrency,
                  warmup=warmup if cond == "base" else 0, **params, progress=True)
    # The adversarial suites are asked with the base (EN) question set but stored under their own
    # conditions, so they never leak into the base-condition tables in analysis.py (headline_rates,
    # confusion, calibration, stage_attribution, pair_gap). injection_table keys off the `injection`
    # column directly, so it still sees these rows.
    if inj:
        run_items(decider, build_items(inj, "base"), store=store, run_id=run_id, condition="injection", concurrency=concurrency, progress=True)
    if obf:
        run_items(decider, build_items(obf, "base"), store=store, run_id=run_id, condition="obfuscation", concurrency=concurrency, progress=True)
    return out


def main(argv: list[str]) -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--conditions", default="base")
    ap.add_argument("--sources", default="themsquared,godspede,laya_guard,own")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    records = load_all(root / "data", include=tuple(a.sources.split(",")))
    out = run(a.spec, records, conditions=tuple(a.conditions.split(",")), out_dir=root / "results" / "raw" / "gate",
              data_root=root / "data", concurrency=a.concurrency, warmup=a.warmup, limit=a.limit)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
