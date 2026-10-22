#!/usr/bin/env python3
"""Tool-routing mini-experiment: given a user request and a fixed tool catalog,
each router picks the single tool to call next (or abstains: 'none' / 'escalate_to_human').

Routers compared:
  - decision models (Choice over tool ids + 'none'): e.g. typesafe:jev-1.13.0, openai_decisions:gpt-6-luna
  - llm_native:<model>  -- a generative LLM picks the tool via strict-JSON structured output
    (the "LLM way": a full generation call just to route -- Besthub's routing paradox).

Behind a TLS-intercepting corporate proxy, prefix with SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt.
Dry-run only: nothing is executed; we record each router's choice, latency, cost.

Usage:
  SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt uv run python scripts/bench_toolroute.py \
      --routers typesafe:jev-1.13.0,openai_decisions:gpt-6-luna,llm_native:gpt-6-luna \
      --concurrency 2
"""
from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dalab.backends.openai_sdk import make_openai_client, to_backend_error
from dalab.backends.registry import load_models_config, make_decider
from dalab.questions import Choice
from dalab.util.env import get_secret, load_dotenv
from dalab.util.http import BackendError, with_retries

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "toolroute"
RAW = ROOT / "results" / "raw" / "toolroute"
SUMMARY = ROOT / "results" / "summary" / "toolroute"
ABSTAIN = {"none", "escalate_to_human"}

DEFAULT_ROUTERS = "typesafe:jev-1.13.0,openai_decisions:gpt-6-luna,llm_native:gpt-6-luna"

CHOICE_INSTRUCTIONS = (
    "A coding agent must pick the single tool to call next for the user's request. "
    "Choose exactly one. Pick 'escalate_to_human' for a high-consequence or policy-bound "
    "decision that needs human approval. Pick 'none' only when no listed tool fits -- the "
    "agent should answer directly or the request is out of scope."
)


def load_catalog() -> list[dict]:
    return json.loads((DATA / "catalog.json").read_text())["tools"]


def load_requests(limit: int | None, lang: str | None) -> list[dict]:
    rows = [json.loads(l) for l in (DATA / "requests.jsonl").read_text().splitlines() if l.strip()]
    if lang:
        rows = [r for r in rows if r["lang"] == lang]
    return rows[:limit] if limit else rows


def build_question(catalog: list[dict]) -> Choice:
    criteria = {t["id"]: t["desc"] for t in catalog}
    criteria["none"] = "No listed tool fits; the agent answers directly or the request is out of scope."
    return Choice(CHOICE_INSTRUCTIONS, criteria)


def state_for(request: str) -> str:
    return f"A coding agent received this user request:\n\n{request!r}\n\nDecide which single tool to call next."


# ---- LLM-native router (a generative model routes via strict JSON) ----

LLM_NATIVE_SYSTEM = (
    "You are the routing layer of a coding agent. Given the user's request and the list of available "
    "tools, output JSON choosing the single tool to call next and your confidence (0-1). "
    "Use 'escalate_to_human' for high-consequence or policy-bound decisions. Use 'none' only when no "
    "listed tool fits. Respond only with JSON matching the schema."
)


def _llm_native_prompt(request: str, catalog: list[dict]) -> str:
    lines = [f"USER REQUEST:\n{request}", "", "AVAILABLE TOOLS:"]
    lines += [f"- {t['id']}: {t['desc']}" for t in catalog]
    lines.append("- none: no listed tool fits; answer directly or out of scope")
    return "\n".join(lines)


def _llm_native_schema(option_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "tool": {"type": "string", "enum": option_ids},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["tool", "confidence"],
        "additionalProperties": False,
    }


def route_llm_native(client, model, request, catalog, option_ids, in_price, out_price):
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": LLM_NATIVE_SYSTEM},
            {"role": "user", "content": _llm_native_prompt(request, catalog)},
        ],
        "text": {"format": {"type": "json_schema", "name": "route", "schema": _llm_native_schema(option_ids), "strict": True}},
    }

    def attempt():
        try:
            return client.responses.create(**payload)
        except BackendError:
            raise
        except Exception as e:  # noqa: BLE001
            raise to_backend_error(e) from e

    resp, ms, _ = with_retries(attempt, retries=3, backoff_s=0.5)
    obj = json.loads(getattr(resp, "output_text", "") or "{}")
    usage = getattr(resp, "usage", None)
    in_tok = getattr(usage, "input_tokens", None)
    out_tok = getattr(usage, "output_tokens", None)
    cost = None
    if in_tok is not None:
        cost = in_tok / 1e6 * in_price + (out_tok or 0) / 1e6 * out_price
    return obj.get("tool"), obj.get("confidence"), ms, cost, in_tok


def run_router(router: str, requests: list[dict], catalog: list[dict], question: Choice, concurrency: int) -> list[dict]:
    option_ids = list(question.criteria)
    if router.startswith("llm_native:"):
        model = router.split(":", 1)[1]
        cfg = load_models_config()["backends"]["llm_judge"]
        prices = cfg.get("prices", {}).get(model, {})
        in_price, out_price = prices.get("input", 0.10), prices.get("output", 0.50)
        client = make_openai_client(get_secret("OPENAI_API_KEY"), 30.0)

        def one(r):
            try:
                tool, conf, ms, cost, in_tok = route_llm_native(
                    client, model, r["request"], catalog, option_ids, in_price, out_price)
                return _row(router, r, tool, conf, ms, cost, in_tok, None)
            except Exception as e:  # noqa: BLE001
                return _row(router, r, None, None, None, None, None, str(e)[:200])
    else:
        decider = make_decider(router)

        def one(r):
            d = decider.decide(state_for(r["request"]), {"tool": question})
            ans = d.answers.get("tool") if not d.error else None
            pred = ans.choice if ans else None
            conf = ans.confidence if ans else None
            return _row(router, r, pred, conf, d.latency_ms, d.cost_usd, d.usage_input_tokens, d.error)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        return list(ex.map(one, requests))


def _row(router, r, pred, conf, ms, cost, in_tok, error) -> dict:
    return {
        "router": router, "id": r["id"], "lang": r["lang"], "gold": r["gold"],
        "pred": pred, "correct": (pred == r["gold"]) if pred is not None else False,
        "confidence": conf, "latency_ms": ms, "cost_usd": cost, "input_tokens": in_tok, "error": error,
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    ok = [r for r in rows if not r["error"]]
    acc = sum(r["correct"] for r in rows) / n if n else 0.0
    tool_rows = [r for r in rows if r["gold"] not in ABSTAIN]
    abst_rows = [r for r in rows if r["gold"] in ABSTAIN]
    acc_tool = sum(r["correct"] for r in tool_rows) / len(tool_rows) if tool_rows else 0.0
    abstain_recall = sum(r["correct"] for r in abst_rows) / len(abst_rows) if abst_rows else 0.0
    overfire = sum(1 for r in abst_rows if r["pred"] not in ABSTAIN) / len(abst_rows) if abst_rows else 0.0
    lat = sorted(r["latency_ms"] for r in ok if r["latency_ms"] is not None)
    costs = [r["cost_usd"] for r in ok if r["cost_usd"] is not None]
    p50 = statistics.median(lat) if lat else None
    cost_per_1k = (sum(costs) / len(costs) * 1000) if costs else 0.0
    return {
        "n": n, "errors": n - len(ok), "acc_overall": acc, "acc_tool": acc_tool,
        "abstain_recall": abstain_recall, "overfire_rate": overfire,
        "p50_ms": p50, "cost_per_1k": cost_per_1k,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routers", default=DEFAULT_ROUTERS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--lang", choices=["en", "ko"], default=None)
    ap.add_argument("--concurrency", type=int, default=2)
    args = ap.parse_args()

    load_dotenv()
    RAW.mkdir(parents=True, exist_ok=True)
    SUMMARY.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog()
    question = build_question(catalog)
    requests = load_requests(args.limit, args.lang)
    routers = [s.strip() for s in args.routers.split(",") if s.strip()]

    summary_rows = []
    for router in routers:
        rows = run_router(router, requests, catalog, question, args.concurrency)
        safe = router.replace("/", "_").replace(":", "-")
        (RAW / f"{safe}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
        s = summarize(rows)
        s["router"] = router
        summary_rows.append(s)
        print(f"{router:34s} acc={s['acc_overall']:.3f} tool={s['acc_tool']:.3f} "
              f"abstain={s['abstain_recall']:.3f} overfire={s['overfire_rate']:.3f} "
              f"p50={s['p50_ms']}ms cost/1k=${s['cost_per_1k']:.3f} err={s['errors']}")

    cols = ["router", "n", "errors", "acc_overall", "acc_tool", "abstain_recall", "overfire_rate", "p50_ms", "cost_per_1k"]
    out = [",".join(cols)]
    for s in summary_rows:
        out.append(",".join("" if s.get(c) is None else str(s.get(c)) for c in cols))
    (SUMMARY / "summary.csv").write_text("\n".join(out) + "\n")
    print(f"\nwrote {SUMMARY / 'summary.csv'}")


if __name__ == "__main__":
    main()
