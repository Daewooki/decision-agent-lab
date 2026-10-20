# tests/test_analyze_gate.py
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analyze_gate", ROOT / "scripts" / "analyze_gate.py")
analyze_gate = importlib.util.module_from_spec(spec)
sys.modules["analyze_gate"] = analyze_gate
spec.loader.exec_module(analyze_gate)

from dalab.gate.policy import load_policy
from dalab.gate.records import GateRecord


def _row(item_id, backend, model, destructive, action="read_only"):
    answers = {k: {"type": "noul", "noul": v, "choice": None, "probabilities": None, "confidence": None, "confidence_source": None, "score": None}
               for k, v in {"destructive": destructive, "secrets": 0.0, "exfiltrates": 0.0, "privileged": 0.0, "grounded": 1.0, "instruction_text": 0.0}.items()}
    answers["action_class"] = {"type": "choice", "choice": action, "probabilities": {action: 1.0}, "confidence": 1.0, "confidence_source": "reported", "noul": None, "score": None}
    dec = {"backend": backend, "model_requested": model, "model_reported": model, "ts_utc": "t", "location": "Seoul, KR",
           "latency_ms": 100.0, "server_latency_ms": None, "answers": answers, "usage_input_tokens": 100, "usage_output_tokens": 0,
           "cost_usd": 1e-5, "option_order": {}, "error": None, "raw": {}, "extra": {}}
    return {"key": item_id, "run_id": "r", "item_id": item_id, "condition": "base", "repeat": 0, "order_seed": None, "meta": {}, "decision": dec}


def test_collect_and_write(tmp_path, monkeypatch):
    results = tmp_path / "raw" / "gate"
    results.mkdir(parents=True)
    (results / "fake__m1.jsonl").write_text("\n".join(json.dumps(r) for r in [
        _row("ts-t001", "fake", "m1", 0.0), _row("ts-c135", "fake", "m1", 0.9, action="local_write")]) + "\n")

    recs = [GateRecord("ts-t001", "themsquared", "shell", "en", "echo ok", "allow", "readonly"),
            GateRecord("ts-c135", "themsquared", "shell", "en", "echo bad", "deny", "destructive")]
    monkeypatch.setattr(analyze_gate, "load_all", lambda data_root, include=None: recs)
    df = analyze_gate.collect(results, tmp_path / "data", load_policy())
    assert len(df) == 2 and set(df.pred) <= {"allow", "ask", "deny"}
    out = analyze_gate.write_summaries(df, tmp_path / "summary")
    names = {p.name for p in out}
    assert "headline.csv" in names and "summary.md" in names
    assert (tmp_path / "summary" / "headline.csv").exists()
