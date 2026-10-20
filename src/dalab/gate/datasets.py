# src/dalab/gate/datasets.py
"""Loaders that map every source's native labels to allow / ask / deny."""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from dalab.gate.records import GOLD, GateRecord

THEMSQUARED_MAP = {"readonly": "allow", "privileged": "ask", "destructive": "deny", "exfiltration": "deny"}
GODSPEDE_MAP = {"allow": "allow", "ask": "ask", "escalate": "ask", "deny": "deny"}
LAYA_MAP = {"allow": "allow", "review": "ask", "block": "deny"}  # "any" rows are skipped
AITEJIU_MAP = {"allow": "allow", "review": "ask", "deny": "deny"}

_HANGUL = re.compile(r"[가-힣ㄱ-ㆎ]")
_SHELL_HEAD = re.compile(r"^[a-z][a-z0-9_.+-]*(\s|$)")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _check(gold: str) -> str:
    if gold not in GOLD:
        raise ValueError(f"bad gold label {gold!r}")
    return gold


def load_themsquared(path: Path) -> list[GateRecord]:
    out = []
    for r in _jsonl(path):
        gold = THEMSQUARED_MAP[r["label"]]
        alt = sorted({THEMSQUARED_MAP[a] for a in (r.get("alt") or [])} - {gold})
        out.append(GateRecord(
            id=f"ts-{r['id']}", source="themsquared", kind="shell", lang="en",
            command=r["state"].removeprefix("Agent tool call: ").strip(), gold=_check(gold), gold_native=r["label"],
            difficulty=r["difficulty"], alt_gold=alt, meta={"source_tag": r.get("source"), "state_raw": r["state"]}))
    return out


def load_godspede(dir_: Path) -> list[GateRecord]:
    dir_ = Path(dir_)
    world = json.loads((dir_ / "world.json").read_text(encoding="utf-8")) if (dir_ / "world.json").exists() else {"repos": []}
    default_cwd = (world.get("repos") or [{"root": "/home/dev/src/app"}])[0]["root"]
    out = []
    for fname, prefix in (("battery.jsonl", ""), ("holdout.jsonl", ""), ("real-cases.jsonl", "")):
        for r in _jsonl(dir_ / fname):
            if "tool" in r:  # file-tool calls (read/edit/write) are not shell commands
                continue
            remotes = r.get("remote_urls")
            context = "git remotes: " + ", ".join(f"{k}={v}" for k, v in remotes.items()) if remotes else None
            out.append(GateRecord(
                id=f"gs-{prefix}{r['id']}", source="godspede", kind="shell", lang="en", command=r["command"],
                gold=_check(GODSPEDE_MAP[r["expect"]]), gold_native=r["expect"], cwd=r.get("cwd") or default_cwd,
                context=context, category=r.get("category"),
                meta={"file": fname, "why": r.get("why"), "injection_attempt": r.get("expect_injection_attempt")}))
    return out


def _laya_kind_lang(text: str) -> tuple[str, str]:
    lang = "ko" if _HANGUL.search(text) else "en"
    kind = "shell" if _SHELL_HEAD.match(text) and not _HANGUL.match(text) else "request"
    return kind, lang


def load_laya_guard(dir_: Path) -> list[GateRecord]:
    dir_ = Path(dir_)
    out = []
    for r in _jsonl(dir_ / "testset_shell.jsonl"):
        out.append(GateRecord(id=f"lg-shell-{r['id']}", source="laya_guard", kind="shell", lang="en", command=r["tool_call"],
                              gold=_check(LAYA_MAP[r["expected"]]), gold_native=r["expected"], meta={"file": "testset_shell"}))
    for r in _jsonl(dir_ / "testset_ko_30.jsonl"):
        kind, lang = _laya_kind_lang(r["tool_call"])
        out.append(GateRecord(id=f"lg-ko-{r['id']}", source="laya_guard", kind=kind, lang=lang, command=r["tool_call"],
                              gold=_check(LAYA_MAP[r["expected"]]), gold_native=r["expected"], category=r.get("policy"),
                              meta={"file": "testset_ko_30"}))
    for r in _jsonl(dir_ / "testset_incidents.jsonl"):
        if r["expected"] not in LAYA_MAP:
            continue
        out.append(GateRecord(id=f"lg-inc-{r['id']}", source="laya_guard", kind="shell", lang="en", command=r["tool_call"],
                              gold=_check(LAYA_MAP[r["expected"]]), gold_native=r["expected"],
                              meta={"incident": r["incident"], "fidelity": r["fidelity"]}))
    return out


def load_aitejiu(path: Path) -> list[GateRecord]:
    return [GateRecord(id=f"at-{r['id']}", source="aitejiu", kind="shell", lang="en", command=r["command"],
                       gold=_check(AITEJIU_MAP[r["expected"]]), gold_native=r["expected"], context=r.get("context"),
                       category=",".join(r.get("tags") or []) or None) for r in _jsonl(path)]


def load_own(path: Path) -> list[GateRecord]:
    return [GateRecord(id=r["id"], source="own", kind="shell", lang=r["lang"], command=r["command"], gold=_check(r["gold"]),
                       gold_native=r["gold"], user_request=r["user_request"], cwd=r.get("cwd"), context=r.get("context"),
                       pair_id=r["pair_id"], category=r["category"],
                       meta={"rationale": r.get("rationale"), "context_dependent": bool(r.get("context_dependent")),
                             "injection_seed": bool(r.get("injection_seed"))}) for r in _jsonl(path)]


def load_all(data_root: Path, include=("themsquared", "godspede", "laya_guard", "aitejiu", "own")) -> list[GateRecord]:
    """Load every requested source that is present on disk.

    Sources whose backing file is absent are skipped rather than raising: on a clean checkout the
    gitignored aitejiu cache and the later-authored data/own set may be missing, and the keyless CI
    re-aggregation (analyze_gate) must still run over whatever is present instead of crashing.
    """
    data_root = Path(data_root)
    out: list[GateRecord] = []
    ts = data_root / "public" / "themsquared" / "tasks.jsonl"
    if "themsquared" in include and ts.exists():
        out += load_themsquared(ts)
    gs = data_root / "public" / "godspede"
    if "godspede" in include and (gs / "battery.jsonl").exists():
        out += load_godspede(gs)
    lg = data_root / "public" / "laya_guard"
    if "laya_guard" in include and (lg / "testset_shell.jsonl").exists():
        out += load_laya_guard(lg)
    at = data_root / "cache" / "aitejiu" / "bash_commands.jsonl"
    if "aitejiu" in include and at.exists():
        out += load_aitejiu(at)
    ow = data_root / "own" / "gate_ko_en.jsonl"
    if "own" in include and ow.exists():
        out += load_own(ow)
    ids = [r.id for r in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate record ids across sources")
    return out


def records_by_id(records: list[GateRecord]) -> dict[str, GateRecord]:
    return {r.id: r for r in records}


def themsquared_split(records: list[GateRecord], seed: int = 0) -> tuple[list[GateRecord], list[GateRecord]]:
    """Stratified 50/50 split by (native label, difficulty) for threshold tuning vs evaluation."""
    groups: dict[tuple, list[GateRecord]] = defaultdict(list)
    for r in sorted(records, key=lambda x: x.id):
        groups[(r.gold_native, r.difficulty)].append(r)
    rng = random.Random(seed)
    tune, ev = [], []
    extra_to_tune = False  # alternate the odd-group remainder so the two halves stay balanced
    for key in sorted(groups):
        g = list(groups[key])
        rng.shuffle(g)
        half = len(g) // 2
        if len(g) % 2:
            if extra_to_tune:
                half += 1
            extra_to_tune = not extra_to_tune
        tune += g[:half]
        ev += g[half:]
    return tune, ev
