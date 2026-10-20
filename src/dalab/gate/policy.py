# src/dalab/gate/policy.py
"""Layered gate policy: hard-deny regexes, a read-only allowlist, then the decision model's answers."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from dalab.decision import Answer
from dalab.gate.records import GateRecord
from dalab.util.env import find_repo_root

_RISK = ("destructive", "secrets", "exfiltrates", "privileged")
_SHELL_META = re.compile(r"[|;&><`$()]")  # pipe / sequencing / redirection / subshell / substitution


@dataclass(frozen=True)
class Thresholds:
    t_hi: float = 0.5
    t_lo: float = 0.2
    c_min: float = 0.6


@dataclass
class PolicyDecision:
    label: str   # allow | ask | deny
    stage: str   # hard_deny | allowlist | model | error
    reasons: list[str] = field(default_factory=list)


@dataclass
class Policy:
    hard_deny: list[tuple[str, re.Pattern]]
    allow_heads: set[str]
    git_read_only: set[str]
    thresholds: Thresholds
    tuned: dict[str, Thresholds] = field(default_factory=dict)


def load_policy(path: Path | None = None) -> Policy:
    path = path or find_repo_root() / "config" / "policy.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    # Case-sensitive by design (see the plan's Task 4 design note): shell command names are
    # case-sensitive, and matching case-insensitively only widens the over-match surface.
    hard = [(r["name"], re.compile(r["pattern"])) for r in raw.get("hard_deny", [])]
    ro = raw.get("read_only_allow", {})
    tuned = {k: Thresholds(**v) for k, v in (raw.get("tuned") or {}).items()}
    return Policy(hard_deny=hard, allow_heads=set(ro.get("heads", [])), git_read_only=set(ro.get("git_subcommands", [])),
                 thresholds=Thresholds(**raw.get("thresholds", {})), tuned=tuned)


def thresholds_for(policy: Policy, backend_key: str | None) -> Thresholds:
    return policy.tuned.get(backend_key, policy.thresholds) if backend_key else policy.thresholds


def hard_deny_match(command: str, policy: Policy) -> str | None:
    for name, pat in policy.hard_deny:
        if pat.search(command):
            return name
    return None


def allowlisted(command: str, policy: Policy) -> bool:
    cmd = command.strip()
    if "\n" in cmd or _SHELL_META.search(cmd):
        return False
    tokens = cmd.split()
    if not tokens:
        return False
    if tokens[0] == "git":
        return len(tokens) >= 2 and tokens[1] in policy.git_read_only
    return tokens[0] in policy.allow_heads


def _p(answers: dict[str, Answer], name: str, polarity: bool) -> float:
    a = answers.get(name)
    if a is None or a.noul is None:
        return 0.0
    return 1.0 - a.noul if polarity else a.noul


def decide_from_answers(answers: dict[str, Answer], *, has_request: bool, th: Thresholds, polarity: bool = False) -> PolicyDecision:
    v = {k: _p(answers, k, polarity) for k in (*_RISK, "grounded", "instruction_text")}
    ac = answers.get("action_class")
    action = ac.choice if ac else None
    confidence = ac.confidence if ac and ac.confidence is not None else 1.0

    deny = []
    if v["destructive"] >= th.t_hi:
        deny.append(f"destructive={v['destructive']:.2f}")
    if v["exfiltrates"] >= th.t_hi:
        deny.append(f"exfiltrates={v['exfiltrates']:.2f}")
    if v["secrets"] >= th.t_hi and (not has_request or v["grounded"] < 0.5):
        deny.append(f"secrets={v['secrets']:.2f}")
    if deny:
        return PolicyDecision("deny", "model", deny)

    ask = []
    if v["privileged"] >= th.t_hi:
        ask.append(f"privileged={v['privileged']:.2f}")
    if v["instruction_text"] >= th.t_hi:
        ask.append(f"instruction_text={v['instruction_text']:.2f}")
    if action in ("system", "outward", "unclear"):
        ask.append(f"action_class={action}")
    if has_request and v["grounded"] < 0.5:
        ask.append(f"grounded={v['grounded']:.2f}")
    if confidence < th.c_min:
        ask.append(f"low_confidence={confidence:.2f}")
    ask += [f"midband:{k}={v[k]:.2f}" for k in _RISK if th.t_lo <= v[k] < th.t_hi]
    if ask:
        return PolicyDecision("ask", "model", ask)
    return PolicyDecision("allow", "model", ["clear"])


def decide(record: GateRecord, answers: dict[str, Answer] | None, policy: Policy, *,
           backend_key: str | None = None, error: str | None = None, polarity: bool = False) -> PolicyDecision:
    name = hard_deny_match(record.command, policy)
    if name:
        return PolicyDecision("deny", "hard_deny", [name])
    if record.kind == "shell" and allowlisted(record.command, policy):
        return PolicyDecision("allow", "allowlist", ["read_only_allowlist"])
    if error or answers is None:
        return PolicyDecision("ask", "error", [error or "no_answers"])
    return decide_from_answers(answers, has_request=record.has_request, th=thresholds_for(policy, backend_key), polarity=polarity)
