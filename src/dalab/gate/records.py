# src/dalab/gate/records.py
"""One normalized record per gate item, whatever the source."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

GOLD = ("allow", "ask", "deny")


@dataclass
class GateRecord:
    id: str
    source: str          # themsquared | godspede | laya_guard | aitejiu | own
    kind: str            # shell (a command) | request (natural-language action)
    lang: str            # en | ko
    command: str
    gold: str            # allow | ask | deny
    gold_native: str     # the source's own label
    user_request: str | None = None
    cwd: str | None = None
    context: str | None = None
    pair_id: str | None = None
    difficulty: str | None = None
    category: str | None = None
    alt_gold: list[str] = field(default_factory=list)   # other labels the source also accepts
    injection: str = "none"      # none | crude | authority_owner | authority_lead | counterfeit_policy
    obfuscation: str = "none"    # none | base64 | quote_split | ifs | cmd_subst | zero_width
    base_id: str | None = None   # id of the item a variant was generated from
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def has_request(self) -> bool:
        return bool(self.user_request)
