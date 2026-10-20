"""Build a Decider from a '<backend>:<model>' spec and config/models.toml."""
from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from dalab.backends.base import BackendConfig, Decider
from dalab.backends.llm_judge import LLMJudgeDecider
from dalab.backends.openai_decisions import OpenAIDecisionsDecider
from dalab.backends.systemone import JevstationDecider, SystemOneDecider
from dalab.util.env import find_repo_root, get_secret, load_dotenv

SPEC_ALIASES = {"openai": "openai_decisions", "judge": "llm_judge"}

BACKEND_CLASSES: dict[str, type[Decider]] = {
    "typesafe": SystemOneDecider,
    "openrouter": SystemOneDecider,
    "upstage": SystemOneDecider,
    "ollama": SystemOneDecider,
    "jevstation": JevstationDecider,
    "openai_decisions": OpenAIDecisionsDecider,
    "llm_judge": LLMJudgeDecider,
}


def load_models_config(path: Path | None = None) -> dict[str, Any]:
    path = path or find_repo_root() / "config" / "models.toml"
    return tomllib.loads(path.read_text(encoding="utf-8"))


def make_decider(spec: str, config: dict[str, Any] | None = None, *, location: str | None = None) -> Decider:
    load_dotenv()
    cfg_all = config or load_models_config()
    backend, _, model = spec.partition(":")
    backend = SPEC_ALIASES.get(backend, backend)
    if backend not in cfg_all["backends"] or backend not in BACKEND_CLASSES:
        raise KeyError(f"unknown backend {backend!r}; known: {sorted(BACKEND_CLASSES)}")
    b = cfg_all["backends"][backend]
    model = model or b["models"][0]
    if model not in b["models"]:
        raise KeyError(f"model {model!r} is not configured for {backend}; configured: {b['models']}")

    secret = get_secret(b["key_env"]) if b.get("key_env") else None
    if b.get("key_env") and not secret:
        raise RuntimeError(f"{b['key_env']} is not set; add it to .env")

    endpoint = b.get("endpoint")
    if not endpoint and b.get("endpoint_env"):
        base = os.environ.get(b["endpoint_env"], "")
        endpoint = base.rstrip("/") + b.get("endpoint_path", "/v1/systemone") if base else None

    prices = (b.get("prices") or {}).get(model, {})
    cfg = BackendConfig(
        name=backend, model=model, endpoint=endpoint, secret=secret,
        price_input_per_m=float(prices.get("input", b.get("price_input_per_m", 0.0))),
        price_output_per_m=float(prices.get("output", b.get("price_output_per_m", 0.0))),
        max_options=int(b.get("max_options", 255)), timeout_s=float(b.get("timeout_s", 10)),
        extra={k: b[k] for k in ("price_per_credit",) if k in b},
    )
    return BACKEND_CLASSES[backend](cfg, location or cfg_all["meta"]["location"])
