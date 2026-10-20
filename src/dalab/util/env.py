"""Minimal .env loading and secret masking (no python-dotenv dependency)."""
from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path

_SECRET_NAME = re.compile(r"^[A-Z0-9_]*(KEY|TOKEN|SECRET)[A-Z0-9_]*$")


def find_repo_root(start: Path | None = None) -> Path:
    """Nearest ancestor (including start) that contains pyproject.toml; falls back to start."""
    p = (start or Path.cwd()).resolve()
    for cand in (p, *p.parents):
        if (cand / "pyproject.toml").exists():
            return cand
    return p


def load_dotenv(path: Path | None = None) -> None:
    """Read KEY=VALUE lines into os.environ. Existing variables win. Empty values are skipped."""
    path = path or find_repo_root() / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line or line.startswith("export "):
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def get_secret(name: str) -> str | None:
    value = os.environ.get(name, "")
    return value or None


def mask_secrets(text: str, values: Iterable[str] | None = None) -> str:
    """Replace secret values with ***. Defaults to every env var whose name looks like a secret."""
    if values is None:
        values = [v for k, v in os.environ.items() if _SECRET_NAME.match(k)]
    for v in sorted((v for v in values if v and len(v) >= 8), key=len, reverse=True):
        text = text.replace(v, "***")
    return text
