# tests/test_live.py
"""Live smoke tests. Run with: uv run pytest -m live -v"""
import os
import pytest
from dalab.backends.registry import make_decider
from dalab.questions import Choice, Noul
from dalab.util.env import load_dotenv

pytestmark = pytest.mark.live
load_dotenv()
STATE = "User request: list files\nWorking directory: /work\nProposed shell command:\nls -la\nAdditional context: none"
QS = {"cls": Choice("Kind of action?", {"read_only": "reads only", "local_write": "writes locally", "outward": "leaves the workspace"}),
      "destructive": Noul("Irreversibly destroys data?")}

# expect_reported is the exact `model` the server must echo (spec section 8). It is pinned separately from the
# requested model because jevstation ignores `model` and always serves jev-1.13.0; when the jev-latest alias moves
# (spec section 10) these equalities, not a prefix check, are what will fail.
@pytest.mark.parametrize("spec,expect_reported", [
    ("upstage:solar-decide", "solar-decide"),
    ("openai:gpt-6-luna", "gpt-6-luna"),
    ("judge:gpt-6-luna", "gpt-6-luna"),
    ("jevstation:jev-1.13.0", "jev-1.13.0"),
    ("typesafe:jev-1.14.0", "jev-1.14.0"),
    ("typesafe:jev-1.13.0", "jev-1.13.0"),
])
def test_backend_answers(spec, expect_reported):
    backend = spec.split(":")[0]
    key_env = {"upstage": "UPSTAGE_API_KEY", "openai": "OPENAI_API_KEY", "judge": "OPENAI_API_KEY",
               "jevstation": "JEVSTATION_API_KEY", "typesafe": "TYPESAFE_API_KEY"}[backend]
    if not os.environ.get(key_env):
        pytest.skip(f"{key_env} not set")
    d = make_decider(spec).decide(STATE, QS)
    assert d.error is None, d.error
    assert d.answers["cls"].choice == "read_only" and d.answers["destructive"].noul < 0.5
    assert d.model_reported == expect_reported, f"{spec}: server answered with {d.model_reported!r}"
    assert d.usage_input_tokens and d.latency_ms is not None and d.extra["attempts"] >= 1
