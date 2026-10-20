# tests/test_registry.py
import pytest
from dalab.backends.llm_judge import LLMJudgeDecider
from dalab.backends.openai_decisions import OpenAIDecisionsDecider
from dalab.backends.registry import load_models_config, make_decider
from dalab.backends.systemone import JevstationDecider, SystemOneDecider

CFG = {
    "meta": {"prices_checked": "2026-10-20", "location": "Seoul, KR"},
    "backends": {
        "typesafe": {"endpoint": "https://api.typesafe.ai/v1/systemone", "key_env": "TYPESAFE_API_KEY", "models": ["jev-1.13.0", "jev-1.14.0"],
                     "price_input_per_m": 0.042, "max_options": 255, "timeout_s": 10},
        "jevstation": {"endpoint": "https://jevstation.com/api/v1/systemone", "key_env": "JEVSTATION_API_KEY", "models": ["jev-1.13.0"],
                       "price_input_per_m": 0.0, "price_per_credit": 0.00198},
        "ollama": {"endpoint_env": "OLLAMA_BASE_URL", "endpoint_path": "/v1/systemone", "models": ["nimble"], "price_input_per_m": 0.0, "max_options": 26, "timeout_s": 120},
        "openai_decisions": {"key_env": "OPENAI_API_KEY", "models": ["gpt-6-luna"], "price_input_per_m": 0.10},
        "llm_judge": {"key_env": "OPENAI_API_KEY", "models": ["gpt-6-luna", "gpt-6-sol"],
                      "prices": {"gpt-6-luna": {"input": 0.10, "output": 0.50}, "gpt-6-sol": {"input": 2.0, "output": 10.0}}},
    },
}

@pytest.fixture(autouse=True)
def keys(monkeypatch):
    # Unit tests must not depend on the developer's .env: make_decider() calls load_dotenv(), which would
    # otherwise re-populate a variable that test_errors deliberately deletes.
    monkeypatch.setattr("dalab.backends.registry.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-ts")
    monkeypatch.setenv("JEVSTATION_API_KEY", "sk_js")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oa")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/")

def test_real_config_file_loads():
    cfg = load_models_config()
    assert "typesafe" in cfg["backends"] and cfg["backends"]["typesafe"]["models"] == ["jev-1.13.0", "jev-1.14.0"]
    assert cfg["meta"]["prices_checked"] == "2026-10-20"

def test_typesafe_spec():
    d = make_decider("typesafe:jev-1.14.0", CFG)
    assert isinstance(d, SystemOneDecider) and d.cfg.model == "jev-1.14.0" and d.cfg.secret == "sk-ts"
    assert d.cfg.endpoint == "https://api.typesafe.ai/v1/systemone" and d.cfg.price_input_per_m == 0.042 and d.location == "Seoul, KR"

def test_default_model_and_aliases():
    assert make_decider("typesafe", CFG).cfg.model == "jev-1.13.0"
    oa = make_decider("openai:gpt-6-luna", CFG)
    assert isinstance(oa, OpenAIDecisionsDecider) and oa.cfg.endpoint is None
    j = make_decider("judge:gpt-6-sol", CFG)
    assert isinstance(j, LLMJudgeDecider) and j.cfg.price_input_per_m == 2.0 and j.cfg.price_output_per_m == 10.0

def test_jevstation_and_ollama():
    js = make_decider("jevstation", CFG)
    assert isinstance(js, JevstationDecider) and js.cfg.extra == {"price_per_credit": 0.00198}
    ol = make_decider("ollama:nimble", CFG)
    assert ol.cfg.endpoint == "http://localhost:11434/v1/systemone" and ol.cfg.secret is None and ol.cfg.max_options == 26

def test_openrouter_prices_solar_decide_per_model(monkeypatch):
    # The flat openrouter price is Jev's; upstage/solar-decide has its own OpenRouter listing (see config/models.toml).
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("UPSTAGE_API_KEY", "sk-up-test")
    cfg = load_models_config()
    assert make_decider("openrouter:upstage/solar-decide", cfg).cfg.price_input_per_m == 0.05
    assert make_decider("openrouter:typesafe/jev-1.14", cfg).cfg.price_input_per_m == 0.042
    assert make_decider("upstage:solar-decide", cfg).cfg.price_input_per_m == 0.10

def test_errors(monkeypatch):
    with pytest.raises(KeyError, match="not configured"):
        make_decider("typesafe:jev-9.9.9", CFG)
    with pytest.raises(KeyError, match="unknown backend"):
        make_decider("nope:x", CFG)
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        make_decider("openai", CFG)
