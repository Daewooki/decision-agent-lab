# tests/test_env.py
import os
from pathlib import Path
from dalab.util.env import find_repo_root, get_secret, load_dotenv, mask_secrets

def test_load_dotenv_sets_missing_only(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nFOO_API_KEY=abc12345\nBAR_TOKEN='quoted'\nEMPTY=\nexport NOPE=1\n")
    # load_dotenv writes os.environ directly, and delenv on an absent key records nothing to undo;
    # setenv first so monkeypatch removes FOO_API_KEY at teardown instead of leaking it to later tests.
    monkeypatch.setenv("FOO_API_KEY", "placeholder")
    monkeypatch.delenv("FOO_API_KEY", raising=False)
    monkeypatch.setenv("BAR_TOKEN", "preexisting")
    monkeypatch.delenv("EMPTY", raising=False)
    load_dotenv(env)
    assert os.environ["FOO_API_KEY"] == "abc12345"
    assert os.environ["BAR_TOKEN"] == "preexisting"
    assert "EMPTY" not in os.environ
    assert get_secret("FOO_API_KEY") == "abc12345" and get_secret("EMPTY") is None

def test_mask_secrets_explicit_and_env(monkeypatch):
    monkeypatch.setenv("SOME_SECRET", "s3cr3tvalue")
    text = "key=abc12345 other=s3cr3tvalue short=ab"
    assert mask_secrets(text, ["abc12345"]) == "key=*** other=s3cr3tvalue short=ab"
    assert mask_secrets(text) == "key=abc12345 other=*** short=ab"

def test_find_repo_root_walks_up(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    assert find_repo_root(sub) == tmp_path.resolve()
