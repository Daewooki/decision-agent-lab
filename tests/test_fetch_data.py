# tests/test_fetch_data.py
import hashlib
import importlib.util
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fetch_data", ROOT / "scripts" / "fetch_data.py")
fetch_data = importlib.util.module_from_spec(spec)
sys.modules["fetch_data"] = fetch_data
spec.loader.exec_module(fetch_data)


def test_manifest_is_unique_and_pinned():
    keys = [(s.repo, s.commit, s.path) for s in fetch_data.MANIFEST]
    assert len(keys) == len(set(keys))
    for s in fetch_data.MANIFEST:
        assert len(s.commit) == 40 and len(s.sha256) == 64 and s.name in {"themsquared", "godspede", "laya_guard", "aitejiu"}
    assert all(not s.vendor for s in fetch_data.MANIFEST if s.name == "aitejiu")


def test_sha256_file(tmp_path):
    p = tmp_path / "x"
    p.write_bytes(b"hello")
    assert fetch_data.sha256_file(p) == hashlib.sha256(b"hello").hexdigest()


def test_fetch_verifies_and_vendors(tmp_path):
    body = b'{"id": 1}\n'
    good = fetch_data.Source("themsquared", "o/r", "a" * 40, "tasks.jsonl", hashlib.sha256(body).hexdigest(), True, "data")
    bad = fetch_data.Source("aitejiu", "o/r", "b" * 40, "x.jsonl", "0" * 64, False, "data")
    client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, content=body)))
    out = fetch_data.fetch_one(good, tmp_path / "cache", tmp_path / "public", client)
    assert out == tmp_path / "cache" / "themsquared" / "tasks.jsonl" and out.read_bytes() == body
    assert (tmp_path / "public" / "themsquared" / "tasks.jsonl").read_bytes() == body
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        fetch_data.fetch_one(bad, tmp_path / "cache", tmp_path / "public", client)
    assert not (tmp_path / "public" / "aitejiu").exists()


def test_sources_md_lists_every_source(tmp_path):
    p = tmp_path / "SOURCES.md"
    fetch_data.write_sources_md(fetch_data.MANIFEST, {"themsquared": 240}, p)
    text = p.read_text(encoding="utf-8")
    assert "themsquared/jev-calibration" in text and "311daf22" in text and "Aitejiu/jev-harness-lab" in text
    assert "런타임" in text and "240" in text
