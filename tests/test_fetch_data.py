# tests/test_fetch_data.py
import dataclasses
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
    edited = {(s.name, s.path) for s in fetch_data.MANIFEST if s.vendored_sha256}
    assert edited == set(fetch_data.VENDOR_EDITS)
    assert all(s.vendor and len(s.vendored_sha256) == 64 for s in fetch_data.MANIFEST if s.vendored_sha256)


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


def test_committed_vendored_files_match_manifest():
    # data/public/ must hold exactly what fetch_data.py would vendor, so running it never dirties the tree.
    for s in fetch_data.MANIFEST:
        if not s.vendor:
            continue
        p = ROOT / "data" / "public" / s.name / Path(s.path).name
        assert p.exists(), p
        assert fetch_data.sha256_file(p) == (s.vendored_sha256 or s.sha256), f"{p} drifted from the manifest"


def test_vendor_edits_change_only_identifiers(monkeypatch):
    # Rows are matched on their command (MATCH_FIELD), not their upstream id.
    upstream = (b'{"id": "a-1", "incident": "x", "expected": "block", "tool_call": "rm -rf ~/"}\n'
                b'{"id": "b-1", "incident": "y", "expected": "allow", "tool_call": "ls"}\n')
    src = fetch_data.Source("laya_guard", "o/r", "c" * 40, "eval/t.jsonl", hashlib.sha256(upstream).hexdigest(), True, "data")
    key = ("laya_guard", "eval/t.jsonl")
    monkeypatch.setitem(fetch_data.VENDOR_EDITS, key, {"rm -rf ~/": {"id": "n-1", "incident": "n"}})
    out = fetch_data.apply_vendor_edits(src, upstream).splitlines(keepends=True)
    assert out[0] == b'{"id": "n-1", "incident": "n", "expected": "block", "tool_call": "rm -rf ~/"}\n'
    assert out[1] == upstream.splitlines(keepends=True)[1]
    monkeypatch.setitem(fetch_data.VENDOR_EDITS, key, {"rm -rf ~/": {"expected": "allow"}})
    with pytest.raises(ValueError, match="may only change"):
        fetch_data.apply_vendor_edits(src, upstream)
    monkeypatch.setitem(fetch_data.VENDOR_EDITS, key, {"no such command": {"id": "n-9"}})
    with pytest.raises(RuntimeError, match="not found"):
        fetch_data.apply_vendor_edits(src, upstream)


def test_fetch_vendors_edited_copy_and_checks_its_digest(tmp_path, monkeypatch):
    upstream = b'{"id": "a-1", "incident": "x", "expected": "block", "tool_call": "rm -rf ~/"}\n'
    edited = b'{"id": "n-1", "incident": "x", "expected": "block", "tool_call": "rm -rf ~/"}\n'
    monkeypatch.setitem(fetch_data.VENDOR_EDITS, ("laya_guard", "eval/t.jsonl"), {"rm -rf ~/": {"id": "n-1"}})
    client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, content=upstream)))
    src = fetch_data.Source("laya_guard", "o/r", "c" * 40, "eval/t.jsonl", hashlib.sha256(upstream).hexdigest(), True, "data",
                            vendored_sha256=hashlib.sha256(edited).hexdigest())
    fetch_data.fetch_one(src, tmp_path / "cache", tmp_path / "public", client)
    assert (tmp_path / "cache" / "laya_guard" / "t.jsonl").read_bytes() == upstream
    assert (tmp_path / "public" / "laya_guard" / "t.jsonl").read_bytes() == edited
    with pytest.raises(RuntimeError, match="vendored copy"):
        fetch_data.fetch_one(dataclasses.replace(src, vendored_sha256="0" * 64), tmp_path / "cache", tmp_path / "public2", client)
    assert not (tmp_path / "public2").exists()


def test_sources_md_lists_every_source(tmp_path):
    p = tmp_path / "SOURCES.md"
    fetch_data.write_sources_md(fetch_data.MANIFEST, {"themsquared": 240}, p)
    text = p.read_text(encoding="utf-8")
    assert "themsquared/jev-calibration" in text and "311daf22" in text and "Aitejiu/jev-harness-lab" in text
    assert "런타임" in text and "240" in text
    assert "VENDOR_EDITS" in text and "2bf062dc9d775946" in text
