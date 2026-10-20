# tests/test_runner.py
import json
import threading
import pytest
from dalab.backends.base import BackendConfig, Decider
from dalab.bench.runner import Item, RawStore, cache_key, run_items
from dalab.decision import Answer
from dalab.questions import Noul

class FakeDecider(Decider):
    """Returns noul=0.5 and counts calls; _call keeps the plan's 3-tuple shape to show it is still accepted."""
    def __init__(self):
        super().__init__(BackendConfig("fake", "m1", None, None, 1.0))
        self.calls, self.lock = 0, threading.Lock()
    def _call(self, state, questions):
        with self.lock:
            self.calls += 1
        return {"model": "m1-reported", "state": state, "usage": {"input_tokens": 10, "output_tokens": 0}}, 1.5, None
    def _normalize(self, raw, questions):
        return raw["model"], {n: Answer("noul", noul=0.5) for n in questions}, 10, 0, {}

def items(n):
    return [Item(f"i{k}", f"state {k}", {"q": Noul("ok?")}, {"gold": "allow"}) for k in range(n)]

def test_cache_key_ignores_option_order_and_is_stable(gate_questions):
    from dalab.questions import permute_choices
    k1 = cache_key("typesafe", "jev-1.14.0", "s", gate_questions)
    k2 = cache_key("typesafe", "jev-1.14.0", "s", permute_choices(gate_questions, 5)[0])
    assert k1 == k2 and len(k1) == 64
    assert cache_key("typesafe", "jev-1.13.0", "s", gate_questions) != k1

def test_run_items_stores_rows_and_skips_cached(tmp_path):
    store = RawStore(tmp_path / "raw.jsonl")
    dec = FakeDecider()
    rows = run_items(dec, items(5), store=store, run_id="r1", concurrency=3)
    assert [r["item_id"] for r in rows] == [f"i{k}" for k in range(5)] and dec.calls == 5
    assert rows[0]["decision"]["model_reported"] == "m1-reported" and rows[0]["meta"] == {"gold": "allow"}
    assert rows[0]["condition"] == "base" and rows[0]["repeat"] == 0 and rows[0]["order_seed"] is None
    rows2 = run_items(dec, items(5), store=RawStore(tmp_path / "raw.jsonl"), run_id="r2", concurrency=3)
    assert dec.calls == 5 and rows2[3]["run_id"] == "r1"  # served from the store, original run id kept
    assert len(RawStore(tmp_path / "raw.jsonl")) == 5

def test_repeat_and_condition_make_new_calls(tmp_path):
    store = RawStore(tmp_path / "raw.jsonl")
    dec = FakeDecider()
    run_items(dec, items(2), store=store, run_id="r1")
    run_items(dec, items(2), store=store, run_id="r1", repeat=1)
    run_items(dec, items(2), store=store, run_id="r1", condition="perm1", order_seed=1)
    assert dec.calls == 6 and len(store) == 6
    key = cache_key("fake", "m1", "state 0", items(1)[0].questions)
    assert store.has(key, "perm1", 0, 1) and store.has(key, "base", 0) and store.has(key, "base", 1)
    assert not store.has(key, "perm1", 0) and not store.has("nope", "base", 0)

def test_order_seed_is_part_of_the_store_identity(tmp_path, gate_questions):
    store = RawStore(tmp_path / "raw.jsonl")
    dec = FakeDecider()
    it = [Item("c", "s", gate_questions)]
    seeded = run_items(dec, it, store=store, run_id="r1", order_seed=3)
    plain = run_items(dec, it, store=store, run_id="r1", order_seed=None)
    assert dec.calls == 2  # a permuted row never stands in for the unpermuted condition or vice versa
    assert seeded[0]["order_seed"] == 3 and plain[0]["order_seed"] is None
    assert run_items(dec, it, store=RawStore(tmp_path / "raw.jsonl"), run_id="r2", order_seed=3)[0]["order_seed"] == 3
    assert dec.calls == 2

def test_duplicate_content_shares_one_call_but_keeps_each_items_row(tmp_path):
    store = RawStore(tmp_path / "raw.jsonl")
    dec = FakeDecider()
    dup = [Item("en-1", "same state", {"q": Noul("ok?")}, {"gold": "allow"}),
           Item("en-1-dup", "same state", {"q": Noul("ok?")}, {"gold": "deny", "injection": "crude"})]
    rows = run_items(dec, dup, store=store, run_id="r1")
    assert dec.calls == 1 and [r["item_id"] for r in rows] == ["en-1", "en-1-dup"]
    assert rows[1]["meta"] == {"gold": "deny", "injection": "crude"} and rows[1]["decision"] == rows[0]["decision"]
    assert len(store) == 2 and [r["item_id"] for r in store.rows()] == ["en-1", "en-1-dup"]
    rows2 = run_items(dec, dup, store=RawStore(tmp_path / "raw.jsonl"), run_id="r2")  # resumed: both rows served, no call
    assert dec.calls == 1 and [r["item_id"] for r in rows2] == ["en-1", "en-1-dup"] and rows2[1]["meta"]["gold"] == "deny"

def test_warmup_rows_are_stored_separately(tmp_path):
    store = RawStore(tmp_path / "raw.jsonl")
    dec = FakeDecider()
    rows = run_items(dec, items(2), store=store, run_id="r1", warmup=3)
    assert dec.calls == 5 and len(rows) == 2
    warm = [r for r in store.rows() if r["condition"] == "warmup"]
    assert len(warm) == 3 and all(r["item_id"] == "i0" for r in warm)

def test_truncated_last_line_is_dropped_and_the_run_resumes(tmp_path):
    path = tmp_path / "raw.jsonl"
    dec = FakeDecider()
    run_items(dec, items(2), store=RawStore(path), run_id="r1")
    with path.open("a", encoding="utf-8") as f:
        f.write('{"key": "k3", "run_id": "r1", "item_id": "i2", "condi')  # killed mid-write
    with pytest.warns(UserWarning, match="truncated last line 3"):
        store = RawStore(path)
    assert len(store) == 2
    rows = run_items(dec, items(3), store=store, run_id="r2")
    assert dec.calls == 3 and [r["item_id"] for r in rows] == ["i0", "i1", "i2"]
    reloaded = RawStore(path)  # the garbage was cut, so the file is clean again
    assert len(reloaded) == 3 and all(json.loads(l) for l in path.read_text().splitlines())

def test_corrupt_middle_line_is_reported_with_its_number(tmp_path):
    path = tmp_path / "raw.jsonl"
    run_items(FakeDecider(), items(2), store=RawStore(path), run_id="r1")
    lines = path.read_text().splitlines()
    path.write_text(lines[0][:20] + "\n" + lines[1] + "\n")
    with pytest.raises(ValueError, match="line 1 is not valid JSON"):
        RawStore(path)
