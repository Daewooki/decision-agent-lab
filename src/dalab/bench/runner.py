"""Resumable, threaded execution of a Decider over items, with an append-only JSONL store."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dalab.backends.base import Decider
from dalab.questions import Question, to_systemone


@dataclass
class Item:
    id: str
    state: str
    questions: dict[str, Question]
    meta: dict[str, Any] = field(default_factory=dict)


def cache_key(backend: str, model: str, state: str, questions: dict[str, Question]) -> str:
    wire = to_systemone(questions)
    for q in wire.values():  # option order is a run condition (order_seed in the store identity), not part of the key
        if isinstance(q.get("criteria"), dict):
            q["criteria"] = dict(sorted(q["criteria"].items()))
    payload = json.dumps({"b": backend, "m": model, "s": state, "q": wire}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


Identity = tuple[str, str, int, int | None]  # (key, condition, repeat, order_seed): one backend call per identity


def _identity(row: dict[str, Any]) -> Identity:
    return (row["key"], row["condition"], row["repeat"], row.get("order_seed"))


class RawStore:
    """Append-only JSONL of run rows.

    One backend call is made per identity (key, condition, repeat, order_seed); spec section 4 puts the option
    order into the cache key, so a permuted run never serves an unpermuted row or vice versa. Rows are kept per
    item: two items with identical content share the call but keep their own item_id and meta.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._calls: dict[Identity, dict[str, Any]] = {}  # the row that made the call for each identity
        self._items: dict[tuple[Identity, str], dict[str, Any]] = {}  # row per (identity, item_id)
        self._order: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        if self.path.exists():
            for row in self._load():
                self._index(row)

    def _load(self) -> list[dict[str, Any]]:
        lines = self.path.read_text(encoding="utf-8").splitlines()
        last = max((i for i, line in enumerate(lines) if line.strip()), default=-1)
        rows = []
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                if i != last:
                    raise ValueError(f"{self.path}: line {i + 1} is not valid JSON ({e.msg})") from e
                # A run killed mid-write leaves a truncated tail. Drop it (the call is simply redone) and cut it
                # from the file so the next append does not glue a new row onto the garbage.
                warnings.warn(f"{self.path}: dropping truncated last line {i + 1} ({e.msg})", stacklevel=3)
                self.path.write_text("".join(l + "\n" for l in lines[:i]), encoding="utf-8")
        return rows

    def _index(self, row: dict[str, Any]) -> None:
        ident = _identity(row)
        self._calls.setdefault(ident, row)
        self._items[(ident, row["item_id"])] = row
        self._order.append(row)

    def __len__(self) -> int:
        return len(self._order)

    def has(self, key: str, condition: str, repeat: int, order_seed: int | None = None) -> bool:
        return (key, condition, repeat, order_seed) in self._calls

    def get(self, key: str, condition: str, repeat: int, order_seed: int | None = None, *, item_id: str | None = None) -> dict[str, Any] | None:
        """The row that made the call for this identity, or with item_id the row stored for that item."""
        ident = (key, condition, repeat, order_seed)
        return self._calls.get(ident) if item_id is None else self._items.get((ident, item_id))

    def rows(self) -> list[dict[str, Any]]:
        return list(self._order)

    def append(self, row: dict[str, Any]) -> None:
        line = json.dumps(row, ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())  # a killed run loses at most the row being written
            self._index(row)


def _row(decider: Decider, item: Item, key: str, run_id: str, condition: str, repeat: int, order_seed: int | None) -> dict[str, Any]:
    d = decider.decide(item.state, item.questions, order_seed=order_seed)
    return {"key": key, "run_id": run_id, "item_id": item.id, "condition": condition, "repeat": repeat,
            "order_seed": order_seed, "meta": item.meta, "decision": d.to_dict()}


def run_items(
    decider: Decider,
    items: list[Item],
    *,
    store: RawStore,
    run_id: str,
    condition: str = "base",
    order_seed: int | None = None,
    repeat: int = 0,
    concurrency: int = 4,
    warmup: int = 0,
    progress: bool = False,
) -> list[dict[str, Any]]:
    """Decide each item once under (condition, repeat, order_seed); reuse stored rows; return rows in item order."""
    if warmup and items:
        first = items[0]
        key = cache_key(decider.name, decider.model_requested, first.state, first.questions)
        # Continue numbering after any warmup rows an earlier run already stored for this key.
        offset = len([r for r in store.rows() if r["condition"] == "warmup" and r["key"] == key])
        for i in range(warmup):
            store.append(_row(decider, first, key, run_id, "warmup", offset + i, None))

    keyed = [(item, cache_key(decider.name, decider.model_requested, item.state, item.questions)) for item in items]
    todo, claimed = [], set()
    for item, key in keyed:
        if not store.has(key, condition, repeat, order_seed) and key not in claimed:
            claimed.add(key)  # one call per identity; items with the same content get a copy of the row below
            todo.append((item, key))

    def work(pair):
        item, key = pair
        row = _row(decider, item, key, run_id, condition, repeat, order_seed)
        store.append(row)
        return row

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            it = pool.map(work, todo)
            if progress:
                from tqdm import tqdm
                it = tqdm(it, total=len(todo), desc=f"{decider.name}:{decider.model_requested}:{condition}")
            for _ in it:
                pass

    out = []
    for item, key in keyed:
        row = store.get(key, condition, repeat, order_seed, item_id=item.id)
        if row is None:  # same content as another item: share its decision, keep this item's id and meta
            src = store.get(key, condition, repeat, order_seed)
            assert src is not None
            row = {**src, "item_id": item.id, "meta": item.meta}
            store.append(row)
        out.append(row)
    return out
