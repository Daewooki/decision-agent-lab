#!/usr/bin/env python
"""Download the pinned public datasets, verify sha256, vendor the redistributable ones, write data/public/SOURCES.md.

Usage: uv run python scripts/fetch_data.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass(frozen=True)
class Source:
    name: str      # local set name: themsquared | godspede | laya_guard | aitejiu
    repo: str      # GitHub owner/repo
    commit: str    # 40-hex commit SHA the file is pinned to
    path: str      # path inside the repo
    sha256: str    # expected digest of the raw file
    vendor: bool   # copy into data/public/<name>/ (only if the license allows redistribution)
    kind: str      # data | license | meta
    vendored_sha256: str | None = None  # digest of the data/public/ copy when VENDOR_EDITS changes it


MANIFEST: list[Source] = [
    # themsquared/jev-calibration — Apache-2.0 (pinned 2026-10-20)
    Source("themsquared", "themsquared/jev-calibration", "311daf22fdae7b88696ed6384aa474af7d17cc3e", "tasks.jsonl",
           "a10a7ed68c6dd063b220c0505851ef7e5bd53bfa69962036774722ec1447fa66", True, "data"),
    Source("themsquared", "themsquared/jev-calibration", "311daf22fdae7b88696ed6384aa474af7d17cc3e", "LICENSE",
           "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30", True, "license"),
    # godspede/construct-auto-classifier — Apache-2.0
    Source("godspede", "godspede/construct-auto-classifier", "881fb45a96af4126dc854c4b6b77d813d41f4510", "bench/battery.jsonl",
           "56e57aebf10828268509018b3ec0976fa544a29e731826eb99a9a50f0e90a02b", True, "data"),
    Source("godspede", "godspede/construct-auto-classifier", "881fb45a96af4126dc854c4b6b77d813d41f4510", "bench/holdout.jsonl",
           "284c2f7b7d6c6342c0ffcd8f50074bec8b4caf2e22f0f16d415ceed4bc895009", True, "data"),
    Source("godspede", "godspede/construct-auto-classifier", "881fb45a96af4126dc854c4b6b77d813d41f4510", "bench/real-cases.jsonl",
           "38ee9706fe9d7bc80f722a995edcba35b3c71e1e2c4be219901b06e7508ab031", True, "data"),
    Source("godspede", "godspede/construct-auto-classifier", "881fb45a96af4126dc854c4b6b77d813d41f4510", "bench/world.json",
           "ee7f2005019ccf393cea204abf6c4c71e884f7eb12b82e1aee75ca4b0c1a169d", True, "meta"),
    Source("godspede", "godspede/construct-auto-classifier", "881fb45a96af4126dc854c4b6b77d813d41f4510", "LICENSE",
           "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4", True, "license"),
    # scs0209/laya-guard — MIT
    Source("laya_guard", "scs0209/laya-guard", "3859853dd69332fd7795c53151332dc3c29a3e7e", "eval/testset_shell.jsonl",
           "ede1c6f9a0cb00ccb15a89808d06bb8cbbfc5288d40f254c56d6f21bf7e1f394", True, "data"),
    Source("laya_guard", "scs0209/laya-guard", "3859853dd69332fd7795c53151332dc3c29a3e7e", "eval/testset_ko_30.jsonl",
           "485cc47c5e178cac8f4542820b2ba68037bccf199ce316122b0262ac7c9109fd", True, "data"),
    Source("laya_guard", "scs0209/laya-guard", "3859853dd69332fd7795c53151332dc3c29a3e7e", "eval/testset_incidents.jsonl",
           "8f5fadc632bca4daa803afb045853189c4f0d080f6c73df01bb2675e5eb2c7fd", True, "data",
           vendored_sha256="2bf062dc9d775946b7e3503b84af9830e07633f63b47344dda90714bf841b7d6"),
    Source("laya_guard", "scs0209/laya-guard", "3859853dd69332fd7795c53151332dc3c29a3e7e", "LICENSE",
           "8fa0a3b302e2d203350669ebd33b32d6c6980a44284819e2ce2f360470355e64", True, "license"),
    # Aitejiu/jev-harness-lab — NO LICENSE: runtime cache only, never vendored
    Source("aitejiu", "Aitejiu/jev-harness-lab", "29ddf8210e3a96f9d383e9b479e3e5c39455c394", "eval/datasets/bash_commands.jsonl",
           "81c610d3a6a9f075c0a4b8c744c8d3a085704bd6f598349b5120a659f7068267", False, "data"),
]

LICENSES = {"themsquared": "Apache-2.0", "godspede": "Apache-2.0", "laya_guard": "MIT", "aitejiu": "없음(재배포 불가, 런타임 다운로드)"}

# Vendored copies that deliberately differ from upstream. The upstream bytes are verified against
# Source.sha256 first; then rows whose MATCH_FIELD (the command, which the edits never touch) equals a
# VENDOR_EDITS key get these field values before the copy is written to data/public/. Only identifier
# fields may change: commands and labels stay verbatim, so no gate result depends on these edits, and the
# upstream record ids are never referenced here. SOURCES.md notes every edited file.
VENDOR_EDITS: dict[tuple[str, str], dict[str, dict[str, str]]] = {
    ("laya_guard", "eval/testset_incidents.jsonl"): {
        "rm -rf tests/ patches/ plan/ ~/": {"id": "agent-home-1", "incident": "agent-home-wipe"},
        "rm -rf tests/patches/plan/ ~/": {"id": "agent-home-2", "incident": "agent-home-wipe"},
    },
}
EDITABLE_FIELDS = frozenset({"id", "incident"})
MATCH_FIELD = "tool_call"


def raw_url(src: Source) -> str:
    return f"https://raw.githubusercontent.com/{src.repo}/{src.commit}/{src.path}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_one(src: Source, cache_root: Path, public_root: Path, client: httpx.Client) -> Path:
    target = cache_root / src.name / Path(src.path).name
    if not target.exists() or sha256_file(target) != src.sha256:
        target.parent.mkdir(parents=True, exist_ok=True)
        r = client.get(raw_url(src), follow_redirects=True)
        r.raise_for_status()
        target.write_bytes(r.content)
    got = sha256_file(target)
    if got != src.sha256:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"sha256 mismatch for {src.repo}@{src.commit[:8]}:{src.path}: got {got}, expected {src.sha256}")
    if src.vendor:
        data = apply_vendor_edits(src, target.read_bytes())
        want = src.vendored_sha256 or src.sha256
        got = hashlib.sha256(data).hexdigest()
        if got != want:
            raise RuntimeError(f"vendored copy of {src.repo}:{src.path} hashes to {got}, expected {want}")
        dest = public_root / src.name / Path(src.path).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return target


def apply_vendor_edits(src: Source, raw: bytes) -> bytes:
    """Return the bytes to vendor for `src`: upstream unchanged, or with its VENDOR_EDITS applied.

    Rows are matched on MATCH_FIELD (the command), which the edits never change, so the upstream
    record ids are not referenced.
    """
    edits = VENDOR_EDITS.get((src.name, src.path))
    if not edits:
        return raw
    out, seen = [], set()
    for line in raw.decode("utf-8").splitlines(keepends=True):
        rec = json.loads(line) if line.strip() else None
        key = rec.get(MATCH_FIELD) if isinstance(rec, dict) else None
        patch = edits.get(key)
        if patch:
            if not set(patch) <= EDITABLE_FIELDS:
                raise ValueError(f"VENDOR_EDITS may only change {sorted(EDITABLE_FIELDS)}, got {sorted(patch)}")
            seen.add(key)
            rec.update(patch)
            line = json.dumps(rec, ensure_ascii=False) + line[len(line.rstrip("\r\n")):]
        out.append(line)
    missing = set(edits) - seen
    if missing:
        raise RuntimeError(f"VENDOR_EDITS rows not found in {src.repo}:{src.path}: {sorted(missing)}")
    return "".join(out).encode("utf-8")


def count_rows(path: Path) -> int | None:
    if path.suffix != ".jsonl":
        return None
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def fetch_all(cache_root: Path, public_root: Path, client: httpx.Client | None = None) -> list[Path]:
    client = client or httpx.Client(timeout=60)
    return [fetch_one(s, cache_root, public_root, client) for s in MANIFEST]


def write_sources_md(manifest: list[Source], counts: dict[str, int], path: Path) -> None:
    lines = ["# 데이터 출처 (자동 생성: scripts/fetch_data.py)", "",
             "커밋 SHA와 sha256으로 고정된 스냅샷입니다. 라이선스가 재배포를 허용하는 세트만 이 폴더에 복사했고, "
             "라이선스가 없는 세트는 런타임에 `data/cache/`로만 내려받습니다.", "",
             "| 세트 | 저장소 | 커밋 | 파일 | sha256 | 라이선스 | 재배포 | 행 수 |", "|---|---|---|---|---|---|---|---|"]
    for s in manifest:
        n = counts.get(f"{s.name}/{Path(s.path).name}", counts.get(s.name) if s.kind == "data" else None)
        lines.append(f"| {s.name} | https://github.com/{s.repo} | `{s.commit[:12]}` | `{s.path}` | `{s.sha256[:16]}…` | "
                     f"{LICENSES[s.name]} | {'예' if s.vendor else '아니오(런타임)'} | {n if n is not None else ''} |")
    for s in manifest:
        if not s.vendored_sha256:
            continue
        rows = VENDOR_EDITS.get((s.name, s.path), {})
        fields = ", ".join(f"`{f}`" for f in sorted({f for patch in rows.values() for f in patch}))
        lines += ["", f"`{s.name}/{Path(s.path).name}`은 원본 레코드 {len(rows)}건의 {fields} 값만 "
                  "`scripts/fetch_data.py`의 `VENDOR_EDITS`대로 바꿔 동봉했습니다. 명령과 라벨은 원본과 같습니다. "
                  f"위 표의 sha256은 원본 파일 기준이고, 동봉본의 sha256은 `{s.vendored_sha256[:16]}…`입니다."]
    lines += ["", "라벨 매핑과 제외 규칙은 `src/dalab/gate/datasets.py`에 있습니다."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cache_root, public_root = root / "data" / "cache", root / "data" / "public"
    paths = fetch_all(cache_root, public_root)
    counts: dict[str, int] = {}
    for s, p in zip(MANIFEST, paths, strict=True):
        n = count_rows(p)
        if n is not None:
            counts[f"{s.name}/{p.name}"] = n
    write_sources_md(MANIFEST, counts, public_root / "SOURCES.md")
    print(json.dumps(counts, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
