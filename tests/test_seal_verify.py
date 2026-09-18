"""Phase 1 seal tests: write outer, seal, verify OK, tamper -> FAIL."""

from __future__ import annotations

import json
import os

from blackbox.storage.layout import allocate_run, write_outer_json
from blackbox.storage.seal import seal_run, verify_run


def _outer(run_id: str, sha: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "model_id": "M-001",
        "checkpoint_sha256": sha,
        "watchlist_ref": {
            "watchlist_id": "outer-only-v1",
            "watchlist_sha256": "00" * 32,
            "patch": None,
        },
        "prompt": {"text": "Hello", "tokens": [1, 2], "tokenizer": "test"},
        "output": {"text": "Hi", "tokens": [3], "finish_reason": "stop"},
        "logits_summary": None,
        "timing": {
            "t_start_iso": "2026-09-15T10:00:00Z",
            "t_end_iso": "2026-09-15T10:00:01Z",
            "ms_per_token": 1.0,
            "cuda_event_ms": 0.0,
        },
        "sampling": {"seed": 42, "temp": 0.0, "top_p": 1.0, "sampler": "greedy"},
        "versions": {"torch": "2.4.0", "cuda": "none", "blackbox": "0.1.0", "adapter": "cpu"},
        "cost": None,
        "flags": {"bad_output": False, "ring_overflow": False, "truncated": False},
        "tags": ["good"],
        "replay": {
            "seed": 42,
            "temp": 0.0,
            "top_p": 1.0,
            "sampler": "greedy",
            "cudnn_deterministic": True,
            "checkpoint_sha256": sha,
            "versions": {"torch": "2.4.0", "cuda": "none"},
        },
        "error": None,
    }


def test_seal_verify_ok_tamper_fail(tmp_path: object, monkeypatch: object) -> None:
    store = os.path.join(str(tmp_path), "store")
    monkeypatch.setenv("BLACKBOX_DIR", store)
    rid, rpath = allocate_run(store)
    assert rid == "RUN-000001"
    sha = "ab" * 32
    write_outer_json(rpath, _outer(rid, sha))  # type: ignore[arg-type]
    digest = seal_run(rpath)
    assert len(digest) == 64
    assert verify_run(rid, store) is True
    with open(os.path.join(rpath, "outer.json"), encoding="utf-8") as fh:
        obj = json.load(fh)
    obj["prompt"]["text"] = "Hello!"  # 1-byte-class tamper
    with open(os.path.join(rpath, "outer.json"), "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
    assert verify_run(rid, store) is False


def test_chain_continuity(tmp_path: object) -> None:
    store = os.path.join(str(tmp_path), "store")
    sha = "cd" * 32
    rid1, rp1 = allocate_run(store)
    write_outer_json(rp1, _outer(rid1, sha))  # type: ignore[arg-type]
    seal_run(rp1)
    rid2, rp2 = allocate_run(store)
    assert rid2 == "RUN-000002"
    write_outer_json(rp2, _outer(rid2, sha))  # type: ignore[arg-type]
    seal_run(rp2)
    assert verify_run(rid1, store) is True
    assert verify_run(rid2, store) is True
    with open(os.path.join(rp2, "seal.json"), encoding="utf-8") as fh:
        seal2 = json.load(fh)
    with open(os.path.join(rp1, "seal.json"), encoding="utf-8") as fh:
        seal1 = json.load(fh)
    assert seal2["prev_hash"] == seal1["hash"]
