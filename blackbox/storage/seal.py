"""Hash seal + chain verify (spec §FILE-seal, §A.5, plan.md §8.4).

Normative formula (verbatim)::

    hash_i=SHA256(prev||SHA256(outer)||SHA256(deep)||canonical(replay))

``SHA256(deep)`` is the sha of the empty string when Deep is absent.
``canonical(replay)`` is canonical JSON (sorted keys, no whitespace) of
``outer.json:replay``. Genesis ``prev`` is 64x``0``.

Verify recomputes 5 steps: outer sha, deep sha, canonical replay,
``hash_i``, then compare + chain continuity. Any 1-byte edit -> FAIL.

Exit mapping for CLI: E06 ``RunNotFoundError`` (unknown run), E07 FAIL
(``verify()`` returns ``False``; ``SealBrokenError`` is programmatic twin).
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

from blackbox.exceptions import RunNotFoundError
from blackbox.storage.layout import get_store_dir, run_dir

GENESIS_PREV = "0" * 64
ALGORITHM = "hash_i=SHA256(prev||SHA256(outer)||SHA256(deep)||canonical(replay))"
EMPTY_SHA = hashlib.sha256(b"").hexdigest()


def _sha_file(path: str) -> str:
    """SHA256 hex of file bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_replay(replay: dict[str, Any]) -> str:
    """Canonical JSON: sorted keys, no whitespace."""
    return json.dumps(replay, sort_keys=True, separators=(",", ":"))


def compute_hash(prev: str, sha_outer: str, sha_deep: str, replay: dict[str, Any]) -> str:
    """Compute ``hash_i`` from parts per the normative formula."""
    return hashlib.sha256(
        (prev + sha_outer + sha_deep + canonical_replay(replay)).encode("utf-8")
    ).hexdigest()


def _chain_path_for_run(run_path: str) -> str:
    """``chain.jsonl`` for a run dir (``$BLACKBOX_DIR/chain.jsonl``)."""
    store = os.path.dirname(os.path.dirname(os.path.normpath(run_path)))
    return os.path.join(store, "chain.jsonl")


def _last_chain_hash(chain_path: str) -> str:
    """Last ``hash`` in chain, or genesis when absent/empty."""
    if not os.path.isfile(chain_path):
        return GENESIS_PREV
    prev = GENESIS_PREV
    with open(chain_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                prev = str(json.loads(line)["hash"])
            except (ValueError, KeyError, TypeError):
                continue
    return prev


def seal_run(run_path: str, sealed_by: str = "blackbox-phase1") -> str:
    """Seal a run dir: write ``seal.json`` + append ``chain.jsonl``.

    ``SHA256(deep)`` covers ``deep.bin`` bytes, or the empty-string hash when
    Deep is absent (``triggered=false`` / ``outer_only``); ``verify`` walks
    the same chain including the deep bytes, so any 1-byte edit fails.

    Returns:
        The ``hash_i`` hex string.

    Raises:
        FileNotFoundError: E02/E06 missing ``outer.json``.
        ValueError: E02 missing ``replay`` object in ``outer.json``.
    """
    outer_path = os.path.join(run_path, "outer.json")
    if not os.path.isfile(outer_path):
        raise FileNotFoundError(f"E02: no outer.json in {run_path}")
    with open(outer_path, encoding="utf-8") as fh:
        outer = json.load(fh)
    if not isinstance(outer.get("replay"), dict):
        raise ValueError(f"E02: outer.json missing replay in {run_path}")
    run_id = str(outer.get("run_id", os.path.basename(run_path)))
    sha_outer = _sha_file(outer_path)
    deep_path = os.path.join(run_path, "deep.bin")
    sha_deep = _sha_file(deep_path) if os.path.isfile(deep_path) else EMPTY_SHA

    chain_path = _chain_path_for_run(run_path)
    prev = _last_chain_hash(chain_path)
    digest = compute_hash(prev, sha_outer, sha_deep, outer["replay"])
    seal: dict[str, Any] = {
        "run_id": run_id,
        "hash": digest,
        "prev_hash": prev,
        "sha_outer": sha_outer,
        "sha_deep": sha_deep,
        "algorithm": ALGORITHM,
        "sealed_by": sealed_by,
        "sealed_at_iso": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(run_path, "seal.json"), "w", encoding="utf-8") as fh:
        json.dump(seal, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.makedirs(os.path.dirname(chain_path) or ".", exist_ok=True)
    with open(chain_path, "a", encoding="utf-8") as fh:
        fh.write(
            json.dumps({"run_id": run_id, "hash": digest, "prev_hash": prev}, sort_keys=True) + "\n"
        )
    return digest


def verify_run(run_id: str, store: str | None = None) -> bool:
    """Recompute the 5 seal steps; True=OK, False=FAIL.

    Raises:
        RunNotFoundError: E06 unknown run (missing run dir).
    """
    root = get_store_dir(store)
    path = run_dir(root, run_id)
    if not os.path.isdir(path):
        raise RunNotFoundError(f"E06: {run_id} not found in {root}/runs/.")
    outer_path = os.path.join(path, "outer.json")
    seal_path = os.path.join(path, "seal.json")
    if not os.path.isfile(outer_path) or not os.path.isfile(seal_path):
        return False
    with open(seal_path, encoding="utf-8") as fh:
        seal = json.load(fh)
    with open(outer_path, encoding="utf-8") as fh:
        outer = json.load(fh)
    if not isinstance(outer.get("replay"), dict):
        return False
    sha_outer = _sha_file(outer_path)
    deep_path = os.path.join(path, "deep.bin")
    sha_deep = _sha_file(deep_path) if os.path.isfile(deep_path) else EMPTY_SHA
    if sha_outer != seal.get("sha_outer") or sha_deep != seal.get("sha_deep"):
        return False
    recomputed = compute_hash(str(seal.get("prev_hash", "")), sha_outer, sha_deep, outer["replay"])
    if recomputed != seal.get("hash"):
        return False
    chain_path = os.path.join(root, "chain.jsonl")
    if not os.path.isfile(chain_path):
        return seal.get("prev_hash") == GENESIS_PREV
    prev_expected = GENESIS_PREV
    with open(chain_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                return False
            if entry.get("prev_hash") != prev_expected:
                return False
            prev_expected = str(entry.get("hash"))
            if entry.get("run_id") == run_id:
                return entry.get("hash") == seal.get("hash")
    return False
