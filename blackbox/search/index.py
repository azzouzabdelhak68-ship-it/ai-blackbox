"""Hot index shards with per-address aggregates (spec §SEARCH-tiers, plan §10.1).

Tier 1 Hot lives in ``$BLACKBOX_DIR/index/shard-*.jsonl`` (one JSON object
per sealed run) plus a per-run copy at ``runs/<RUN>/meta.json``. Each entry
carries conservative aggregates so the Hot filter never drops a true
positive; Cold exact values always decide:

- ``min/max/mean/p95`` over all watched tokens of the run,
- ``firing_rate_0.8`` = fraction of values ``> 0.8``,
- ``win10`` = maxima of non-overlapping 10-token chunks (capped at the
  first 128 chunks; ``n_win10`` records the total), used to pre-filter
  token-range predicates without opening Cold.

Columnar rule (spec): the Day-1 default is JSONL shards (works everywhere,
no server). A community ``search-adapters/`` backend may replace the two
functions :func:`hot_scan` and :func:`cold_verify_one` as long as it keeps
the contract — Hot returns a candidate superset, Cold decides exact. Naive
one-row-per-token-per-neuron SQL tables are rejected (5B rows for
10k x 500 x 1k); do not implement that shape. DuckDB/Parquet stays optional
and is not required for any gate.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import os
from collections.abc import Callable
from typing import Any

SHARD_SIZE = 10000
WIN = 10
WIN_CAP = 128
FIRING_THRESHOLD = 0.8


def _store_root(store: str | None = None) -> str:
    return store or os.environ.get("BLACKBOX_DIR", "./blackbox_store")


def _index_dir(store: str) -> str:
    path = os.path.join(store, "index")
    os.makedirs(path, exist_ok=True)
    return path


def _shard_for(path: str, run_id: str) -> str:
    """Current open shard; rolls to a new shard every SHARD_SIZE lines."""
    idx = 0
    while True:
        cand = os.path.join(path, f"shard-{idx:04d}.jsonl")
        if not os.path.isfile(cand):
            return cand
        try:
            with open(cand, encoding="utf-8") as fh:
                n = sum(1 for _ in fh)
        except OSError:
            return cand
        if n < SHARD_SIZE:
            return cand
        idx += 1
    _ = run_id
    return cand


def _load_matrix(run_dir: str, dtype: str, tokens: int, nwatched: int) -> Any:
    """Memory-map ``deep.bin`` row-major ``[T, W]`` (never fully read eagerly)."""
    import numpy as np

    raw: Any = {"fp32": np.dtype("<f4"), "fp16": np.dtype("<f2"), "bf16": np.dtype("<f2")}[dtype]
    deep_path = os.path.join(run_dir, "deep.bin")
    size = os.path.getsize(deep_path)
    if size != tokens * nwatched * int(raw.itemsize):
        raise ValueError(f"E02: deep.bin size {size} != [{tokens}x{nwatched}] {dtype}")
    return np.memmap(deep_path, dtype=raw, mode="r", shape=(tokens, nwatched))


def _addr_stats(col: Any) -> dict[str, Any]:
    """Conservative aggregates for one watched column (NaN-aware)."""
    import numpy as np

    vals = np.asarray(col, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "p95": None,
            "firing_rate_0.8": 0.0,
            "win10": [],
            "n_win10": 0,
        }
    chunks = int(math.ceil(vals.size / WIN))
    win10 = [float(vals[i * WIN : (i + 1) * WIN].max()) for i in range(min(chunks, WIN_CAP))]
    return {
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
        "p95": float(np.percentile(vals, 95)),
        "firing_rate_0.8": float((vals > FIRING_THRESHOLD).mean()),
        "win10": win10,
        "n_win10": chunks,
    }


def build_hot_entry(run_id: str, store: str | None = None) -> dict[str, Any]:
    """Compute the Hot entry for a sealed run (raises E02/E06 on bad runs)."""
    root = _store_root(store)
    run_dir = os.path.join(root, "runs", run_id)
    outer_path = os.path.join(run_dir, "outer.json")
    if not os.path.isfile(outer_path):
        raise FileNotFoundError(f"E06: {run_id} not found in {root}/runs/.")
    with open(outer_path, encoding="utf-8") as fh:
        outer = json.load(fh)
    if not isinstance(outer, dict):
        raise ValueError(f"E02: bad outer.json for {run_id}")
    entry: dict[str, Any] = {
        "run_id": run_id,
        "model_id": outer.get("model_id", ""),
        "checkpoint_sha256": outer.get("checkpoint_sha256", ""),
        "watchlist_id": (outer.get("watchlist_ref", {}) or {}).get("watchlist_id", ""),
        "tags": list(outer.get("tags", []) or []),
        "flags": dict(outer.get("flags", {}) or {}),
        "prompt_prefix": str((outer.get("prompt", {}) or {}).get("text", ""))[:200],
        "has_deep": False,
        "addresses": {},
    }
    index_path = os.path.join(run_dir, "deep_index.json")
    if not os.path.isfile(index_path):
        return entry
    with open(index_path, encoding="utf-8") as fh:
        index = json.load(fh)
    if not isinstance(index, dict) or not index.get("triggered"):
        return entry
    watched: list[str] = list(index.get("watched", []))
    tokens = int(index.get("tokens", 0))
    dtype = str(index.get("dtype", "fp16"))
    if not watched or tokens < 1:
        return entry
    try:
        mat = _load_matrix(run_dir, dtype, tokens, len(watched))
    except (OSError, ValueError, KeyError):
        return entry
    entry["has_deep"] = True
    entry["dtype"] = dtype
    entry["tokens"] = tokens
    entry["addresses"] = {addr: _addr_stats(mat[:, j]) for j, addr in enumerate(watched)}
    return entry


def append_hot(run_id: str, store: str | None = None) -> dict[str, Any]:
    """Build + persist the Hot entry (shard line + ``meta.json``); idempotent."""
    root = _store_root(store)
    entry = build_hot_entry(run_id, root)
    path = _index_dir(root)
    shard = _shard_for(path, run_id)
    with open(shard, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    meta_path = os.path.join(root, "runs", run_id, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(entry, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return entry


def ensure_hot(run_id: str, store: str | None = None) -> dict[str, Any]:
    """Return the Hot entry, building it lazily when missing (never E06-lossy)."""
    root = _store_root(store)
    meta_path = os.path.join(root, "runs", run_id, "meta.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as fh:
                entry = json.load(fh)
            if isinstance(entry, dict) and entry.get("run_id") == run_id:
                return entry
        except (OSError, ValueError):
            pass
    return append_hot(run_id, root)


def iter_hot_entries(store: str | None = None) -> list[dict[str, Any]]:
    """All Hot entries: shard lines first, then runs missing from shards."""
    root = _store_root(store)
    entries: dict[str, dict[str, Any]] = {}
    path = os.path.join(root, "index")
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if not name.startswith("shard-") or not name.endswith(".jsonl"):
                continue
            try:
                with open(os.path.join(path, name), encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except ValueError:
                            continue
                        if isinstance(obj, dict) and obj.get("run_id"):
                            entries[str(obj["run_id"])] = obj
            except OSError:
                continue
    runs_dir = os.path.join(root, "runs")
    if os.path.isdir(runs_dir):
        import re

        for name in sorted(os.listdir(runs_dir)):
            if re.fullmatch(r"RUN-\d{6}", name) and name not in entries:
                try:
                    entries[name] = ensure_hot(name, root)
                except (OSError, ValueError):
                    continue
    return [entries[k] for k in sorted(entries)]


def hot_scan(
    predicate: Callable[[dict[str, Any]], bool],
    store: str | None = None,
    workers: int = 4,
) -> list[dict[str, Any]]:
    """Parallel Hot filter; predicate sees entries only (no Cold I/O)."""
    entries = iter_hot_entries(store)
    if workers < 2 or len(entries) < 2:
        return [e for e in entries if predicate(e)]
    out: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for entry, keep in zip(entries, pool.map(predicate, entries)):
            if keep:
                out.append(entry)
    return out


def cold_verify_one(run_id: str, check: Callable[[], bool], store: str | None = None) -> bool:
    """Run one Cold exact check; any exception counts as non-match (never raise)."""
    _ = (run_id, store)
    try:
        return bool(check())
    except Exception:
        return False


def reindex(store: str | None = None) -> int:
    """Rebuild shards from all sealed runs; returns entry count."""
    root = _store_root(store)
    path = _index_dir(root)
    for name in os.listdir(path):
        if name.startswith("shard-") and name.endswith(".jsonl"):
            try:
                os.remove(os.path.join(path, name))
            except OSError:
                pass
    count = 0
    runs_dir = os.path.join(root, "runs")
    if os.path.isdir(runs_dir):
        import re

        for name in sorted(os.listdir(runs_dir)):
            if re.fullmatch(r"RUN-\d{6}", name):
                try:
                    append_hot(name, root)
                    count += 1
                except (OSError, ValueError):
                    continue
    return count
