"""Dataframe view over sealed runs (spec section A.12).

``what="outer"`` returns 1 row; ``what="deep"`` returns long-form
tokens x watched rows from ``deep.bin`` via ``deep_index.json`` offsets
(slice only, never the full file); ``"outer+deep"`` broadcasts outer cols
onto the deep rows. Partial-precise: unwatched addresses are omitted unless
explicitly requested (then ``watched=False, value=NaN``); values are float32
logical. Broken seal sets ``seal_ok=False`` with a ``SealBrokenWarning``.

``pandas`` is optional: when missing, a list-of-dicts fallback with the
same columns is returned.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Any

OUTER_COLUMNS: list[str] = [
    "run_id",
    "token_pos",
    "token",
    "logit",
    "address",
    "value",
    "watched",
    "seal_ok",
]


class SealBrokenWarning(UserWarning):
    """Seal failed verification; rows still returned with ``seal_ok=False``."""


def _store_root() -> str:
    return os.environ.get("BLACKBOX_DIR", "./blackbox_store")


def _verify_seal_ok(run_id: str, store: str) -> bool:
    """Best-effort seal check; ``False`` on any problem (never raises)."""
    try:
        from blackbox.storage.seal import verify_run  # owned by another worker

        result = verify_run(run_id, store)
        return bool(result)
    except Exception:
        pass
    try:
        seal_path = os.path.join(store, "runs", run_id, "seal.json")
        with open(seal_path, encoding="utf-8") as fh:
            seal = json.load(fh)
        outer_path = os.path.join(store, "runs", run_id, "outer.json")
        with open(outer_path, "rb") as fh:
            import hashlib

            sha_outer = hashlib.sha256(fh.read()).hexdigest()
        return bool(seal.get("hash")) and seal.get("sha_outer") == sha_outer
    except Exception:
        return False


def _find_manifest(model_id: str, sha: str) -> dict[str, Any] | None:
    """Best-effort manifest lookup for address validation (None if absent)."""
    import glob

    for path in glob.glob(os.path.join("manifests", "*", "manifest.json")):
        try:
            with open(path, encoding="utf-8") as fh:
                man = json.load(fh)
            if man.get("model_id") == model_id and man.get("checkpoint_sha256") == sha:
                return man
        except (OSError, ValueError):
            continue
    return None


def _validate_against_manifest(requested: list[str], model_id: str, sha: str) -> None:
    """Validate requested addresses vs manifest when one is on disk (E04)."""
    man = _find_manifest(model_id, sha)
    if man is None:
        return
    try:
        from blackbox.watchlist import validate_addresses
    except ImportError:
        return
    validate_addresses(requested, man)


def _frame(rows: list[dict[str, Any]]) -> Any:
    """Build the pandas frame (or list-of-dicts fallback) with dtypes."""
    try:
        import pandas as pd  # optional heavy dep

        df = pd.DataFrame(rows, columns=OUTER_COLUMNS)
        df = df.astype(
            {
                "run_id": "string",
                "token_pos": "int32",
                "token": "string",
                "logit": "float32",
                "address": "string",
                "value": "float32",
                "watched": "bool",
                "seal_ok": "bool",
            }
        )
        return df
    except ImportError:
        return rows


def _outer_texts(outer: dict[str, Any]) -> tuple[list[str], str]:
    """Token texts for rows: output tokens preferred, else prompt text."""
    output = outer.get("output", {})
    ids = output.get("tokens", []) if isinstance(output, dict) else []
    texts = [str(t) for t in ids] if isinstance(ids, list) else []
    prompt = outer.get("prompt", {})
    prompt_text = str(prompt.get("text", "")) if isinstance(prompt, dict) else ""
    return texts, prompt_text


def _deep_rows(
    run_id: str,
    run_path: str,
    outer: dict[str, Any],
    tokens: tuple[int, int] | None,
    addresses: list[str] | None,
    seal_ok: bool,
    broadcast_outer: bool,
) -> list[dict[str, Any]]:
    """Long-form tokens x watched rows from the Cold slice (never full file)."""
    from blackbox.storage.layout import read_deep_index, read_deep_slice

    index = read_deep_index(run_path)
    if not index.get("triggered"):
        raise ValueError(f"E02: no Deep for run {run_id}: triggered=false")
    watched: list[str] = list(index.get("watched", []))
    total_tokens = int(index.get("tokens", 0))
    lo, hi = (0, total_tokens - 1) if tokens is None else (tokens[0], tokens[1])
    if lo < 0 or hi < lo or hi >= total_tokens:
        raise ValueError(f"E02: bad tokens slice {(lo, hi)} for T={total_tokens}")
    if addresses is None:
        cols = list(range(len(watched)))
        wanted = list(watched)
    else:
        _validate_against_manifest(
            addresses,
            str(outer.get("model_id", "")),
            str(outer.get("checkpoint_sha256", "")),
        )
        wanted = list(addresses)
        cols = [watched.index(a) if a in watched else -1 for a in wanted]
    present = [c for c in cols if c >= 0]
    values = read_deep_slice(run_path, lo, hi, present if present else [])
    col_of = {c: k for k, c in enumerate(present)}
    texts, prompt_text = _outer_texts(outer)
    rows: list[dict[str, Any]] = []
    for row_i, pos in enumerate(range(lo, hi + 1)):
        text = texts[pos] if pos < len(texts) else prompt_text
        for addr, col in zip(wanted, cols):
            if col < 0:
                rows.append(
                    {
                        "run_id": run_id,
                        "token_pos": pos,
                        "token": text,
                        "logit": float("nan"),
                        "address": addr,
                        "value": float("nan"),
                        "watched": False,
                        "seal_ok": seal_ok,
                    }
                )
            else:
                rows.append(
                    {
                        "run_id": run_id,
                        "token_pos": pos,
                        "token": text,
                        "logit": float("nan"),
                        "address": addr,
                        "value": float(values[row_i, col_of[col]]),
                        "watched": True,
                        "seal_ok": seal_ok,
                    }
                )
    _ = broadcast_outer
    return rows


def to_dataframe(
    run_id: str,
    what: str = "outer+deep",
    tokens: tuple[int, int] | None = None,
    addresses: list[str] | None = None,
) -> Any:
    """Frame over a sealed run (spec §API-dataframe).

    ``what="outer"`` returns 1 row; ``"deep"`` returns long-form
    tokens x watched rows; ``"outer+deep"`` returns the deep rows (outer
    cols are identical in every row, so no extra broadcast column is
    needed). Broken seal warns (``SealBrokenWarning``) and sets
    ``seal_ok=False`` — rows are never silently dropped.

    Raises:
        ValueError: bad ``what`` or bad ``tokens`` slice.
        RunNotFoundError: when the run directory is missing.
    """
    if what not in ("outer", "deep", "outer+deep"):
        raise ValueError(f"E02: unknown what={what!r}; expected 'outer'|'deep'|'outer+deep'")
    store = _store_root()
    run_path = os.path.join(store, "runs", run_id)
    outer_path = os.path.join(run_path, "outer.json")
    if not os.path.isfile(outer_path):
        try:
            from blackbox import RunNotFoundError as _RNF
        except Exception:  # pragma: no cover - import cycle guard

            class _RNF(Exception):  # type: ignore[no-redef]
                pass

        raise _RNF(f"run {run_id} not found in {store}/runs/.")
    with open(outer_path, encoding="utf-8") as fh:
        outer = json.load(fh)
    seal_ok = _verify_seal_ok(run_id, store)
    if not seal_ok:
        warnings.warn(f"seal broken for {run_id}; seal_ok=False", SealBrokenWarning, stacklevel=2)
    if what == "outer":
        prompt = outer.get("prompt", {}) if isinstance(outer, dict) else {}
        token_text = str(prompt.get("text", "")) if isinstance(prompt, dict) else ""
        address = str(addresses[0]) if addresses else ""
        return _frame(
            [
                {
                    "run_id": run_id,
                    "token_pos": 0,
                    "token": token_text,
                    "logit": float("nan"),
                    "address": address,
                    "value": float("nan"),
                    "watched": False,
                    "seal_ok": bool(seal_ok),
                }
            ]
        )
    rows = _deep_rows(
        run_id, run_path, outer, tokens, addresses, bool(seal_ok), what == "outer+deep"
    )
    return _frame(rows)
