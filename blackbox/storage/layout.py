"""Store layout: manifests + runs/{outer,deep,seal} + chain (spec §FILE-*).

Run IDs match ``RUN-[0-9]{6}``, incrementing from ``chain.jsonl`` (or
``000001``). Helpers are importable for CLI use.

Deep files (spec §FILE-deepbin / §FILE-deepindex, plan.md §9.5):

- ``deep.bin``: COLD, optional (absent when ``outer_only`` or the trigger
  never fired). Raw little-endian floats, row-major ``float[T][W]``
  (token-major: token ``t``, watched index ``j`` at byte ``(t*W + j) * B``).
  ``T`` = emitted tokens after ``every_n`` subsample, ``W`` =
  ``len(watched)`` in ``deep_index.json`` order. ``fp32`` B=4, ``fp16``/``bf16``
  B=2. Packing uses numpy only (no CUDA import); ``bf16`` without the optional
  ``ml_dtypes`` package falls back to fp16 bytes with a ``dtype_meta`` note.
- ``deep_index.json``: HOT pointer (``triggered, dtype, tokens, every_n,
  watched, columns{addr: {offset, dtype, shape}}, dtype_meta``).
  Absent-Deep case: ``{"triggered": false, ...}`` and no ``deep.bin``.

Exit mapping: E02 ``ValueError`` (bad RUN-ID / bad payload / bad slice).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import numpy as np

RUN_RE = re.compile(r"^RUN-(\d{6})$")

DTYPE_BYTES: dict[str, int] = {"fp32": 4, "fp16": 2, "bf16": 2}
_LE_DTYPE: dict[str, str] = {"fp32": "<f4", "fp16": "<f2", "bf16": "<f2"}


def get_store_dir(store: str | None = None) -> str:
    """Resolve store root: arg > ``$BLACKBOX_DIR`` > ``./blackbox_store``."""
    return store or os.environ.get("BLACKBOX_DIR", "./blackbox_store")


def run_dir(store: str, run_id: str) -> str:
    """Resolve ``$BLACKBOX_DIR/runs/RUN-xxx`` path (validates RUN-ID)."""
    if not RUN_RE.match(run_id):
        raise ValueError(f"E02: bad run id {run_id!r}")
    return os.path.join(store, "runs", run_id)


def _max_run_number(store: str) -> int:
    """Max sealed/allocated run number from chain + run dirs (0 if none)."""
    best = 0
    chain = os.path.join(store, "chain.jsonl")
    if os.path.isfile(chain):
        with open(chain, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    m = RUN_RE.match(str(obj.get("run_id", "")))
                    if m:
                        best = max(best, int(m.group(1)))
                except (ValueError, AttributeError):
                    continue
    runs_root = os.path.join(store, "runs")
    if os.path.isdir(runs_root):
        for name in os.listdir(runs_root):
            m = RUN_RE.match(name)
            if m:
                best = max(best, int(m.group(1)))
    return best


def next_run_id(store: str | None = None) -> str:
    """Next RUN-ID without allocating (``000001`` when store empty)."""
    root = get_store_dir(store)
    return f"RUN-{_max_run_number(root) + 1:06d}"


def allocate_run(store: str | None = None) -> tuple[str, str]:
    """Allocate next RUN-ID and ensure its directory; return ``(id, path)``."""
    root = get_store_dir(store)
    rid = next_run_id(root)
    path = run_dir(root, rid)
    os.makedirs(path, exist_ok=True)
    return (rid, path)


def write_outer(run_path: str, payload: bytes) -> str:
    """Write ``outer.json`` hot file; return its path.

    Raises:
        ValueError: E02 bad RUN-ID dir or non-JSON payload.
    """
    base = os.path.basename(os.path.normpath(run_path))
    if not RUN_RE.match(base):
        raise ValueError(f"E02: bad run path {run_path!r}")
    try:
        json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"E02: outer payload must be JSON: {exc}") from exc
    os.makedirs(run_path, exist_ok=True)
    out = os.path.join(run_path, "outer.json")
    with open(out, "wb") as fh:
        fh.write(payload)
    return out


def write_outer_json(run_path: str, obj: dict[str, Any]) -> str:
    """Write ``obj`` as ``outer.json`` (sorted keys); return its path."""
    return write_outer(run_path, json.dumps(obj, indent=2, sort_keys=True).encode("utf-8"))


def precision_bytes(precision: str) -> int:
    """Bytes per value for a logical precision (E02 on unknown)."""
    try:
        return DTYPE_BYTES[precision]
    except KeyError as exc:
        raise ValueError(f"E02: bad precision {precision!r}; expected fp32|fp16|bf16") from exc


def expected_deep_bytes(w: int, t: int, precision: str) -> int:
    """Per-run ``deep.bin`` bytes ``W*T*B`` (estimator adds ``R`` + 1.15)."""
    return int(w) * int(t) * precision_bytes(precision)


def _as_le_matrix(matrix: np.ndarray, precision: str) -> np.ndarray:
    """Cast to a 2D little-endian array for ``precision`` (E02 on bad shape)."""
    arr = np.asarray(matrix)
    if arr.ndim != 2:
        raise ValueError(f"E02: deep matrix must be 2D [T, W], got ndim={arr.ndim}")
    precision_bytes(precision)  # validate name early
    if precision == "bf16":
        try:
            import ml_dtypes  # type: ignore[import-untyped]

            return np.asarray(arr, dtype=ml_dtypes.bfloat16)
        except ImportError:
            pass  # CPU fallback: fp16 bytes, noted in dtype_meta
    return np.asarray(arr, dtype=np.dtype(_LE_DTYPE[precision]))


def write_deep_bin(run_path: str, matrix: np.ndarray, precision: str = "fp16") -> str:
    """Write row-major ``[T, W]`` LE ``deep.bin`` via numpy; return its path.

    Raises:
        ValueError: E02 bad RUN-ID dir, bad precision, or non-2D matrix.
    """
    base = os.path.basename(os.path.normpath(run_path))
    if not RUN_RE.match(base):
        raise ValueError(f"E02: bad run path {run_path!r}")
    arr = _as_le_matrix(matrix, precision)
    os.makedirs(run_path, exist_ok=True)
    out = os.path.join(run_path, "deep.bin")
    with open(out, "wb") as fh:
        fh.write(arr.tobytes(order="C"))
    return out


def write_deep_index(
    run_path: str,
    run_id: str,
    *,
    triggered: bool,
    dtype: str = "fp16",
    tokens: int = 0,
    every_n: int = 1,
    watched: list[str] | None = None,
    window: dict[str, Any] | None = None,
    dtype_meta: dict[str, Any] | None = None,
) -> str:
    """Write ``deep_index.json`` HOT pointer; return its path.

    Absent-Deep case (``triggered=False``): no ``deep.bin`` is expected;
    ``watched`` defaults to ``[]`` and ``columns`` to ``{}``.
    """
    base = os.path.basename(os.path.normpath(run_path))
    if not RUN_RE.match(base):
        raise ValueError(f"E02: bad run path {run_path!r}")
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"E02: bad dtype {dtype!r}; expected fp32|fp16|bf16")
    if every_n < 1:
        raise ValueError(f"E02: every_n must be >= 1 (got {every_n})")
    names = list(watched) if watched else []
    columns: dict[str, dict[str, Any]] = {}
    if triggered:
        for offset, addr in enumerate(names):
            columns[addr] = {"offset": offset, "dtype": dtype, "shape": [int(tokens)]}
    meta: dict[str, Any] = {
        "stored": dtype,
        "source": "cpu_fallback_fp16_bytes" if dtype == "bf16" else "logical_values",
        "endian": "little",
        "order": "row_major_token_major",
    }
    if dtype_meta:
        meta.update(dtype_meta)
    index: dict[str, Any] = {
        "run_id": run_id,
        "triggered": bool(triggered),
        "dtype": dtype,
        "tokens": int(tokens),
        "every_n": int(every_n),
        "window": dict(window) if window else None,
        "watched": names,
        "columns": columns,
        "dtype_meta": meta,
    }
    os.makedirs(run_path, exist_ok=True)
    out = os.path.join(run_path, "deep_index.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return out


def write_absent_deep_index(
    run_path: str,
    run_id: str,
    *,
    dtype: str = "fp16",
    every_n: int = 1,
    watched: list[str] | None = None,
) -> str:
    """Write the absent-Deep index (``triggered=false``, no ``deep.bin``)."""
    return write_deep_index(
        run_path,
        run_id,
        triggered=False,
        dtype=dtype,
        tokens=0,
        every_n=every_n,
        watched=watched or [],
    )


def read_deep_index(run_path: str) -> dict[str, Any]:
    """Load ``deep_index.json`` (E02 when missing/not a dict)."""
    path = os.path.join(run_path, "deep_index.json")
    if not os.path.isfile(path):
        raise ValueError(f"E02: no deep_index.json in {run_path}")
    with open(path, encoding="utf-8") as fh:
        index = json.load(fh)
    if not isinstance(index, dict):
        raise ValueError(f"E02: bad deep_index.json in {run_path}")
    return index


def read_deep_slice(
    run_path: str,
    token_lo: int,
    token_hi: int,
    watched_idx: list[int] | None = None,
) -> np.ndarray:
    """Mmap the slice ``deep.bin[token_lo..token_hi, watched_idx]`` (incl).

    Returns float32 logical values with shape ``[slice_T, len(watched_idx)]``
    (all columns when ``watched_idx`` is None). Only the slice is mapped;
    the viewer MUST NOT use this to embed a full ``deep.bin``.

    Raises:
        ValueError: E02 missing Deep, bad range, or bad column index.
    """
    index = read_deep_index(run_path)
    if not index.get("triggered"):
        raise ValueError(f"E02: no Deep for run {index.get('run_id')}: triggered=false")
    watched = list(index.get("watched", []))
    tokens = int(index.get("tokens", 0))
    dtype = str(index.get("dtype", "fp16"))
    if token_lo < 0 or token_hi < token_lo or token_hi >= tokens:
        raise ValueError(f"E02: bad token slice {token_lo}..{token_hi} for T={tokens}")
    cols = list(range(len(watched))) if watched_idx is None else list(watched_idx)
    for j in cols:
        if j < 0 or j >= len(watched):
            raise ValueError(f"E02: bad watched index {j} for W={len(watched)}")
    deep_path = os.path.join(run_path, "deep.bin")
    if not os.path.isfile(deep_path):
        raise ValueError(f"E02: deep_index triggered but no deep.bin in {run_path}")
    precision_bytes(dtype)  # validate name early
    raw_dt: Any = np.dtype(_LE_DTYPE[dtype])
    if dtype == "bf16":
        try:
            import ml_dtypes  # type: ignore[import-untyped]

            raw_dt = np.dtype(ml_dtypes.bfloat16)
        except ImportError:
            pass  # stored as fp16 bytes on the CPU fallback path
    full = np.memmap(deep_path, dtype=raw_dt, mode="r")
    w = len(watched)
    expect = tokens * w
    if full.size < expect:
        raise ValueError(f"E02: deep.bin short: {full.size} < T*W={expect}")
    view = np.asarray(full[:expect].reshape(tokens, w))
    part = view[token_lo : token_hi + 1, :][:, cols]
    return np.asarray(part, dtype=np.float32)
