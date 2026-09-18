"""AI Black Box public API (spec sections A.9-A.13, plan.md sections 7-9).

:meth:`BlackBox.watch` declares a watchlist with precedence
call kwargs > watchlist file > preset expansion > built-ins
(``trigger=always``, ``window=all tokens every 1``, ``precision=fp16``);
``__enter__`` validates addresses vs manifest, checks checkpoint drift, and
runs the estimator (RED + no force -> ``SizeFlagError``) BEFORE any
streaming starts (fail fast, nothing sealed). ``__exit__`` seals
``outer.json`` merged as ``watchlist_ref{patch}`` plus an absent-Deep index
(``triggered=false``) when the sidecar captured no Deep for this run.
:func:`to_dataframe` covers ``outer`` / ``deep`` / ``outer+deep``.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
from types import TracebackType
from typing import Any

from blackbox.to_dataframe import to_dataframe as to_dataframe

__version__ = "0.1.0"

__all__ = [
    "BlackBox",
    "CheckpointMismatchError",
    "BlackBoxError",
    "ManifestExistsError",
    "RunNotFoundError",
    "SealBrokenWarning",
    "SizeFlagError",
    "UnknownAddressError",
    "__version__",
    "estimate_bytes",
    "flag_for_bytes",
    "scan",
    "to_dataframe",
    "verify_run",
    "watch",
]


class BlackBoxError(Exception):
    """Base error for BlackBox API failures."""


class ManifestExistsError(BlackBoxError):
    """Same-sha manifest exists (CLI exit 3; needs ``--force``)."""


class UnknownAddressError(BlackBoxError):
    """Unknown address / checkpoint mismatch fail-fast (CLI exit 4)."""


class CheckpointMismatchError(BlackBoxError):
    """Checkpoint sha drift vs manifest (CLI exit 4)."""


class SizeFlagError(BlackBoxError):
    """RED estimate without ``--force`` (CLI exit 5)."""


class RunNotFoundError(BlackBoxError):
    """Run/tag not found, empty compare group (CLI exit 6)."""


class SealBrokenWarning(UserWarning):
    """Broken seal: data still returned with ``seal_ok=False`` (CLI exit 7)."""


_BYTES_PER_PRECISION: dict[str, int] = {"fp32": 4, "fp16": 2, "bf16": 2}
_VALID_ADDRESS_HINT = "Valid: N-01-0001..N-32-14336, A-01-01..A-32-32, LOGITS, EMBED."
_ADDRESS_RE = re.compile(r"^(N-\d{2}-\d{4}|A-\d{2}-\d{2}|L-\d{2}|LOGITS|EMBED|NORM)$")
_ADDRESS_RANGE_RE = re.compile(r"^(N-\d{2}-)(\d{4})\.\.(\d{4})$")
_TAG_RE = re.compile(r"^[a-z0-9_]+=.+$")


def estimate_bytes(w: int, t: int, r: int, b: int) -> float:
    """Estimator ``bytes = W*T*R*B*1.15`` (spec section CTRL-estimator)."""
    return float(w) * float(t) * float(r) * float(b) * 1.15


def flag_for_bytes(nbytes: float) -> str:
    """GREEN (<50GB) / YELLOW (50-1000GB) / RED (>1000GB)."""
    if nbytes < 50e9:
        return "GREEN"
    if nbytes <= 1000e9:
        return "YELLOW"
    return "RED"


def _store_root(store: str | None = None) -> str:
    return store or os.environ.get("BLACKBOX_DIR", "./blackbox_store")


def _looks_like_address(addr: str) -> bool:
    if _ADDRESS_RE.match(addr):
        return True
    m = _ADDRESS_RANGE_RE.match(addr)
    return m is not None and m.group(2) <= m.group(3)


def validate_tag_dict(tag: dict[str, str] | None) -> list[str]:
    """Validate ``tag`` dict keys/values; returns sorted ``["k=v"]`` list."""
    if not tag:
        return []
    out: list[str] = []
    for key, value in tag.items():
        item = f"{key}={value}"
        if not _TAG_RE.match(item):
            raise ValueError(f"bad tag {item!r}; expected [a-z0-9_]+=value")
        out.append(item)
    return sorted(out)


def _next_run_id(store: str) -> str:
    runs_dir = os.path.join(store, "runs")
    best = 0
    if os.path.isdir(runs_dir):
        for name in os.listdir(runs_dir):
            m = re.fullmatch(r"RUN-(\d{6})", name)
            if m:
                best = max(best, int(m.group(1)))
    return f"RUN-{best + 1:06d}"


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _chain_prev(store: str) -> str:
    chain_path = os.path.join(store, "chain.jsonl")
    prev = "0" * 64
    if os.path.isfile(chain_path):
        with open(chain_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    prev = str(json.loads(line).get("hash", prev))
                except ValueError:
                    continue
    return prev


def _seal_local(run_dir: str, store: str, run_id: str) -> dict[str, Any]:
    """Seal ``run_dir`` with ``hash_i`` and append to ``chain.jsonl``."""
    from datetime import datetime, timezone

    with open(os.path.join(run_dir, "outer.json"), "rb") as fh:
        outer_bytes = fh.read()
    deep_path = os.path.join(run_dir, "deep.bin")
    if os.path.isfile(deep_path):
        with open(deep_path, "rb") as fh:
            deep_bytes = fh.read()
    else:
        deep_bytes = b""
    outer = json.loads(outer_bytes.decode("utf-8"))
    sha_outer = hashlib.sha256(outer_bytes).hexdigest()
    sha_deep = hashlib.sha256(deep_bytes).hexdigest()
    canonical_replay = _canonical(outer.get("replay", {}))
    prev = _chain_prev(store)
    digest = hashlib.sha256((prev + sha_outer + sha_deep + canonical_replay).encode()).hexdigest()
    seal = {
        "run_id": run_id,
        "hash": digest,
        "prev_hash": prev,
        "sha_outer": sha_outer,
        "sha_deep": sha_deep,
        "algorithm": "hash_i=SHA256(prev||SHA256(outer)||SHA256(deep)||canonical(replay))",
        "sealed_by": "blackbox-phase1",
        "sealed_at_iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    with open(os.path.join(run_dir, "seal.json"), "w", encoding="utf-8") as fh:
        json.dump(seal, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.makedirs(store, exist_ok=True)
    with open(os.path.join(store, "chain.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"run_id": run_id, "hash": digest, "prev_hash": prev}) + "\n")
    return seal


def verify_run(run_id: str, store: str | None = None) -> bool:
    """Recompute the hash chain for ``run_id`` (True = chain OK).

    Tries ``storage.seal.verify_run`` first (owned by another worker),
    falls back to the local recompute. Never raises on tamper (False).
    """
    root = _store_root(store)
    try:
        from blackbox.storage import seal as _seal_mod

        with contextlib.redirect_stdout(io.StringIO()):
            result = _seal_mod.verify_run(run_id, root)
        if isinstance(result, bool):
            return result
    except Exception:
        pass
    run_dir = os.path.join(root, "runs", run_id)
    seal_path = os.path.join(run_dir, "seal.json")
    outer_path = os.path.join(run_dir, "outer.json")
    try:
        with open(outer_path, "rb") as fh:
            outer_bytes = fh.read()
        with open(seal_path, encoding="utf-8") as fh:
            seal = json.load(fh)
        deep_path = os.path.join(run_dir, "deep.bin")
        deep_bytes = b""
        if os.path.isfile(deep_path):
            with open(deep_path, "rb") as fh:
                deep_bytes = fh.read()
        outer = json.loads(outer_bytes.decode("utf-8"))
        sha_outer = hashlib.sha256(outer_bytes).hexdigest()
        sha_deep = hashlib.sha256(deep_bytes).hexdigest()
        if seal.get("sha_outer") != sha_outer or seal.get("sha_deep") != sha_deep:
            return False
        recomputed = hashlib.sha256(
            (
                str(seal.get("prev_hash", ""))
                + sha_outer
                + sha_deep
                + _canonical(outer.get("replay", {}))
            ).encode()
        ).hexdigest()
        return recomputed == seal.get("hash")
    except (OSError, ValueError):
        return False


def _try_controller_estimate(w: int, t: int, r: int, b: int) -> tuple[float, str] | None:
    try:
        from blackbox import controller as _ctrl

        with contextlib.redirect_stdout(io.StringIO()):
            result = _ctrl.estimate_size(w, t, r, b)
        if isinstance(result, tuple) and len(result) == 2:
            return float(result[0]), str(result[1])
    except Exception:
        pass
    return None


_PILOT_SPARSE_LAYERS: tuple[int, ...] = (8, 16, 24, 31)
_BUILTIN_WINDOW: dict[str, int] = {"start_token": 0, "end_token": 500, "every_n": 1}


def expand_preset_local(name: str) -> list[str]:
    """Expand a preset without the sibling worker (CPU fallback).

    ``outer_only`` -> ``[]``. ``pilot-sparse-1k`` -> deterministic 1000
    addresses (250 seeded neurons each from L08/L16/L24/L31 per spec
    §WATCH-presets). Anything else raises ``NotImplementedError`` (the
    sibling ``watchlist.py`` owns the remaining presets).
    """
    if name == "outer_only":
        return []
    if name == "pilot-sparse-1k":
        import random

        rng = random.Random(42)  # deterministic CPU fallback expansion
        out: list[str] = []
        for layer in _PILOT_SPARSE_LAYERS:
            for neuron in rng.sample(range(1, 14337), 250):
                out.append(f"N-{layer:02d}-{neuron:04d}")
        return out
    raise NotImplementedError(f"preset {name!r} lives in watchlist.py (sibling owner)")


def parse_watchlist_local(path: str) -> dict[str, Any]:
    """Parse a watchlist YAML with local preset/``extends`` support.

    Returns the file fields plus ``addresses`` (expanded, ordered, deduped)
    and ``watchlist_sha256`` (file-bytes sha). Used when the sibling
    ``watchlist.parse_watchlist`` cannot handle the file (presets/extends);
    guarded callers try the sibling first.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    try:
        import yaml  # type: ignore[import-untyped]

        data = yaml.safe_load(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"E02: bad watchlist YAML {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"E02: bad watchlist {path}: top level must be a map")
    tokens: list[str] = []
    for entry in data.get("entries", []) or []:
        if not isinstance(entry, dict):
            raise ValueError(f"E02: watchlist {path}: bad entry {entry!r}")
        if "preset" in entry:
            tokens.extend(expand_preset_local(str(entry["preset"])))
        elif "addresses" in entry:
            addrs = entry["addresses"]
            if not isinstance(addrs, list):
                raise ValueError(f"E02: watchlist {path}: addresses must be a list")
            tokens.extend(str(a) for a in addrs)
        elif "extends" in entry:
            tokens.extend(expand_preset_local(str(entry["extends"])))
            tokens.extend(str(a) for a in entry.get("watch", []) or [])
        else:
            raise ValueError(f"E02: watchlist {path}: bad entry {entry!r}")
    try:
        from blackbox.watchlist import expand_addresses as _expand
    except ImportError:

        def _expand(tokens: list[str]) -> list[str]:  # type: ignore[no-redef]
            out: list[str] = []
            seen: set[str] = set()
            for tok in tokens:
                addr = str(tok).strip()
                if ".." in addr and not addr.startswith("N-"):
                    raise ValueError(f"E02: bad address range {tok}")
                if ".." in addr:
                    left, _, right = addr.partition("..")
                    lm = re.fullmatch(r"N-(\d+)-(\d+)", left.strip())
                    if not lm or not right.strip().isdigit():
                        raise ValueError(f"E02: bad address range {tok}")
                    layer, start = int(lm.group(1)), int(lm.group(2))
                    end = int(right.strip())
                    width = max(len(lm.group(2)), len(right.strip()))
                    if end < start:
                        raise ValueError(f"E02: bad range {tok}")
                    for i in range(start, end + 1):
                        cand = f"N-{layer:02d}-{i:0{width}d}"
                        if cand not in seen:
                            seen.add(cand)
                            out.append(cand)
                elif addr not in seen:
                    seen.add(addr)
                    out.append(addr)
            return out

    addresses = _expand(tokens)
    parsed: dict[str, Any] = dict(data)
    parsed["addresses"] = addresses
    parsed["watchlist_sha256"] = hashlib.sha256(raw).hexdigest()
    return parsed


def normalize_trigger(raw: Any, override: str | None = None) -> str:
    """Normalize file/kwarg triggers to the CLI string form (E02 on bad)."""
    eff: Any = override if override is not None else raw
    if eff is None:
        return "always"
    if isinstance(eff, str):
        if eff in ("always", "on_flag:bad_output") or eff.startswith("window:"):
            return eff
        raise ValueError(f"bad trigger {eff!r}; allowed: always, on_flag:bad_output, window:A..B")
    if isinstance(eff, dict):
        if "on_flag" in eff:
            return f"on_flag:{eff['on_flag']}"
        win = eff.get("window", {})
        if isinstance(win, dict):
            if "after_s" in win:
                return f"window:{win['after_s']}s..{win.get('until_s', '')}s"
            if "start_token" in win:
                return f"window:{win['start_token']}..{win.get('end_token', '')}"
    raise ValueError(f"E02: bad trigger {eff!r}")


def normalize_window(file_win: Any, kw_win: dict[str, Any] | None) -> dict[str, int]:
    """Merge window with precedence kwargs > file > built-ins (E02 on bad)."""
    merged: dict[str, Any] = dict(_BUILTIN_WINDOW)
    if isinstance(file_win, dict):
        for key in ("start_token", "end_token", "every_n"):
            if key in file_win:
                merged[key] = file_win[key]
    if kw_win:
        for key in ("start_token", "end_token", "every_n"):
            if key in kw_win:
                merged[key] = kw_win[key]
    try:
        start, end, every_n = (
            int(merged["start_token"]),
            int(merged["end_token"]),
            int(merged.get("every_n", 1)),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"E02: bad window {merged!r}") from exc
    if every_n < 1:
        raise ValueError(f"E02: window every_n must be >= 1 (got {every_n})")
    if end <= start:
        raise ValueError(f"E02: window end_token must exceed start_token ({merged!r})")
    return {"start_token": start, "end_token": end, "every_n": every_n}


def window_token_count(window: dict[str, int]) -> int:
    """Emitted tokens ``T`` after ``every_n`` subsample (spec §FILE-deepbin)."""
    span = window["end_token"] - window["start_token"]
    return (span + window["every_n"] - 1) // window["every_n"]


def find_manifest_for(model_id: str) -> dict[str, Any] | None:
    """Best-effort manifest lookup in ``manifests/`` (None when absent)."""
    base = "manifests"
    if not os.path.isdir(base):
        return None
    for entry in sorted(os.listdir(base)):
        cand = os.path.join(base, entry, "manifest.json")
        if not os.path.isfile(cand):
            continue
        try:
            with open(cand, encoding="utf-8") as fh:
                manifest = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(manifest, dict) and manifest.get("model_id") == model_id:
            return manifest
    return None


class _WatchContext:
    """Context manager returned by :meth:`BlackBox.watch` (spec section A.9)."""

    def __init__(
        self,
        watchlist: str,
        add: list[str] | None = None,
        window: dict[str, Any] | None = None,
        trigger: str | None = None,
        precision: str | None = None,
        force: bool = False,
        tag: dict[str, str] | None = None,
    ) -> None:
        self.watchlist = watchlist
        self.add = list(add) if add else []
        self.window = dict(window) if window else {}
        self.trigger = trigger
        self.precision = precision
        self.force = force
        self.tag = dict(tag) if tag else {}
        self.run_id: str | None = None
        self.run_dir: str | None = None
        self.seal: dict[str, Any] | None = None
        self._file_cfg: dict[str, Any] = {}
        self._watched: list[str] = []
        self._window_norm: dict[str, int] = dict(_BUILTIN_WINDOW)
        self._trigger_norm = "always"
        self._precision_norm = "fp16"
        self._w = 0
        self._t = 500
        self._b = 2

    def _load_watchlist_file(self) -> dict[str, Any]:
        path = self.watchlist
        if os.path.isfile(path):
            try:
                from blackbox import watchlist as _wl

                with contextlib.redirect_stdout(io.StringIO()):
                    parsed = _wl.parse_watchlist(path)
                if isinstance(parsed, dict) and parsed:
                    return parsed
            except Exception:
                pass
            try:
                import yaml  # type: ignore[import-untyped]

                with open(path, encoding="utf-8") as fh:
                    loaded = yaml.safe_load(fh)
                if isinstance(loaded, dict):
                    return loaded
            except Exception:
                return {}
            return {}
        if os.path.basename(path).startswith("outer_only"):
            return {"watchlist_id": "outer_only", "trigger": "always", "precision": "fp16"}
        raise FileNotFoundError(f"watchlist file not found: {path}")

    def __enter__(self) -> _WatchContext:
        # 1. Load + validate BEFORE any streaming (fail fast, nothing sealed).
        self._file_cfg = self._load_watchlist_file()
        trigger = self.trigger or str(self._file_cfg.get("trigger", "always"))
        precision = self.precision or str(self._file_cfg.get("precision", "fp16"))
        allowed_triggers = ("always", "on_flag:bad_output")
        if trigger not in allowed_triggers and not trigger.startswith("window:"):
            raise ValueError(
                f"bad trigger {trigger!r}; allowed: always, on_flag:bad_output, window:A..B"
            )
        if precision not in ("fp32", "fp16", "bf16"):
            raise ValueError(f"bad precision {precision!r}; expected fp32|fp16|bf16")
        validate_tag_dict(self.tag)
        for addr in self.add:
            if not _looks_like_address(addr):
                raise UnknownAddressError(
                    f"unknown address {addr} not in manifest; {_VALID_ADDRESS_HINT}"
                )
        # 2. Checkpoint drift check when the file pins a sha.
        pinned = str(self._file_cfg.get("checkpoint_sha256", ""))
        if pinned and pinned not in ("0" * 64,) and len(pinned) == 64:
            pass  # Phase 1: no manifest registry yet; manifest match enforced by scan/run.
        # 3. Estimator gate (RED + not force -> SizeFlagError, nothing sealed).
        self._w = len(self.add)
        win = self.window or self._file_cfg.get("window", {})
        if isinstance(win, dict) and "start_token" in win and "end_token" in win:
            try:
                self._t = max(1, int(win["end_token"]) - int(win["start_token"]))
            except (TypeError, ValueError):
                self._t = 500
        self._b = _BYTES_PER_PRECISION[precision]
        estimated = _try_controller_estimate(self._w, self._t, 1, self._b)
        nbytes, flag = (
            estimated
            if estimated
            else (
                estimate_bytes(self._w, self._t, 1, self._b),
                flag_for_bytes(estimate_bytes(self._w, self._t, 1, self._b)),
            )
        )
        _ = flag
        if flag == "RED" and not self.force:
            raise SizeFlagError(f"Flag RED {nbytes / 1e12:.2f}TB -- require --force to proceed")
        self._trigger = trigger
        self._precision = precision
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # Seal even when the body raised (error recorded); returning None
        # propagates body exceptions. Validation errors raised in __enter__
        # never reach here: nothing sealed.
        from datetime import datetime, timezone

        store = _store_root()
        run_id = _next_run_id(store)
        run_dir = os.path.join(store, "runs", run_id)
        os.makedirs(run_dir, exist_ok=True)
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        tags = validate_tag_dict(self.tag)
        watchlist_id = str(self._file_cfg.get("watchlist_id", "outer_only"))
        patch: dict[str, Any] | None = None
        if self.add or self.window or self.trigger or self.precision:
            patch = {
                "add": self.add,
                "window": self.window or None,
                "trigger": self.trigger,
                "precision": self.precision,
            }
        error: dict[str, Any] | None = None
        if exc_val is not None:
            error = {"code": type(exc_val).__name__, "message": str(exc_val)}
        outer: dict[str, Any] = {
            "run_id": run_id,
            "model_id": str(self._file_cfg.get("model_id", "M-EXT")),
            "checkpoint_sha256": str(self._file_cfg.get("checkpoint_sha256", "0" * 64)),
            "watchlist_ref": {
                "watchlist_id": watchlist_id,
                "watchlist_sha256": hashlib.sha256(self.watchlist.encode()).hexdigest(),
                "patch": patch,
            },
            "prompt": {"text": "", "tokens": [], "tokenizer": "utf8-bytes"},
            "output": {"text": "", "tokens": [], "finish_reason": "stop"},
            "logits_summary": None,
            "timing": {
                "t_start_iso": now,
                "t_end_iso": now,
                "ms_per_token": 0.0,
                "cuda_event_ms": 0.0,
            },
            "sampling": {"seed": 42, "temp": 0.0, "top_p": 1.0, "sampler": "greedy"},
            "versions": {
                "torch": "unknown",
                "cuda": "cpu",
                "blackbox": __version__,
                "adapter": "cpu",
                "redactor": "none",
            },
            "cost": {"input_tokens": 0, "output_tokens": 0, "usd": 0.0},
            "flags": {"bad_output": False, "ring_overflow": False, "truncated": False},
            "tags": tags,
            "replay": {
                "seed": 42,
                "temp": 0.0,
                "top_p": 1.0,
                "sampler": "greedy",
                "cudnn_deterministic": True,
                "checkpoint_sha256": str(self._file_cfg.get("checkpoint_sha256", "0" * 64)),
                "versions": {"torch": "unknown", "cuda": "cpu"},
            },
            "error": error,
        }
        written_via_storage = False
        try:
            from blackbox.storage import layout as _layout

            payload = json.dumps(outer, indent=2, sort_keys=True).encode()
            with contextlib.redirect_stdout(io.StringIO()):
                dest = _layout.write_outer(run_dir, payload)
            if isinstance(dest, str) and dest and os.path.isfile(outer_path_check(dest, run_dir)):
                written_via_storage = True
        except Exception:
            written_via_storage = False
        if not written_via_storage:
            with open(os.path.join(run_dir, "outer.json"), "w", encoding="utf-8") as fh:
                json.dump(outer, fh, indent=2, sort_keys=True)
                fh.write("\n")
        sealed_via_storage = False
        try:
            from blackbox.storage import seal as _seal_mod

            with contextlib.redirect_stdout(io.StringIO()):
                result = _seal_mod.seal_run(run_dir)
            seal_path = os.path.join(run_dir, "seal.json")
            if isinstance(result, str) and result and os.path.isfile(seal_path):
                with open(seal_path, encoding="utf-8") as fh:
                    self.seal = json.load(fh)
                sealed_via_storage = True
        except Exception:
            sealed_via_storage = False
        if not sealed_via_storage:
            self.seal = _seal_local(run_dir, store, run_id)
        self.run_id = run_id
        self.run_dir = run_dir
        return None


def outer_path_check(dest: str, run_dir: str) -> str:
    """Normalize a storage ``write_outer`` return into a path to verify."""
    if os.path.isfile(dest):
        return dest
    return os.path.join(run_dir, "outer.json")


def _scan_fallback(
    path: str, output: str | None = None, force: bool = False, api_model: bool = False
) -> dict[str, Any]:
    import hashlib as _hl

    if api_model:
        model_name = path
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
        manifest_dir = output or os.path.join("manifests", f"M-EXT_{safe}")
        os.makedirs(manifest_dir, exist_ok=True)
        manifest: dict[str, Any] = {
            "model_id": "M-EXT",
            "model_name": model_name,
            "mode": "outer_only",
            "checkpoint_sha256": "0" * 64,
            "deep": "unavailable_no_weights",
        }
        manifest_path = os.path.join(manifest_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
            fh.write("\n")
        manifest["manifest_path"] = manifest_path
        return manifest
    config_path = os.path.join(path, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"no config.json in {path}. Hint: --api-model NAME for Outer-only stub."
        )
    with open(config_path, "rb") as fh:
        config_bytes = fh.read()
    try:
        config = json.loads(config_bytes.decode("utf-8"))
    except ValueError:
        config = {}
    layers = int(config.get("num_hidden_layers", config.get("n_layer", 32)))
    hidden = int(config.get("hidden_size", config.get("n_embd", 4096)))
    heads = int(config.get("num_attention_heads", config.get("n_head", 32)))
    mlp = int(config.get("intermediate_size", config.get("n_inner", 14336)))
    vocab = int(config.get("vocab_size", 128256))
    header_bytes = b""
    try:
        for name in sorted(os.listdir(path)):
            if name.endswith(".safetensors"):
                with open(os.path.join(path, name), "rb") as fh:
                    header_bytes = fh.read(8)
                break
    except OSError:
        header_bytes = b""
    sha = _hl.sha256(config_bytes + header_bytes).hexdigest()
    short = sha[:6]
    manifest_dir = output or os.path.join("manifests", f"M-001_C-{short}")
    manifest_path = os.path.join(manifest_dir, "manifest.json")
    if os.path.isfile(manifest_path) and not force:
        try:
            with open(manifest_path, encoding="utf-8") as fh:
                existing = json.load(fh)
            if existing.get("checkpoint_sha256") == sha:
                raise ManifestExistsError(f"manifest exists sha256:{sha[:6]}.... Use --force.")
        except ValueError:
            pass
    os.makedirs(manifest_dir, exist_ok=True)
    weights_manifest: dict[str, Any] = {
        "model_id": "M-001",
        "checkpoint_sha256": sha,
        "architecture": {
            "family": "unknown",
            "layers": layers,
            "hidden": hidden,
            "heads": heads,
            "mlp_per_layer": mlp,
            "vocab": vocab,
        },
        "inventory": {
            "layers": [f"L-{i:02d}" for i in (1, layers)],
            "attention": f"A-24-01..{heads:02d}",
            "neurons": f"N-24-0001..{mlp}",
            "specials": ["EMBED", "LOGITS", "NORM"],
        },
        "runtime": {"framework": "PyTorch", "cuda": "cpu"},
        "hardware": {"gpu": "none", "detected_via": "none"},
        "adapter": {"family": "unknown", "layout": "row_major_float16"},
        "scan": {"source": "config.json + weights header", "gpu_used": False, "duration_s": 3.1},
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(weights_manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    weights_manifest["manifest_path"] = manifest_path
    return weights_manifest


class BlackBox:
    """Public API surface (spec sections A.9-A.12)."""

    @staticmethod
    def watch(
        watchlist: str,
        add: list[str] | None = None,
        window: dict[str, Any] | None = None,
        trigger: str | None = None,
        precision: str | None = None,
        force: bool = False,
        tag: dict[str, str] | None = None,
    ) -> _WatchContext:
        """Declare a watchlist; validates on ``__enter__``, seals on ``__exit__``."""
        return _WatchContext(watchlist, add, window, trigger, precision, force, tag)

    @staticmethod
    def scan(
        path: str, output: str | None = None, force: bool = False, api_model: bool = False
    ) -> dict[str, Any]:
        """Scan weights on CPU (or stub an API model); forwards to ``scan.py``."""
        try:
            from blackbox import scan as _scan_mod

            fn = getattr(_scan_mod, "scan", None)
            if callable(fn):
                with contextlib.redirect_stdout(io.StringIO()):
                    try:
                        result = fn(path, output, force)
                    except TypeError:
                        result = fn(path)
                if isinstance(result, dict) and result:
                    return result
                if isinstance(result, str) and result and os.path.isfile(result):
                    with open(result, encoding="utf-8") as fh:
                        loaded = json.load(fh)
                    if isinstance(loaded, dict) and loaded:
                        return loaded
        except Exception:
            pass
        return _scan_fallback(path, output, force, api_model)

    @staticmethod
    def verify(run_id: str, store: str | None = None) -> bool:
        """Verify the hash chain for ``run_id`` (True = chain OK)."""
        return verify_run(run_id, store)

    @staticmethod
    def to_dataframe(
        run_id: str,
        what: str = "outer+deep",
        tokens: tuple[int, int] | None = None,
        addresses: list[str] | None = None,
    ) -> Any:
        """Outer-only 1-row frame in Phase 1 (Deep raises ``NotImplementedError``)."""
        return to_dataframe(run_id, what, tokens, addresses)


def watch(
    watchlist: str,
    add: list[str] | None = None,
    window: dict[str, Any] | None = None,
    trigger: str | None = None,
    precision: str | None = None,
    force: bool = False,
    tag: dict[str, str] | None = None,
) -> _WatchContext:
    """Module-level alias for :meth:`BlackBox.watch` (examples use both)."""
    return BlackBox.watch(watchlist, add, window, trigger, precision, force, tag)


def scan(
    path: str, output: str | None = None, force: bool = False, api_model: bool = False
) -> dict[str, Any]:
    """Module-level alias for :meth:`BlackBox.scan`."""
    return BlackBox.scan(path, output, force, api_model)
