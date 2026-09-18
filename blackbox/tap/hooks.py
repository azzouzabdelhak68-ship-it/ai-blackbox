"""Deep gather hooks — ONLY hook code lives here (spec §TAP-fsst Deep, plan §9.2).

Per-token path (weights + trigger armed only):

- ``register_forward_hook`` ONLY on parent modules named in the watchlist
  (never per neuron).
- Per token: ``gathered = activation.index_select(-1, idx)`` stays on
  device (watched columns only), then
  ``pinned.copy_(gathered, non_blocking=True)`` runs on a dedicated
  copyStream so the copy overlaps the next MatMul (~80% hidden).
- BAN (spec §TAP-fsst): per-hook blocking host copies are forbidden, and
  the default stream is never blocked here — this file must contain no
  blocking stream call, never a host copy call.
- Ring full -> ``flags["ring_overflow"] = True``: drop the Deep payload,
  never stall inference, Outer still seals.
- Hook throw -> auto ``detach()`` (model restored exactly) +
  ``flags["hook_detached"] = True``; the run continues on Outer only.
- int8 weights dequantize on device (``gathered * scale``) before the copy;
  the logical value plus ``dtype_meta`` is what seals.

``torch`` is imported lazily inside functions only (CPU hosts stay clean);
fake tensor/module stand-ins work as long as they expose ``index_select``,
``copy_`` and ``register_forward_hook``. No GPU code lives here — streams,
pinning and Events belong to ``blackbox/gpu_adapters/``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import Any

import numpy as np

from blackbox.controller import is_armed, parse_trigger

_RE_N = re.compile(r"^N-(\d+)-(\d+)$")
_RE_A = re.compile(r"^A-(\d+)-(\d+)$")
_RE_L = re.compile(r"^L-(\d+)$")

_NP_DTYPE: dict[str, Any] = {
    "fp32": np.dtype("<f4"),
    "fp16": np.dtype("<f2"),
    "bf16": np.dtype("<f2"),
}
_DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2}


def parent_module_name(addr: str) -> str:
    """Map a watched address to its parent module name (hook lands here)."""
    addr = addr.strip()
    m = _RE_N.match(addr)
    if m:
        return f"layer-{int(m.group(1)):02d}.mlp"
    m = _RE_A.match(addr)
    if m:
        return f"layer-{int(m.group(1)):02d}.attn"
    m = _RE_L.match(addr)
    if m:
        return f"layer-{int(m.group(1)):02d}"
    if addr in ("LOGITS", "EMBED", "NORM"):
        return addr.lower()
    raise ValueError(f"E02: bad address {addr}")


def address_local_index(addr: str) -> int:
    """Column offset of ``addr`` inside its parent activation (last dim)."""
    addr = addr.strip()
    m = _RE_N.match(addr)
    if m:
        return int(m.group(2)) - 1
    m = _RE_A.match(addr)
    if m:
        return int(m.group(2)) - 1
    if _RE_L.match(addr) or addr in ("LOGITS", "EMBED", "NORM"):
        return 0
    raise ValueError(f"E02: bad address {addr}")


def group_by_parent(addresses: list[str]) -> dict[str, list[tuple[str, int, int]]]:
    """Group watched addresses by parent: parent -> [(addr, local, col)]."""
    groups: dict[str, list[tuple[str, int, int]]] = {}
    for col, addr in enumerate(addresses):
        parent = parent_module_name(addr)
        groups.setdefault(parent, []).append((addr, address_local_index(addr), col))
    return groups


def _as_index(activation: Any, local: list[int]) -> Any:
    """Build an index object for ``index_select`` (device-placed when real)."""
    try:
        import torch  # noqa: F401  (lazy: CPU hosts without torch stay clean)

        if activation.__class__.__module__.split(".")[0] == "torch":
            device = getattr(activation, "device", None)
            return torch.as_tensor(list(local), dtype=torch.long, device=device)
    except ImportError:
        pass
    return list(local)


def _to_floats(gathered: Any) -> list[float]:
    """Drain a gathered 1-D buffer to Python floats (CPU/test drain)."""
    if hasattr(gathered, "tolist"):
        return [float(v) for v in gathered.tolist()]
    if isinstance(gathered, (list, tuple)):
        return [float(v) for v in gathered]
    try:
        return [float(v) for v in gathered]
    except TypeError as exc:
        raise ValueError(f"E02: cannot drain gathered buffer {type(gathered)}") from exc


class _HostSlot:
    """One pinned host staging slot (double-buffered per parent group)."""

    def __init__(self, width: int) -> None:
        self.buf: list[float] = [0.0] * width

    def copy_(self, src: Any, non_blocking: bool = True) -> _HostSlot:
        """Stage ``src`` into this slot on the dedicated copyStream."""
        _ = non_blocking  # async on CUDA via the adapter stream; queued on CPU
        vals = _to_floats(src)
        self.buf = list(vals)
        return self

    def read(self) -> list[float]:
        """Read staged values back (CPU/test drain; sidecar owns CUDA drain)."""
        return list(self.buf)


HookFn = Callable[[Any, Any, Any], None]


class DeepTap:
    """Portable Deep gather recorder (one hook per watched parent module)."""

    def __init__(
        self,
        addresses: list[str],
        window: dict[str, int] | None = None,
        trigger: str | dict[str, Any] | None = None,
        dtype: str = "fp16",
        capacity_tokens: int | None = None,
        scale: float = 1.0,
        stream: Any = None,
    ) -> None:
        if dtype not in _DTYPE_BYTES:
            raise ValueError(f"E02: bad precision {dtype!r}")
        self.watched: list[str] = list(addresses)
        self.window: dict[str, int] = dict(
            window if window is not None else {"start_token": 0, "end_token": 1 << 30, "every_n": 1}
        )
        if isinstance(trigger, dict) and "kind" in trigger:
            self.trigger: dict[str, Any] = dict(trigger)
        else:
            self.trigger = parse_trigger(trigger)
        self.dtype: str = dtype
        self.capacity_tokens: int | None = capacity_tokens
        self.scale: float = float(scale)
        self.stream: Any = stream  # dedicated copyStream; one Event per group
        self.groups: dict[str, list[tuple[str, int, int]]] = group_by_parent(self.watched)
        self.pinned: dict[str, list[_HostSlot]] = {
            parent: [_HostSlot(len(cols)), _HostSlot(len(cols))]
            for parent, cols in self.groups.items()
        }
        self._flip: dict[str, int] = {parent: 0 for parent in self.groups}
        self.events: dict[str, Any] = {parent: None for parent in self.groups}
        self.ring: list[list[float]] = []
        self.flags: dict[str, bool] = {
            "ring_overflow": False,
            "hook_detached": False,
            "triggered": False,
        }
        self.dtype_meta: dict[str, str] = {
            "stored": dtype,
            "source": dtype,
            "endian": "little",
            "order": "row_major_token_major",
        }
        self._staging: dict[int, float] = {}
        self._handles: list[Any] = []
        self._detached: bool = False

    @property
    def detached(self) -> bool:
        """True once hooks were removed (throw path or explicit detach)."""
        return self._detached

    def _resolve(self, model: Any, parent: str) -> Any:
        """Resolve a parent module: ``get_submodule`` first, then attr walk."""
        getter = getattr(model, "get_submodule", None)
        if callable(getter):
            try:
                return getter(parent)
            except (AttributeError, KeyError, ValueError):
                pass
        node: Any = model
        for part in parent.split("."):
            node = getattr(node, part)
        return node

    def attach(self, model: Any) -> list[Any]:
        """Register one forward hook per watched parent module (idempotent)."""
        if self._handles:
            return list(self._handles)
        for parent, cols in self.groups.items():
            try:
                module = self._resolve(model, parent)
            except AttributeError as exc:
                raise ValueError(f"E02: model has no parent module {parent!r}") from exc
            hook = self._make_hook(parent, cols)
            self._handles.append(module.register_forward_hook(hook))
        return list(self._handles)

    def _make_hook(self, parent: str, cols: list[tuple[str, int, int]]) -> HookFn:
        locals_only = [local for _, local, _ in cols]
        col_ids = [col for _, _, col in cols]

        def _hook(module: Any, inputs: Any, output: Any) -> None:
            _ = (module, inputs)
            if self._detached:
                return
            try:
                activation = output[0] if isinstance(output, tuple) else output
                raw_dtype = str(getattr(activation, "dtype", self.dtype))
                idx = _as_index(activation, locals_only)
                gathered = activation.index_select(-1, idx)
                if "int8" in raw_dtype or "char" in raw_dtype:
                    gathered = gathered * self.scale  # dequant on device before copy
                    self.dtype_meta["source"] = "int8_dequantized_on_device"
                flip = self._flip[parent]
                slot = self.pinned[parent][flip]
                self._flip[parent] = flip ^ 1
                slot.copy_(gathered, non_blocking=True)  # copyStream, overlaps next MatMul
                for col, value in zip(col_ids, slot.read()):
                    self._staging[col] = value
            except Exception:
                self.detach()  # hook throw -> auto detach, model restored exactly
                self.flags["hook_detached"] = True

        return _hook

    def detach(self) -> None:
        """Remove all hook handles; idempotent, never raises."""
        self._detached = True
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                continue
        self._handles = []

    def end_token(
        self,
        token_pos: int,
        elapsed_s: float | None = None,
        flags: dict[str, Any] | None = None,
        values: dict[str, float] | None = None,
    ) -> bool:
        """Seal one token: armed -> append row; unarmed -> write nothing.

        Returns True when a row was appended. Ring-full sets
        ``ring_overflow`` and drops the Deep payload (Outer still seals).
        After auto-detach (hook throw) no Deep rows append: the run
        continues on Outer only.
        """
        if self._detached or self.flags.get("hook_detached"):
            self._staging = {}
            return False
        gate_values: dict[str, float] = {}
        for col, addr in enumerate(self.watched):
            if col in self._staging:
                gate_values[addr] = self._staging[col]
        if values:
            gate_values.update(values)
        armed = is_armed(self.trigger, token_pos, self.window, elapsed_s, flags, gate_values)
        self._staging = {}
        if not armed:
            return False
        if self.capacity_tokens is not None and len(self.ring) >= self.capacity_tokens:
            self.flags["ring_overflow"] = True
            return False
        row = [gate_values.get(addr, math.nan) for addr in self.watched]
        self.ring.append(row)
        self.flags["triggered"] = True
        return True

    def to_bytes(self, dtype: str | None = None) -> bytes:
        """Encode the ring row-major ``[T, W]`` little-endian (spec §FILE-deepbin)."""
        stored = dtype or self.dtype
        if stored not in _DTYPE_BYTES:
            raise ValueError(f"E02: bad precision {stored!r}")
        if not self.ring:
            return b""
        if stored == "bf16":
            import struct

            out = bytearray()
            for row in self.ring:
                for value in row:
                    out += struct.pack("<f", float(value))[:2]  # truncate fp32 -> bf16
            return bytes(out)
        arr = np.asarray(self.ring, dtype=_NP_DTYPE[stored])
        return bytes(arr.tobytes())

    def deep_index(self, run_id: str) -> dict[str, Any]:
        """Build the ``deep_index.json`` pointer for the sealed ring."""
        tokens = len(self.ring)
        columns: dict[str, dict[str, Any]] = {}
        for col, addr in enumerate(self.watched):
            columns[addr] = {"offset": col, "dtype": self.dtype, "shape": [tokens]}
        return {
            "run_id": run_id,
            "triggered": bool(self.flags["triggered"]),
            "dtype": self.dtype,
            "tokens": tokens,
            "every_n": int(self.window.get("every_n", 1)),
            "window": {
                "start_token": int(self.window.get("start_token", 0)),
                "end_token": int(self.window.get("end_token", 0)),
            },
            "watched": list(self.watched),
            "columns": columns,
            "dtype_meta": dict(self.dtype_meta),
        }


class AttachedTaps(list[Any]):
    """Hook handles list carrying the :class:`DeepTap` recorder."""

    def __init__(self, handles: list[Any], recorder: DeepTap) -> None:
        super().__init__(handles)
        self.recorder: DeepTap = recorder


def attach_hooks(
    model: Any,
    watchlist: dict[str, Any],
    window: dict[str, int] | None = None,
    trigger: str | dict[str, Any] | None = None,
    dtype: str | None = None,
    capacity_tokens: int | None = None,
    scale: float = 1.0,
    stream: Any = None,
) -> AttachedTaps:
    """Register Deep gather hooks only on watched parent modules.

    ``watchlist`` must carry ``addresses`` (see :func:`group_by_parent`);
    window/trigger/dtype default to the watchlist's merged values.
    """
    if not isinstance(watchlist, dict) or "addresses" not in watchlist:
        raise ValueError("E02: watchlist dict must carry 'addresses'")
    addresses = [str(a) for a in watchlist["addresses"]]
    eff_window = window or watchlist.get("merged_window") or watchlist.get("window")
    eff_trigger = (
        trigger
        if trigger is not None
        else watchlist.get("effective_trigger", watchlist.get("trigger"))
    )
    eff_dtype = dtype or str(
        watchlist.get("effective_precision", watchlist.get("precision", "fp16"))
    )
    tap = DeepTap(
        addresses,
        window=dict(eff_window) if isinstance(eff_window, dict) else None,
        trigger=eff_trigger,
        dtype=eff_dtype,
        capacity_tokens=capacity_tokens,
        scale=scale,
        stream=stream,
    )
    handles = tap.attach(model)
    return AttachedTaps(handles, tap)


def detach_hooks(handles: list[Any]) -> None:
    """Remove hook handles; idempotent, never raises (hook-throw safe)."""
    recorder = getattr(handles, "recorder", None)
    if isinstance(recorder, DeepTap):
        recorder.detach()
        try:
            handles.clear()
        except Exception:
            pass
        return
    for handle in list(handles):
        try:
            handle.remove()
        except Exception:
            continue
    try:
        handles.clear()
    except Exception:
        pass
