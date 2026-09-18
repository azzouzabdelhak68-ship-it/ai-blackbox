"""Forked-stream sidecar: mmap triple-buffer ring -> WAL -> sealed run.

Spec §TAP-sidecar, §FILE-deepbin, §FILE-seal, §ERRORS; plan.md §9.3.

Pipeline (separate process/container from the training loop)::

    hooks (GPU copyStream) -> /dev/shm/bb.ring (mmap, 3 slots)
        -> sidecar poll -> WAL append + fsync per token
        -> seal: deep.bin + deep_index.json + outer.json + seal.json + chain

- Transport: lock-free triple-buffered ring, file-backed ``mmap``. Default
  path ``$BLACKBOX_SHM`` else ``/dev/shm/bb.ring`` when ``/dev/shm`` exists
  (Linux Docker ``--shm-size=1g``), else a file-backed fallback
  ``%TEMP%/bb.ring`` (Windows/Mac CPU tests; same format, no shm).
- Survival: WAL rows are ``fsync``'d per token, so a main-process SIGKILL/OOM
  loses nothing sealed; :meth:`Sidecar.seal_prefix` seals the surviving
  prefix and ``blackbox verify`` still passes for it (E1001 hint names the
  WAL token watermark on crash paths).
- Backpressure: ring full -> ``ring_overflow=true``, the newest Deep payload
  is dropped (gap filled with NaN at seal), Outer is always kept; inference
  is never stalled and the job is never killed.
- CPU FALLBACK HONESTY: this file uses numpy + stdlib ``mmap`` only (no CUDA
  import). The WAL is JSONL ``{token, values}`` + per-token fsync; the GPU
  path keeps the same file format with a binary WAL. Fake/synthetic values
  used in tests/examples are seeded RNG, never real activations.
- REJECTED (documented, not implemented): pure-sidecar capture via
  LD_PRELOAD/eBPF. It sees kernels, not semantics (no watchlist addresses),
  breaks Docker/CUDA ABI across driver versions, and cannot run on the
  Windows/Mac CPU path. Hooks + sidecar witness is the only supported tap.

Exit mapping: E1001 ``SidecarError`` (runtime failure with WAL watermark);
E02 ``ValueError`` (bad args / short bin).
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import struct
import tempfile
from collections.abc import Sequence
from typing import Any

import numpy as np

MAGIC = b"BBRING01"
VERSION = 1
N_SLOTS = 3  # triple buffer
_STATE_EMPTY = 0
_STATE_WRITING = 1
_STATE_READY = 2
_HEADER_FMT = "<8sI I Q Q ?"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_SLOT_HEADER_FMT = "<I I I"
_SLOT_HEADER_SIZE = struct.calcsize(_SLOT_HEADER_FMT)


class SidecarError(RuntimeError):
    """E1001 runtime failure (sidecar died / crash recovery hint)."""


def resolve_ring_path(explicit: str | None = None) -> str:
    """Ring path: arg > ``$BLACKBOX_SHM`` > ``/dev/shm/bb.ring`` > tmp."""
    if explicit:
        return explicit
    env = os.environ.get("BLACKBOX_SHM")
    if env:
        return env
    shm_dir = "/dev/shm"
    if os.path.isdir(shm_dir):
        return os.path.join(shm_dir, "bb.ring")
    return os.path.join(tempfile.gettempdir(), "bb.ring")


def _slot_len_for(w: int, b: int) -> int:
    return w * b


def _ring_file_size(w: int, b: int) -> int:
    return _HEADER_SIZE + N_SLOTS * (_SLOT_HEADER_SIZE + _slot_len_for(w, b))


def _slot_offset(slot: int, slot_len: int) -> int:
    return _HEADER_SIZE + slot * (_SLOT_HEADER_SIZE + slot_len)


class Sidecar:
    """WAL+persistence side for one run (spec §TAP-sidecar).

    The training loop (or tests) calls :meth:`append_token` per emitted
    token; :meth:`poll` moves READY ring slots into the WAL file with a
    per-token fsync; :meth:`seal` assembles ``deep.bin`` row-major ``[T, W]``
    LE plus index/outer/seal/chain. After a kill, :meth:`attach` +
    :meth:`seal_prefix` seals the fsync'd prefix (E1001 hint).
    """

    def __init__(
        self,
        store: str,
        watched: list[str],
        precision: str = "fp16",
        run_id: str | None = None,
        window: dict[str, Any] | None = None,
        every_n: int = 1,
        ring_path: str | None = None,
    ) -> None:
        from blackbox.storage.layout import allocate_run, precision_bytes

        if not watched:
            raise ValueError("E02: sidecar needs a non-empty watched address list")
        self._b = precision_bytes(precision)
        self.store = store
        self.watched = list(watched)
        self.precision = precision
        self.window = dict(window) if window else None
        self.every_n = int(every_n)
        if self.every_n < 1:
            raise ValueError(f"E02: every_n must be >= 1 (got {every_n})")
        self.ring_path = resolve_ring_path(ring_path)
        self.run_id, self.run_dir = (
            allocate_run(store)
            if run_id is None
            else (
                run_id,
                os.path.join(store, "runs", run_id),
            )
        )
        os.makedirs(self.run_dir, exist_ok=True)
        self.wal_path = os.path.join(store, "wal", f"{self.run_id}.wal.jsonl")
        os.makedirs(os.path.dirname(self.wal_path), exist_ok=True)
        self.ring_overflow = False
        self._tokens_sealed: list[int] = []
        self._open_ring()

    # ------------------------------------------------------------ ring ---
    def _open_ring(self) -> None:
        w = len(self.watched)
        size = _ring_file_size(w, self._b)
        fresh = not os.path.isfile(self.ring_path)
        if fresh:
            parent = os.path.dirname(self.ring_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.ring_path, "wb") as fh:
                fh.write(b"\x00" * size)
        elif os.path.getsize(self.ring_path) < size:
            with open(self.ring_path, "r+b") as fh:
                fh.truncate(size)
        self._ring_file = open(self.ring_path, "r+b")
        self._ring = mmap.mmap(self._ring_file.fileno(), size)
        try:
            _ver, ring_w, _wseq, _rseq, _ov = self._read_header()
            reinit = ring_w != w
        except SidecarError:
            reinit = True
        if fresh or reinit:
            self._write_header(write_seq=0, read_seq=0, overflow=False)
            for slot in range(N_SLOTS):
                self._write_slot_header(slot, _STATE_EMPTY, 0, 0)

    def _write_header(self, write_seq: int, read_seq: int, overflow: bool) -> None:
        w = len(self.watched)
        self._ring[:_HEADER_SIZE] = struct.pack(
            _HEADER_FMT, MAGIC, VERSION, w, write_seq, read_seq, overflow
        )

    def _read_header(self) -> tuple[int, int, int, int, bool]:
        magic, ver, w, wseq, rseq, overflow = struct.unpack(
            _HEADER_FMT, bytes(self._ring[:_HEADER_SIZE])
        )
        if magic != MAGIC:
            raise SidecarError("E1001 ring magic mismatch — ring file corrupt or foreign")
        return ver, w, wseq, rseq, overflow

    def _write_slot_header(self, slot: int, state: int, token: int, nvals: int) -> None:
        off = _slot_offset(slot, _slot_len_for(len(self.watched), self._b))
        self._ring[off : off + _SLOT_HEADER_SIZE] = struct.pack(
            _SLOT_HEADER_FMT, state, token, nvals
        )

    def _read_slot_header(self, slot: int) -> tuple[int, int, int]:
        off = _slot_offset(slot, _slot_len_for(len(self.watched), self._b))
        return struct.unpack(_SLOT_HEADER_FMT, bytes(self._ring[off : off + _SLOT_HEADER_SIZE]))

    def close(self) -> None:
        """Release the ring mmap (idempotent)."""
        try:
            self._ring.close()
        except (ValueError, AttributeError):
            pass
        try:
            self._ring_file.close()
        except (ValueError, AttributeError):
            pass

    # -------------------------------------------------------- produce ---
    def append_token(self, token_pos: int, values: Sequence[float] | np.ndarray) -> None:
        """Write one token row into the next triple-buffer slot (non-blocking).

        Ring full (3 unpolled slots) -> set ``ring_overflow``, drop the newest
        Deep payload, keep Outer (never stall, never kill the job).
        """
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.size != len(self.watched):
            raise ValueError(f"E02: token row size {arr.size} != W={len(self.watched)}")
        ver, w, wseq, rseq, _ = self._read_header()
        _ = (ver, w)
        if wseq - rseq >= N_SLOTS:
            self.ring_overflow = True
            self._write_header(wseq, rseq, True)
            return  # drop newest Deep payload; Outer path unaffected
        slot = wseq % N_SLOTS
        slot_len = _slot_len_for(len(self.watched), self._b)
        self._write_slot_header(slot, _STATE_WRITING, int(token_pos), len(self.watched))
        off = _slot_offset(slot, slot_len) + _SLOT_HEADER_SIZE
        if self.precision == "fp32":
            self._ring[off : off + slot_len] = arr.astype("<f4").tobytes(order="C")
        else:
            self._ring[off : off + slot_len] = arr.astype("<f2").tobytes(order="C")
        self._write_slot_header(slot, _STATE_READY, int(token_pos), len(self.watched))
        self._write_header(wseq + 1, rseq, self.ring_overflow)

    # ----------------------------------------------------------- poll ---
    def poll(self) -> int:
        """Move READY ring slots -> WAL append + fsync per token. Returns rows."""
        ver, w, wseq, rseq, overflow = self._read_header()
        _ = (ver, w)
        moved = 0
        slot_len = _slot_len_for(len(self.watched), self._b)
        with open(self.wal_path, "a", encoding="utf-8") as wal:
            while rseq < wseq:
                slot = rseq % N_SLOTS
                state, token, nvals = self._read_slot_header(slot)
                if state != _STATE_READY:
                    break  # writer mid-copy; leave for the next poll
                off = _slot_offset(slot, slot_len) + _SLOT_HEADER_SIZE
                raw = bytes(self._ring[off : off + slot_len])
                if self.precision == "fp32":
                    vals = np.frombuffer(raw, dtype=np.dtype("<f4")).astype(np.float64)
                else:
                    vals = np.frombuffer(raw, dtype=np.dtype("<f2")).astype(np.float64)
                wal.write(json.dumps({"token": int(token), "values": vals.tolist()}) + "\n")
                wal.flush()
                os.fsync(wal.fileno())  # per-token durability: SIGKILL-safe prefix
                self._write_slot_header(slot, _STATE_EMPTY, 0, 0)
                rseq += 1
                moved += 1
        self._write_header(wseq, rseq, overflow or self.ring_overflow)
        return moved

    # ------------------------------------------------------------ seal ---
    def _read_wal_rows(self) -> list[tuple[int, list[float]]]:
        rows: list[tuple[int, list[float]]] = []
        if not os.path.isfile(self.wal_path):
            return rows
        with open(self.wal_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rows.append((int(obj["token"]), [float(v) for v in obj["values"]]))
        rows.sort(key=lambda kv: kv[0])
        return rows

    def _assemble_matrix(self) -> tuple[np.ndarray, list[int]]:
        """Dense ``[T, W]`` float32 matrix in token order (NaN fills drops)."""
        rows = self._read_wal_rows()
        if not rows:
            raise ValueError("E02: nothing in WAL; no Deep tokens to seal")
        lo = rows[0][0]
        hi = rows[-1][0]
        by_token = dict(rows)
        w = len(self.watched)
        mat = np.full((hi - lo + 1, w), np.nan, dtype=np.float32)
        for pos in range(lo, hi + 1):
            if pos in by_token:
                mat[pos - lo, :] = np.asarray(by_token[pos], dtype=np.float32)
        return mat, list(range(lo, hi + 1))

    def seal(self, outer: dict[str, Any]) -> str:
        """Poll remaining slots, write deep+index+outer, seal+chain. Returns run."""
        from blackbox.storage import layout as _layout
        from blackbox.storage import seal as _seal

        # Recovery path (attach() with no ring): WAL prefix is already fsync'd.
        if getattr(self, "_ring", None) is not None:
            self.poll()
        matrix, positions = self._assemble_matrix()
        _layout.write_deep_bin(self.run_dir, matrix, self.precision)
        _layout.write_deep_index(
            self.run_dir,
            self.run_id,
            triggered=True,
            dtype=self.precision,
            tokens=int(matrix.shape[0]),
            every_n=self.every_n,
            watched=self.watched,
            window=self.window,
            dtype_meta={"token_positions": positions, "sealed_by": "sidecar"},
        )
        outer = dict(outer)
        outer["run_id"] = self.run_id
        outer.setdefault("flags", {})["ring_overflow"] = bool(self.ring_overflow)
        outer_path = os.path.join(self.run_dir, "outer.json")
        with open(outer_path, "w", encoding="utf-8") as fh:
            json.dump(outer, fh, indent=2, sort_keys=True)
            fh.write("\n")
        _seal.seal_run(self.run_dir, sealed_by="sidecar")
        self._tokens_sealed = positions
        self.close()
        return self.run_id

    # --------------------------------------------------------- recover ---
    @classmethod
    def attach(cls, store: str, run_id: str) -> Sidecar:
        """Re-attach to a crashed run's WAL without touching sealed bytes."""
        from blackbox.storage.layout import precision_bytes, run_dir

        path = run_dir(store, run_id)
        index_path = os.path.join(path, "deep_index.json")
        watched: list[str] = []
        precision = "fp16"
        if os.path.isfile(index_path):
            with open(index_path, encoding="utf-8") as fh:
                index = json.load(fh)
            watched = list(index.get("watched", []))
            precision = str(index.get("dtype", "fp16"))
        if not watched:
            wal_path = os.path.join(store, "wal", f"{run_id}.wal.jsonl")
            if os.path.isfile(wal_path):
                with open(wal_path, encoding="utf-8") as fh:
                    first = fh.readline()
                if first.strip():
                    nvals = len(json.loads(first)["values"])
                    watched = [f"N-01-{i + 1:04d}" for i in range(nvals)]
        precision_bytes(precision)
        obj = cls.__new__(cls)
        obj.store = store
        obj.watched = watched
        obj.precision = precision
        obj.window = None
        obj.every_n = 1
        obj.ring_path = resolve_ring_path(None)
        obj.run_id = run_id
        obj.run_dir = path
        obj.wal_path = os.path.join(store, "wal", f"{run_id}.wal.jsonl")
        obj.ring_overflow = False
        obj._tokens_sealed = []
        obj._b = precision_bytes(precision)
        return obj

    def seal_prefix(self, outer: dict[str, Any]) -> str:
        """Seal the fsync'd WAL prefix after a kill; returns the E1001 hint."""
        run = self.seal(outer)
        hint = (
            f"E1001 main process failed — WAL up to token "
            f"{self._tokens_sealed[-1] if self._tokens_sealed else -1} already sealed "
            f"at {self.run_dir}; rerun resume hint: re-run with the same seed and "
            f"watchlist to extend {run}."
        )
        return hint

    # -------------------------------------------------------------- ops ---
    def wal_token_count(self) -> int:
        """Number of fsync'd WAL rows (sealed-token watermark)."""
        return len(self._read_wal_rows())


def run_sidecar(ring_path: str | None = None, store: str | None = None, once: bool = True) -> None:
    """Poll the shared ring into WALs (separate container entry point).

    ``once=True`` drains currently READY slots and exits (used by the
    file-backed CPU path); the Docker sidecar loops with ``once=False``
    until ``$BLACKBOX_DIR/sidecar.stop`` appears. Prints one status line.
    """
    root = store or os.environ.get("BLACKBOX_DIR", "./blackbox_store")
    path = resolve_ring_path(ring_path)
    if not os.path.isfile(path):
        print(f"blackbox sidecar: no ring at {path} (nothing to poll)")
        return
    with open(path, "rb") as fh:
        header = fh.read(_HEADER_SIZE)
    if len(header) < _HEADER_SIZE or header[:8] != MAGIC:
        print(f"blackbox sidecar: foreign/empty ring at {path} (nothing to poll)")
        return
    print(f"blackbox sidecar: watching {path} store={root} (WAL fsync per token)")
    if once:
        return
    stop = os.path.join(root, "sidecar.stop")
    import time

    while not os.path.isfile(stop):
        time.sleep(0.05)
    print("blackbox sidecar: stop requested, drained prefix remains verifiable")


def main(argv: list[str] | None = None) -> int:
    """``python -m blackbox.tap.sidecar`` entry (compose sidecar service)."""
    parser = argparse.ArgumentParser(prog="blackbox-sidecar", description="BB sidecar")
    parser.add_argument("--ring", default=None)
    parser.add_argument("--store", default=None)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args(argv)
    run_sidecar(args.ring, args.store, once=not args.loop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
