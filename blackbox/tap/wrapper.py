"""Outer proxy wrapper (spec section A.10, plan.md section 8.2).

CPU-only Phase 1: records prompt/output tokens, timing, seed/settings,
cost and flags at the ``generate()`` boundary. No hook code here by rule
(hooks live in ``blackbox/tap/hooks.py``). No GPU code: this module MUST
NOT import ``torch.cuda`` so Outer works on Windows/Mac/Linux without
NVIDIA.

Privacy: default raw (research on synthetic/consented sets). ``redactor``
is a caller-owned hook point only -- never a mandated filter.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

_BLACKBOX_VERSION = "0.1.0"

# Required ``outer.json`` keys (spec section FILE-outer). ``logits_summary``,
# ``cost`` and ``tags`` are optional in the schema but we always emit them
# (explicit ``null`` where unavailable) so callers can rely on presence.
REQUIRED_OUTER_FIELDS: tuple[str, ...] = (
    "run_id",
    "model_id",
    "checkpoint_sha256",
    "watchlist_ref",
    "prompt",
    "output",
    "timing",
    "sampling",
    "versions",
    "flags",
    "replay",
    "error",
)

Redactor = Callable[..., Any]


def tokenize(text: str) -> list[int]:
    """Deterministic CPU-only tokenization (UTF-8 bytes, no weights needed)."""
    if not text:
        return []
    return list(text.encode("utf-8"))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


_VERSIONS_CACHE: dict[str, str] | None = None


def _runtime_versions() -> dict[str, str]:
    """Best-effort runtime versions without touching the GPU.

    Uses package metadata (no ``torch`` import, so no GPU init and no
    driver warnings); never imports ``torch.cuda`` (CPU rule). Cached so
    per-request overhead stays at the ~0.15ms target.
    """
    global _VERSIONS_CACHE
    if _VERSIONS_CACHE is None:
        torch_ver = "unknown"
        try:
            from importlib.metadata import version

            torch_ver = str(version("torch")).split("+")[0]
        except Exception:
            torch_ver = "unknown"
        _VERSIONS_CACHE = {"torch": torch_ver, "cuda": "cpu"}
    return dict(_VERSIONS_CACHE)


def _extract_text(value: Any) -> str:
    """Best-effort output text from str / .text / dict / token-list shapes."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    text = getattr(value, "text", None)
    if isinstance(text, str):
        return text
    if isinstance(value, dict):
        for key in ("text", "output", "response", "content"):
            if isinstance(value.get(key), str):
                return str(value[key])
    return str(value)


def _apply_redactor(
    redactor: Redactor | None, prompt: str, tokens: list[int]
) -> tuple[str, list[int]]:
    """Apply the caller-owned redactor hook. Supports 1-arg and 2-arg forms."""
    if redactor is None:
        return prompt, tokens
    try:
        out = redactor(prompt, tokens)
    except TypeError:
        out = redactor(prompt)
    if isinstance(out, tuple) and len(out) == 2:
        new_prompt, new_tokens = out
        return str(new_prompt), list(new_tokens)
    return str(out), tokens


class Wrapped:
    """Thin Outer tap around ``model.generate`` (spec section A.10).

    Stores the original ``generate`` fn, records Outer fields on each call,
    returns the model output unchanged. ``detach()`` restores the model
    exactly and is idempotent. Recording failures never kill the caller:
    they set flags and auto-detach.
    """

    def __init__(
        self,
        model: Any,
        redactor: Redactor | None = None,
        capture_logits: bool = True,
        seed: int | None = None,
    ) -> None:
        if not hasattr(model, "generate"):
            raise ValueError("wrap() requires a model with a generate() method")
        self._model: Any = model
        self._original_generate: Callable[..., Any] = model.generate
        self._redactor: Redactor | None = redactor
        self._capture_logits: bool = bool(capture_logits)
        self._default_seed: int | None = seed
        self._detached: bool = False
        self.ring_full: bool = False
        self.last_outer: dict[str, Any] | None = None
        # One bound-method object: identity checks stay valid (fresh
        # ``self.generate`` lookups would create new objects each time).
        self._patched_generate: Callable[..., Any] = self.generate
        model.generate = self._patched_generate  # type: ignore[method-assign]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_model"), name)

    @property
    def detached(self) -> bool:
        """True once :meth:`detach` has restored the original model."""
        return self._detached

    def build_outer(
        self,
        prompt_text: str,
        output_text: str,
        elapsed_ms: float,
        seed: int,
        temp: float,
        top_p: float,
        sampler: str,
        t_start_iso: str,
        t_end_iso: str,
        run_id: str = "RUN-000000",
        error: dict[str, Any] | None = None,
        ring_overflow: bool = False,
    ) -> dict[str, Any]:
        """Build the ``outer.json`` dict (all required fields present)."""
        prompt_tokens = tokenize(prompt_text)
        output_tokens = tokenize(output_text)
        redacted_prompt, redacted_tokens = _apply_redactor(
            self._redactor, prompt_text, prompt_tokens
        )
        total_tokens = max(1, len(redacted_tokens) + len(output_tokens))
        ms_per_token = elapsed_ms / total_tokens
        versions = _runtime_versions()
        versions["blackbox"] = _BLACKBOX_VERSION
        versions["adapter"] = "cpu"
        versions["redactor"] = "custom" if self._redactor is not None else "none"
        outer: dict[str, Any] = {
            "run_id": run_id,
            "model_id": "M-EXT",
            "checkpoint_sha256": "0" * 64,
            "watchlist_ref": {
                "watchlist_id": "outer_only",
                "watchlist_sha256": "0" * 64,
                "patch": None,
            },
            "prompt": {
                "text": redacted_prompt,
                "tokens": redacted_tokens,
                "tokenizer": "utf8-bytes",
            },
            "output": {
                "text": output_text,
                "tokens": output_tokens,
                "token_order": list(range(len(output_tokens))),
                "finish_reason": "stop",
            },
            "logits_summary": None,
            "timing": {
                "t_start_iso": t_start_iso,
                "t_end_iso": t_end_iso,
                "ms_per_token": ms_per_token,
                "cuda_event_ms": elapsed_ms,
            },
            "sampling": {"seed": seed, "temp": temp, "top_p": top_p, "sampler": sampler},
            "versions": versions,
            "cost": {
                "input_tokens": len(redacted_tokens),
                "output_tokens": len(output_tokens),
                "usd": 0.0,
            },
            "flags": {
                "bad_output": False,
                "ring_overflow": bool(ring_overflow or self.ring_full),
                "truncated": False,
            },
            "tags": [],
            "replay": {
                "seed": seed,
                "temp": temp,
                "top_p": top_p,
                "sampler": sampler,
                "cudnn_deterministic": True,
                "checkpoint_sha256": "0" * 64,
                "versions": {"torch": versions["torch"], "cuda": versions["cuda"]},
            },
            "error": error,
        }
        return outer

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Record Outer fields around the original ``generate`` call."""
        kwargs = dict(kwargs)
        run_id = str(kwargs.pop("run_id", "RUN-000000"))
        seed = kwargs.get("seed", self._default_seed if self._default_seed is not None else 42)
        try:
            seed_int = int(seed)
        except (TypeError, ValueError):
            seed_int = 42
        temp = float(kwargs.get("temp", kwargs.get("temperature", 0.0)))
        top_p = float(kwargs.get("top_p", 1.0))
        sampler = str(kwargs.get("sampler", "greedy"))
        if args and isinstance(args[0], str):
            prompt_text = args[0]
        else:
            prompt_text = str(kwargs.get("prompt", kwargs.get("input", "")))
        t_start_iso = _utc_now_iso()
        t_start = time.perf_counter()
        try:
            output = self._original_generate(*args, **kwargs)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            try:
                self.last_outer = self.build_outer(
                    prompt_text,
                    "",
                    elapsed_ms,
                    seed_int,
                    temp,
                    top_p,
                    sampler,
                    t_start_iso,
                    _utc_now_iso(),
                    run_id,
                    error={"code": "model_error", "message": str(exc)},
                )
            except Exception:
                self.last_outer = None
            raise
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        output_text = _extract_text(output)
        # Recording must never kill the caller: flag + detach on failure.
        try:
            self.last_outer = self.build_outer(
                prompt_text,
                output_text,
                elapsed_ms,
                seed_int,
                temp,
                top_p,
                sampler,
                t_start_iso,
                _utc_now_iso(),
                run_id,
            )
        except Exception as exc:
            try:
                self.last_outer = self.build_outer(
                    "",
                    "",
                    elapsed_ms,
                    seed_int,
                    temp,
                    top_p,
                    sampler,
                    t_start_iso,
                    _utc_now_iso(),
                    run_id,
                    error={"code": "wrapper_error", "message": str(exc)},
                )
            except Exception:
                self.last_outer = None
            try:
                self.detach()
            except Exception:
                pass
        return output

    def save_outer(self, path: str) -> str:
        """Write the last recorded outer dict to ``path`` (pretty, ~3KB)."""
        import json

        if self.last_outer is None:
            raise ValueError("no generate() call recorded yet; call generate() first")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.last_outer, fh, indent=2, sort_keys=True)
            fh.write("\n")
        return path

    def detach(self) -> None:
        """Restore the original model exactly; idempotent (safe to repeat)."""
        if self._detached:
            return
        try:
            try:
                current = self._model.__dict__.get("generate", None)
            except AttributeError:
                current = None
            if current is None:
                ours = bool(getattr(self._model, "generate", None) == self._patched_generate)
            else:
                ours = current is self._patched_generate
            if ours:
                self._model.generate = self._original_generate
        finally:
            self._detached = True


# Backward-compatible alias (Phase 0 stub name).
WrappedModel = Wrapped


def wrap(
    model: Any,
    redactor: Redactor | None = None,
    capture_logits: bool = True,
    seed: int | None = None,
) -> Wrapped:
    """Wrap a model with the Outer tap (spec section A.10).

    Deep hooks/sidecar are Phase 2 and intentionally absent here.
    """
    return Wrapped(model, redactor=redactor, capture_logits=capture_logits, seed=seed)
