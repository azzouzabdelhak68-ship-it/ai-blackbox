"""Outer wrapper tests (spec sections A.10, FILE-outer; plan.md section 8.6).

CPU-only: fake model ``generate``, no weights, no GPU. Run with
``BLACKBOX_GPU=cpu pytest tests/test_outer_wrapper.py -q``.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from blackbox.tap.wrapper import REQUIRED_OUTER_FIELDS, wrap


class FakeOutput:
    """Minimal closed-API response stand-in."""

    def __init__(self, text: str) -> None:
        self.text = text


class FakeModel:
    """Minimal closed-API client stand-in (no weights, no GPU)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def generate(self, prompt: str, seed: int = 42, **kwargs: Any) -> FakeOutput:
        _ = kwargs
        self.calls.append((prompt, seed))
        return FakeOutput(text=f"echo:{prompt}")


def test_wrap_records_outer_and_returns_unchanged() -> None:
    model = FakeModel()
    original = model.generate
    wrapped = wrap(model)
    out = wrapped.generate("Hello world, this is a slightly longer prompt.", seed=42)
    assert isinstance(out, FakeOutput)
    assert out.text == "echo:Hello world, this is a slightly longer prompt."
    assert model.calls == [("Hello world, this is a slightly longer prompt.", 42)]
    assert wrapped.last_outer is not None
    assert wrapped.last_outer["prompt"]["text"].startswith("Hello world")
    assert wrapped.last_outer["sampling"]["seed"] == 42
    assert wrapped.last_outer["timing"]["ms_per_token"] >= 0.0
    assert wrapped.last_outer["cost"]["output_tokens"] > 0
    assert wrapped.last_outer["error"] is None
    wrapped.detach()
    _ = original


def test_outer_json_size_and_required_fields(tmp_path: Any) -> None:
    model = FakeModel()
    wrapped = wrap(model)
    wrapped.generate("Explain why the sky is blue", seed=42)
    assert wrapped.last_outer is not None
    for field in REQUIRED_OUTER_FIELDS:
        assert field in wrapped.last_outer, f"missing required field {field}"
    assert wrapped.last_outer["flags"]["ring_overflow"] is False
    path = str(tmp_path / "outer.json")
    wrapped.save_outer(path)
    size = os.path.getsize(path)
    assert 1000 <= size <= 7000, f"outer.json {size}B not ~3KB"
    with open(path, encoding="utf-8") as fh:
        parsed = json.load(fh)
    for field in REQUIRED_OUTER_FIELDS:
        assert field in parsed
    wrapped.detach()


def test_detach_restores_and_idempotent() -> None:
    # NOTE: bound methods compare by value (==), not identity (is): each
    # attribute access builds a new object, so == is the exactness check.
    model = FakeModel()
    original = model.generate
    wrapped = wrap(model)
    assert model.generate != original
    wrapped.detach()
    assert model.generate == original
    wrapped.detach()  # idempotent: second call is a safe no-op
    wrapped.detach()
    assert model.generate == original
    assert wrapped.detached is True


def test_redactor_hook_point_masks_but_default_raw() -> None:
    model = FakeModel()
    plain = wrap(model)
    plain.generate("my ssn is 123", seed=42)
    assert plain.last_outer is not None
    assert plain.last_outer["prompt"]["text"] == "my ssn is 123"
    assert plain.last_outer["versions"]["redactor"] == "none"
    plain.detach()

    def own(prompt: str) -> str:
        return "[REDACTED]" if "ssn" in prompt.lower() else prompt

    model2 = FakeModel()
    masked = wrap(model2, redactor=own, capture_logits=True, seed=7)
    masked.generate("my ssn is 123", seed=7)
    assert masked.last_outer is not None
    assert masked.last_outer["prompt"]["text"] == "[REDACTED]"
    assert masked.last_outer["versions"]["redactor"] == "custom"
    masked.detach()


def test_ring_overflow_flag_path() -> None:
    model = FakeModel()
    wrapped = wrap(model)
    wrapped.ring_full = True  # simulate a full ring: flag, never stall
    wrapped.generate("Hello", seed=42)
    assert wrapped.last_outer is not None
    assert wrapped.last_outer["flags"]["ring_overflow"] is True
    wrapped.detach()


def test_wrapper_never_kills_caller_on_hook_error() -> None:
    model = FakeModel()
    original = model.generate

    def bad_redactor(prompt: str) -> str:
        raise RuntimeError("redactor blew up")

    wrapped = wrap(model, redactor=bad_redactor)
    out = wrapped.generate("Hello", seed=42)  # must still return output
    assert isinstance(out, FakeOutput)
    assert out.text == "echo:Hello"
    assert model.generate == original  # auto-detach restored the model
    wrapped.detach()


def test_watch_seals_outer_and_validates(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from blackbox import BlackBox, UnknownAddressError

    monkeypatch.setenv("BLACKBOX_DIR", str(tmp_path / "store"))
    before = (
        list((tmp_path / "store" / "runs").iterdir())
        if (tmp_path / "store" / "runs").is_dir()
        else []
    )
    with BlackBox.watch("watchlists/outer_only.yaml", tag={"cohort": "good"}) as ctx:
        assert ctx.run_id is None  # assigned on seal (__exit__)
    assert ctx.run_id is not None
    outer_path = os.path.join(str(tmp_path / "store"), "runs", ctx.run_id, "outer.json")
    seal_path = os.path.join(str(tmp_path / "store"), "runs", ctx.run_id, "seal.json")
    assert os.path.isfile(outer_path)
    assert os.path.isfile(seal_path)
    size = os.path.getsize(outer_path)
    assert 1000 <= size <= 7000, f"sealed outer.json {size}B not ~3KB"
    with open(outer_path, encoding="utf-8") as fh:
        outer = json.load(fh)
    for field in REQUIRED_OUTER_FIELDS:
        assert field in outer
    assert outer["tags"] == ["cohort=good"]
    assert BlackBox.verify(ctx.run_id, str(tmp_path / "store")) is True
    # Unknown address fails fast with nothing sealed.
    runs_before = set(os.listdir(os.path.join(str(tmp_path / "store"), "runs")))
    with pytest.raises(UnknownAddressError):
        with BlackBox.watch("watchlists/outer_only.yaml", add=["NOPE-NOT-AN-ADDRESS"]):
            pass
    runs_after = set(os.listdir(os.path.join(str(tmp_path / "store"), "runs")))
    assert runs_before == runs_after
    assert before is not None


def test_wrap_rejects_model_without_generate() -> None:
    with pytest.raises(ValueError, match="generate"):
        wrap(object())


def test_output_shapes_none_dict_and_bad_seed() -> None:
    class _M:
        def __init__(self, out: Any) -> None:
            self.out = out

        def generate(self, prompt: str, seed: int = 42, **kwargs: Any) -> Any:
            _ = (prompt, seed, kwargs)
            return self.out

    assert wrap(_M(None)).generate("hi", seed=1) is None
    assert wrap(_M({"text": "dict-out"})).generate("hi") == {"text": "dict-out"}
    assert wrap(_M({"other": 1})).generate("hi") == {"other": 1}
    m = _M("ok")
    w = wrap(m)
    assert w.generate("hi", seed="not-a-number") == "ok"  # bad seed -> 42, never raises


def test_two_tuple_redactor_and_save_outer(tmp_path: Any) -> None:
    class _M:
        def generate(self, prompt: str, seed: int = 42, **kwargs: Any) -> str:
            _ = (seed, kwargs)
            return "raw:" + prompt

    redactor = lambda p, toks: ("[REDACTED]", [])  # noqa: E731 (test hook shape)
    w = wrap(_M(), redactor=redactor)
    assert w.generate("secret") == "raw:secret"  # output unchanged, recording masked
    assert w.last_outer is not None and w.last_outer["prompt"]["text"] == "[REDACTED]"
    target = os.path.join(str(tmp_path), "outer.json")
    assert w.save_outer(target) == target
    assert os.path.isfile(target)
    with pytest.raises(ValueError, match="no generate"):
        wrap(_M()).save_outer(os.path.join(str(tmp_path), "empty.json"))


def test_model_error_seals_error_outer_and_reraises() -> None:
    class _Boom:
        def generate(self, prompt: str, seed: int = 42, **kwargs: Any) -> str:
            _ = (prompt, seed, kwargs)
            raise RuntimeError("boom")

    w = wrap(_Boom())
    with pytest.raises(RuntimeError, match="boom"):
        w.generate("hi")
    assert w.last_outer is not None
    assert w.last_outer["error"]["code"] == "model_error"


def test_getattr_passthrough() -> None:
    model = FakeModel()
    w = wrap(model)
    assert w.calls == model.calls  # attribute delegates to the wrapped model
    with pytest.raises(AttributeError):
        _ = w._missing_private  # type: ignore[attr-defined]
