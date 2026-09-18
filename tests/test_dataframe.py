"""Dataframe contract (spec §A.12): columns, dtypes, NaN, seal warnings."""

from __future__ import annotations

import json
import os
import random
from typing import Any

import numpy as np
import pytest

from blackbox import to_dataframe
from blackbox.storage import layout as _layout
from blackbox.storage import seal as _seal
from blackbox.to_dataframe import SealBrokenWarning

SHA = "ab12cd34" * 8
WATCHED = [f"N-02-{i:04d}" for i in range(1, 5)]
T = 10
COLUMNS = ["run_id", "token_pos", "token", "logit", "address", "value", "watched", "seal_ok"]


def _outer(run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "model_id": "M-001",
        "checkpoint_sha256": SHA,
        "prompt": {"text": "hi", "tokens": [1, 2]},
        "output": {"text": "yo", "tokens": [3, 4, 5]},
        "replay": {"seed": 1, "temp": 0.0, "top_p": 1.0, "sampler": "greedy"},
    }


@pytest.fixture()
def dstore(tmp_path: object, monkeypatch: object) -> dict[str, str]:
    store = os.path.join(str(tmp_path), "store")
    monkeypatch.setenv("BLACKBOX_DIR", store)
    run_id, run_dir = _layout.allocate_run(store)
    rng = random.Random(4)
    mat = np.asarray(
        [[rng.uniform(-1.0, 1.0) for _ in WATCHED] for _ in range(T)], dtype=np.float32
    )
    _layout.write_deep_bin(run_dir, mat, "fp16")
    _layout.write_deep_index(
        run_dir, run_id, triggered=True, dtype="fp16", tokens=T, watched=WATCHED
    )
    _layout.write_outer_json(run_dir, _outer(run_id))
    _seal.seal_run(run_dir)
    outer_id, outer_dir = _layout.allocate_run(store)
    _layout.write_absent_deep_index(outer_dir, outer_id, watched=[])
    _layout.write_outer_json(outer_dir, _outer(outer_id))
    _seal.seal_run(outer_dir)
    return {"store": store, "deep": run_id, "outer": outer_id}


def test_outer_one_row_columns_dtypes(dstore: dict[str, str]) -> None:
    df = to_dataframe(dstore["deep"], what="outer")
    assert list(df.columns) == COLUMNS
    assert len(df) == 1
    assert str(df.dtypes["token_pos"]) == "int32"
    assert str(df.dtypes["value"]) == "float32"
    assert str(df.dtypes["watched"]) == "bool"
    assert bool(df.iloc[0]["seal_ok"]) is True


def test_deep_long_form_values(dstore: dict[str, str]) -> None:
    df = to_dataframe(dstore["deep"], tokens=(2, 5), addresses=["N-02-0001", "N-02-0003"])
    assert len(df) == 4 * 2  # slice tokens x addresses
    assert set(df["address"].tolist()) == {"N-02-0001", "N-02-0003"}
    assert df["token_pos"].tolist() == [2, 2, 3, 3, 4, 4, 5, 5]
    assert bool(df["watched"].all())
    assert str(df.dtypes["token_pos"]) == "int32"


def test_outer_plus_deep_broadcast(dstore: dict[str, str]) -> None:
    df = to_dataframe(dstore["deep"], what="outer+deep", tokens=(0, 1))
    assert len(df) == 2 * len(WATCHED)


def test_unwatched_nan_not_interpolated(dstore: dict[str, str]) -> None:
    df = to_dataframe(dstore["deep"], tokens=(0, 0), addresses=["N-02-0009"])
    assert len(df) == 1
    assert bool(df.iloc[0]["watched"]) is False
    assert float(df.iloc[0]["value"]) != float(df.iloc[0]["value"])  # NaN


def test_broken_seal_warns_not_drops(dstore: dict[str, str]) -> None:
    victim = os.path.join(dstore["store"], "runs", dstore["deep"], "outer.json")
    with open(victim, encoding="utf-8") as fh:
        outer = json.load(fh)
    outer["output"]["text"] = "edited"
    with open(victim, "w", encoding="utf-8") as fh:
        json.dump(outer, fh, indent=2, sort_keys=True)
    with pytest.warns(SealBrokenWarning):
        df = to_dataframe(dstore["deep"], what="outer")
    assert bool(df.iloc[0]["seal_ok"]) is False


def test_missing_run_bad_slice_outer_only(dstore: dict[str, str]) -> None:
    with pytest.raises(Exception, match="not found"):
        to_dataframe("RUN-999999", what="outer")
    with pytest.raises(ValueError, match="E02"):
        to_dataframe(dstore["deep"], tokens=(8, 3))
    with pytest.raises(ValueError, match="triggered=false"):
        to_dataframe(dstore["outer"], what="deep")
    with pytest.raises(ValueError, match="unknown what"):
        to_dataframe(dstore["deep"], what="sideways")
