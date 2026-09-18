"""CLI Outer paths: scan/run/size/export/verify/find/compare errors (spec §A).

CPU-only; a tmp ``BLACKBOX_DIR`` per test via monkeypatch. Exercises the
user-visible contracts: exact exits, single-line errors, JSON parsing.
"""

from __future__ import annotations

import json
import os

from blackbox.cli import main


def _env(tmp_path: object, monkeypatch: object) -> str:
    store = os.path.join(str(tmp_path), "store")
    monkeypatch.setenv("BLACKBOX_DIR", store)
    return store


def test_scan_api_model_and_missing(tmp_path: object, monkeypatch: object, capsys: object) -> None:
    _env(tmp_path, monkeypatch)
    out = os.path.join(str(tmp_path), "manifests")
    assert main(["scan", "--api-model", "gpt-4o-test", "--output", out, "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["model_id"] == "M-EXT"
    assert main(["scan", os.path.join(str(tmp_path), "nope")]) == 2
    assert "E02" in capsys.readouterr().err


def test_run_dry_run_seals_nothing_limit_zero(
    tmp_path: object, monkeypatch: object, capsys: object
) -> None:
    _env(tmp_path, monkeypatch)
    assert main(["run", "--watch", "watchlists/outer_only.yaml", "--dry-run"]) == 0
    assert "GREEN" in capsys.readouterr().out
    assert main(["run", "--watch", "watchlists/outer_only.yaml", "--limit", "0"]) == 2


def test_run_seal_size_export_verify_all(
    tmp_path: object, monkeypatch: object, capsys: object
) -> None:
    _env(tmp_path, monkeypatch)
    assert (
        main(["run", "--watch", "watchlists/outer_only.yaml", "--tag", "cohort=good", "--json"])
        == 0
    )
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    sealed = json.loads(lines[-1])  # estimator line precedes the JSON row
    run_id = sealed["run"]
    assert main(["size", "--by-model"]) == 0
    assert "GREEN" in capsys.readouterr().out
    assert main(["size", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)
    target = os.path.join(str(tmp_path), "run.zip")
    assert main(["export", "--zip", "--run", run_id, "--out", target]) == 0
    assert os.path.isfile(target)
    assert main(["export", "--zip", "--run", "RUN-999999"]) == 6
    assert main(["verify", "--all"]) == 0
    assert "SUMMARY" in capsys.readouterr().out
    assert main(["verify", "RUN-999999"]) == 6


def test_find_compare_error_contracts(
    tmp_path: object, monkeypatch: object, capsys: object
) -> None:
    _env(tmp_path, monkeypatch)
    assert main(["find", "N-99-9999 AT token=0 > 0.5"]) == 4
    assert "E04" in capsys.readouterr().err
    assert main(["find", "not a query"]) == 2
    assert (
        main(
            [
                "compare",
                "--good",
                "tag:cohort=good",
                "--bad",
                "tag:cohort=bad",
                "--neurons",
                "N-99-9999",
            ]
        )
        == 4
    )
    assert (
        main(
            [
                "compare",
                "--good",
                "tag:cohort=good",
                "--bad",
                "tag:cohort=bad",
                "--neurons",
                "N-02-0001",
            ]
        )
        == 6
    )
    assert "E06" in capsys.readouterr().err
    assert main(["run", "--watch", "watchlists/nope.yaml"]) == 2
