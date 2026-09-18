"""Placeholder tests — Phase 0 Loop 1 DoD (plan.md §7 Loop 3, §13)."""

from __future__ import annotations

from blackbox.cli import COMMANDS, build_parser


def test_cli_lists_eight_commands() -> None:
    """CLI must expose exactly the 8 spec §A.1-A.8 subcommands."""
    assert sorted(COMMANDS) == [
        "compare",
        "export",
        "find",
        "run",
        "scan",
        "size",
        "verify",
        "view",
    ]


def test_placeholder() -> None:
    """Minimal passing test so BLACKBOX_GPU=cpu pytest is green."""
    parser = build_parser()
    assert parser.prog == "blackbox"
    assert True
