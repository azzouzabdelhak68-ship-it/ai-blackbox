"""Search package: Hot/Cold two-phase index + query (spec §SEARCH-*, plan.md §10)."""

from __future__ import annotations

from blackbox.search.index import (
    append_hot,
    ensure_hot,
    hot_scan,
    iter_hot_entries,
    reindex,
)
from blackbox.search.query import compare_runs, find_runs, parse_query

__all__ = [
    "append_hot",
    "ensure_hot",
    "hot_scan",
    "iter_hot_entries",
    "reindex",
    "compare_runs",
    "find_runs",
    "parse_query",
]
