"""Typed errors with CLI exit-code mapping (spec §A.13, §ERRORS).

Exit map for later CLI use (check most-specific first):

- E02 usage/parse/missing-file -> ``ValueError`` / ``FileNotFoundError`` (exit 2)
- E03 manifest exists, needs ``--force`` -> ``ManifestExistsError`` (exit 3)
- E04 unknown address / checkpoint mismatch -> ``UnknownAddressError`` /
  ``CheckpointMismatchError`` (exit 4, fail fast, nothing sealed)
- E05 RED estimate without ``--force`` -> ``SizeFlagError`` (exit 5)
- E06 run/tag not found -> ``RunNotFoundError`` (exit 6)
- E07 seal FAIL (tamper) -> ``verify()`` returns ``False``; ``SealBrokenError``
  is the programmatic twin (exit 7)
"""

from __future__ import annotations


class ManifestExistsError(FileExistsError):
    """E03: manifest exists for same sha; retry with ``--force`` (exit 3)."""


class UnknownAddressError(ValueError):
    """E04: unknown address vs manifest; fail fast, nothing sealed (exit 4)."""


class CheckpointMismatchError(ValueError):
    """E04: checkpoint sha drift; re-scan, old watchlists invalid (exit 4)."""


class SizeFlagError(ValueError):
    """E05: RED estimate without ``--force``; nothing ran (exit 5)."""


class RunNotFoundError(FileNotFoundError):
    """E06: run/tag not found or empty compare group (exit 6)."""


class SealBrokenError(ValueError):
    """E07: seal mismatch (tamper); data still viewable with red badge (exit 7)."""
