# VIEWER-REACT-CONTRACT — future community work (no code, by design)

> Status: **blueprint only — `good-first-issue`, no SLA.** The normative v1
> viewer is CLI + static HTML (`blackbox/viewer.py`, spec §VIEW-*). If the two
> viewers ever disagree, the static viewer is right by definition. Do not
> write React code until you have read this whole note.

## Why this file exists instead of code

A second viewer is a liability unless it obeys the contract below. This note
preserves the design so a future contributor can build it correctly — or the
owner can confirm nobody ever needs it.

## The contract (all five rules are mandatory)

1. **Same files, no new backend.** Read only sealed store files
   (`outer.json`, `deep_index.json`, Cold slices via the window-only rule —
   never the full `deep.bin`) and the JSON the CLI already emits. No server,
   no database, no new query language.
2. **Same semantics.** Seal badge meanings, `null`/`--`/grey for unwatched,
   `co-occurrence, not causality` label, float32 logical values — identical
   to `viewer.py`. Copy the wording, not just the values.
3. **Never a gate.** Lives in `viewer-react/` with its own `package.json`.
   npm never leaks into the Python package, Dockerfile, CI, or any DoD.
   The folder must be deletable with zero loss to v1.0.
4. **One parity test.** For a fixed sealed run, rendered values + seal badge
   must equal the CLI output. That single test is the entire insurance
   against the two viewers diverging. It lives in `viewer-react/` and runs
   outside the Python suite.
5. **Community-maintained, no SLA.** Its own README states this. Rotting is
   acceptable; breaking rules 1–4 is not — a violating viewer gets deleted,
   not fixed.

## Suggested shape (when someone builds it)

- File readers (`outer.json` → header, `deep_index.json` → offsets, slice
  fetch for the viewed window) mirroring `viewer.load_run` / `token_table`.
- Four panels mirroring the static wireframe: PROMPT chips → token strip
  (`#token=N` hash) → DETAIL (values + sparklines + co-occurrence) → OUTPUT.
- Calm styling to match (white, 960px, system-ui); blue reserved for watched
  values, grey outline for unwatched, red banner on tamper (render anyway).

## Definition of done for the contributor

Parity test green on a sealed fixture run + `viewer-react/README.md` stating
rules 1–5 + a screenshot in the PR. No Python-side changes allowed.
