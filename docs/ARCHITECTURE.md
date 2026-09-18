# ARCHITECTURE — generated from `blackbox/` (Phase 4, plan §11.2)

> Every module below exists in this repo. Spec references are the
> implementation contract; `product.md` is philosophy (read-only).

## Data flow (one run, front to back)

```
weights/ ──scan.py──▶ manifests/<M>_<sha8>/manifest.json   (CPU, 2–10s, once)
watchlists/*.yaml ──watchlist.py──▶ addresses[]
        │ controller.py: validate vs manifest (E04) → estimate (GREEN/YELLOW/RED)
        ▼
model.generate() ──tap/wrapper.py──▶ Outer (prompt/tokens/timing/seed, ~3KB)
        │ trigger armed? ──tap/hooks.py──▶ index_select watched cols (GPU)
        │                      │ copyStream → pinned (non-blocking, async)
        ▼                      ▼
ring (/dev/shm/bb.ring, triple buffer; tmp fallback on CPU hosts)
        │ tap/sidecar.py: poll → WAL append + fsync/token → seal
        ▼
blackbox_store/runs/RUN-00000N/{outer.json, deep.bin, deep_index.json, seal.json}
        + chain.jsonl (hash_i = SHA256(prev||SHA256(outer)||SHA256(deep)||canonical(replay)))
        ▼
search/index.py (Hot shards: min/max/mean/p95/firing_rate/win10)
        ▼
search/query.py (Hot filter → Cold mmap verify) → cli find/compare
viewer.py → cli view (table/JSON/static HTML) · to_dataframe.py (pandas/long form)
```

## Module map (file → role → spec)

| Module | Role | Spec |
|---|---|---|
| `__init__.py` | `BlackBox.watch/scan/verify`, `to_dataframe` alias, error twins, estimator + local seal fallback | §A.9/A.11/A.13, §CTRL-estimator |
| `__main__.py` | `python -m blackbox` entry (`raise SystemExit(main())` so exit codes propagate) | §A.1–A.8 |
| `cli.py` | 8 subcommands, argparse only, single-line errors, exits 0/2/3/4/5/6/7 | §A.1–A.8, §A.13, §ERRORS |
| `scan.py` | CPU header read (`config.json` + safetensors header) → `manifest.json`; `--api-model` stub | §A.1/A.11, §FILE-manifest |
| `watchlist.py` | YAML DSL + 7 presets + `extends`/`add` patch + validation (E04 names first bad) | §WATCH-grammar/presets |
| `controller.py` | triggers (`always`/`on_flag`/`window`/address gate), window gating, estimator `W·T·R·B·1.15` | §CTRL-triggers/estimator |
| `exceptions.py` | `ManifestExistsError/UnknownAddressError/CheckpointMismatchError/SizeFlagError/RunNotFoundError/SealBrokenError` → exit map | §A.13, §ERRORS |
| `tap/wrapper.py` | Outer proxy at `generate()` boundary (~0.15 ms), `redactor=` hook point, idempotent `detach()` | §A.10, §TAP-fsst, §REDACTOR |
| `tap/hooks.py` | **only hook code**: per-parent `register_forward_hook`, `index_select` + `copy_(non_blocking)`; ring-full→flag, throw→auto-detach; int8 dequant on device | §TAP-fsst |
| `tap/sidecar.py` | mmap triple ring → WAL + per-token fsync → seal; `attach`+`seal_prefix` crash recovery (E1001 hint) | §TAP-sidecar |
| `gpu_adapters/{base,nvidia,cpu,amd,apple}.py` | 6-func port contract; CUDA priority, CPU fallback, HIP/MPS community stubs | §GPU-adapters |
| `storage/layout.py` | run alloc, `outer.json`, `deep.bin` row-major `[T,W]` LE, `deep_index.json`, slice reads | §FILE-outer/deepbin/deepindex/retention |
| `storage/seal.py` | `seal_run` + `verify_run` (5 recompute steps + chain walk) | §FILE-seal |
| `search/index.py` | Hot shards + `meta.json`, parallel `hot_scan`, `reindex`, adapter seam | §SEARCH-tiers |
| `search/query.py` | EBNF grammar, two-phase `find_runs`, `compare_runs` pooled stats | §SEARCH-grammar |
| `to_dataframe.py` | `outer` 1-row / `deep` long-form / `outer+deep`; float32 logical, NaN unwatched, `SealBrokenWarning` | §A.12 |
| `viewer.py` | CLI table/JSON data + self-contained HTML (chips→strip→DETAIL→OUTPUT) | §VIEW-cli/html/partial |
| `benchmark.sh` | outer + deep methodology; JSON keys per §BENCH; appends `benchmarks/community/` | §BENCH |

## Key invariants (gates enforce)

- Validate watchlist vs manifest **before** any GPU work (E04, nothing sealed).
- Hooks never block: ring-full → `ring_overflow`, throw → auto-`detach()`; job never killed.
- Never auto-delete / auto-redact: flag only (`retention_state.json`), user-owned policy.
- Raw save for research; `redactor=` is a caller-owned hook point, never a shipped filter.
- Partial-but-precise: unwatched renders as null/`--`/grey, never interpolated.
- `product.md` is never edited; `spec.md` changes need owner `apply`.
