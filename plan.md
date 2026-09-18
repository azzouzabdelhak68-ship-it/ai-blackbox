# AI Black Box — plan.md (Full Build Manual)

> Truth sources: `product.md` (idea, 38,950 bytes, locked Sept 15) + `spec.md` (exact behavior, 61,068 bytes, RED>1000GB, calm viewer, DuckDB default).
> This file changes nothing about the idea. It is the step-by-step manual for turning idea → working product.
> How to read: plain English first, exact command second. Non-expert can read headings. Coder reads code blocks.

---

## CONTENTS

1. Big picture + loops + cost
2. Team, roles, how to work (human + AI helpers)
3. Skills — every helper, when, exact steps, what good output looks like
4. Resources + links — official docs, what page to open, why, fallback
5. Machine setup — Ubuntu, Docker, Python, CUDA checks (copy-paste)
6. Repo skeleton — every folder + every file + what goes inside
7. Phase 0 Bootstrap loops
8. Phase 1 Outer + Scan file-by-file (v0.1 shippable)
9. Phase 2 Deep + Sidecar + GPU file-by-file (v0.2)
10. Phase 3 Search + Viewer file-by-file (v0.3)
11. Phase 4 Polish + Publish (v1.0)
12. Methods explained simply (FSST, estimator, hash chain, two-phase search)
13. Testing playbook per phase (what to run, what must pass)
14. Troubleshooting — every common error + fix
15. Risks register (likelihood, damage, fix)
16. Warnings + Advice + Suggestions (from 10 supervisor promises)
17. Rules for coders + AI workers (must / must-not)
18. Checklists you can tick (per phase)
19. Open decisions + how to lock them
20. Appendix: size math, end-to-end demo, glossary

---

## 1. BIG PICTURE + TIMELINE + COST

**What we build in one sentence:** a flight recorder for AI that saves what the model was given, what it did inside (only the parts you asked to watch before the run), and what it answered — sealed so nobody can secretly edit it later, searchable across thousands of runs, viewable as PROMPT → Token → OUTPUT.

**Two modes (never confuse them):**
- **Outer — always.** Works with any model (OpenAI API, Claude API, Llama local). Saves prompt, output tokens, order, timing, seed/settings, cost. About 3KB per run. Overhead under 0.5%. No GPU needed. This is the 5-minute try.
- **Deep — only if you have the model weights on your own machine.** Saves exact numbers for neurons/heads you listed BEFORE the run. 1,000 watched × 500 tokens ≈ 1MB per run. Overhead ~5%. Needs Linux + Docker + NVIDIA + PyTorch. This is the 15-minute Docker path.

**Loops (agent work — no timeline, repeat until PASS, no stop):**

| Phase | Loops | Ships | Must demo (gate PASS) |
|---|---|---|---|
| 0 Bootstrap | Loop ×3: scaffold → gates → fix | folders + install works | `pip install -e .` + `blackbox --help` |
| 1 Outer + Scan | Loop ×3: draft → test → skeptic | v0.1 | Outer 3KB sealed + verify passes, Mac CPU OK |
| 2 Deep + Sidecar | Loop ×3: hooks → sidecar → bench | v0.2 | 1k×500 → 1MB exact, kill -9 still verifies |
| 3 Search + Viewer | Loop ×3: index → query → viewer | v0.3 | 10k query <2s, viewer Token17 0.92 |
| 4 Polish + Publish | Loop ×3: docs → tests → publish check | v1.0 GitHub (+PyPI) | clean Ubuntu install green |

Each loop: draft → 4 gates (consistency/honesty/implementability/completeness must contain PASS) → skeptic kill attempts → fix → next round. Max 2 fixes per gate, 3rd fail escalates with evidence. All 3 rounds run no-stop.

**Cost thinking:** GPU cost is not in this plan (you rent or own). Disk cost = estimator tells you before each run: `bytes = W × T × R × B × 1.15`. Example: 1k watched × 500 tokens × 10k runs × 2 bytes fp16 × 1.15 ≈ 11.5GB GREEN. 100k watched same = 1.15TB RED>1000GB needs `--force`.

**Suggestion:** start with TinyLlama 1.1B on CPU for Phase 1 (fast loop, free). Move to Llama-3-8B on rented A100/B200 only in Phase 2 final loop. Never rent 70B day 1 — we ship `benchmark.sh`, community numbers are the report.

---

## 2. TEAM, ROLES, HOW TO WORK

**Agents that do the work:** orchestrator + parallel file-workers (one file per worker) + gate checkers. You (owner) lock decisions only.

**Roles:**
- **Owner (you):** locks open decisions (MIT vs Apache, package name, viewer scope, PyPI timing). Reviews honesty (no overpromise). No coding needed.
- **Builder (engineer):** writes `blackbox/` code per Phase sections 7-11. Runs tests + benchmarks. Never edits `product.md`.
- **AI helpers (skills, §3):** do scaffolding, test drafts, doc drafts. Human reviews every AI output against spec.

**How to work (rules that prevent mess):**
1. One task = one file. Two workers never edit the same file at the same time.
2. Read `spec.md` section for that file before coding. If `product.md` philosophy and `spec.md` number disagree, ask owner — default: philosophy wins on why, spec wins on exact numbers.
3. Every code file gets a test file in the same loop. No test → not done.
4. Every Deep change gets a benchmark run in the same loop (`benchmark.sh --mode deep ...`). Numbers go in `benchmarks/community/`.
5. Max 2 fix tries per review comment. 3rd fail → stop, show evidence, ask owner. No silent loops.
6. Commit message style: `phase1: scan.py CPU header read + manifest write`. Small commits, one file each.

---

## 3. SKILLS — EVERY HELPER, WHEN, EXACT STEPS

### 3.1 `project-health` — USE FIRST (Phase 0 loop 1)

**Why:** creates clean project config, catches secrets, sets lint rules before code exists. Prevents 80% of publish-day pain.

**Steps:**
1. Run sweep on empty repo. It creates `opencode.json` (preset `python-patterns`), `.gitignore` (must include `manifests/*`, `blackbox_store/*`, `*.bin`, `__pycache__/`, `.venv/`), `pre-commit` with `ruff`+`mypy`.
2. Expected output: `blackbox --help` stub prints, `ruff check .` green, `mypy blackbox/` no errors on stubs, `git status` shows no secret (no `.env`, no key).
3. If it reports leaked secret → delete file, rotate key, re-run. Never commit secret to fix later.

**Warning:** do not skip. Teams that skip spend Phase 4 cleaning imports.

### 3.2 `python-patterns` — USE ALWAYS

**Why:** enforces SDK shape (typed, `py.typed`, `ruff` clean) so adapters (`nvidia.py` → `amd.py`) stay one-file ports.

**Rules it enforces:**
- Every public function typed: `def watch(watchlist: str, add: list[str] | None = None) -> ContextManager`.
- `ruff` line length 100, `mypy strict` on `blackbox/` (not on `examples/`).
- Hooks in `tap/hooks.py` only. No hook code in `wrapper.py`. No GPU code outside `gpu_adapters/`.

**Advice:** run `ruff check . && ruff format --check . && mypy blackbox/` before every commit. Add as `pre-commit` hook so you cannot forget.

### 3.3 `testing-strategy` — USE Phase 1-3

**What to ask it:** "Design tests for Outer wrapper (no GPU) + scan CPU + hooks async + seal verify + search two-phase."

**Expected plan it returns:**
- Unit (no GPU, run on Mac): `test_scan_cpu.py` (fake `config.json` + fake safetensors header → manifest asserts layers/hidden), `test_outer_wrapper.py` (fake model → outer.json 3KB asserts), `test_seal_verify.py` (edit one byte → verify FAIL).
- Integration (needs GPU, marked `pytest -m gpu`): `test_hooks_async.py` (1k watched × 10 tokens → deep.bin size asserts + no `synchronize()` in code grep), sidecar kill test (`kill -9` main → WAL prefix still verifies).
- Manual checklist per phase (see §18).

**Method:** `BLACKBOX_GPU=cpu pytest -q` must pass everywhere. `pytest -m gpu` only on Linux+NVIDIA.

### 3.4 `security-threat-model` — USE for Seal (Phase 2) + Publish (Phase 4)

**Ask:** "Threat model: main process vs sidecar, hash chain, tamper, disk full, redactor bypass."

**Must cover:**
- Attacker edits `outer.json` after seal → `verify` recompute fails (caught).
- Attacker kills main → WAL already fsync'd per token (prefix safe).
- Attacker fills disk → estimator RED + `retention_state.json` RED, never auto-delete (user deletes).
- Attacker bypasses redactor → hook identity in `outer.json:versions.redactor`, raw bytes sealed post-hook (auditable).
- v1 limits: local chain only, no HSM/Sigstore. Document, don't pretend.

### 3.5 `project-docs` — USE after Phase 2

**Ask:** "Generate ARCHITECTURE.md from blackbox/ + docs/FORMAT_GUIDE.md draft."

**Verify:** every module in §6 tree appears. Every spec section maps to a file. No invented modules.

### 3.6 SKIP list (do not install)

- `workers-best-practices` — Cloudflare Workers, we are Linux Docker. Installing wastes time.
- `vitest`, `tailwind`, `shadcn-ui` — JavaScript/React. Viewer v1 is Python CLI + static HTML. No `npm` in v1.

---

## 4. RESOURCES + LINKS (what to open, why, fallback)

> Only official docs below. If a link dies, search the title in quotes — content lives on official domain.

**PyTorch (hooks + streams + pinned) — Phase 1-2 core:**
- `register_forward_hook` + `Tensor.index_select` + `cuda.Stream` + `pin_memory`: https://pytorch.org/docs/stable/ — open, search each term. Why: hooks gather watched only, streams hide copy, pinned enables async. Fallback: `pip show torch` version → local docs match installed version.
- Advice: pin `torch>=2.4`. CUDA 12.1 runtime image `pytorch/pytorch:2.4.0-cuda12.1-runtime` for Docker. CUDA 12 vs 13 drift → use only `torch.cuda.*` stable API.

**NVIDIA CUDA concepts:** https://docs.nvidia.com/cuda/ — read streams + pinned memory pages only. You don't need kernel writing. Fallback: `nvidia-smi` on machine tells driver + GPU (B200/A100/H100).

**Docker GPUs:** https://docs.docker.com/engine/containers/resource_constraints/#gpu — Why: `--gpus all --shm-size=1g` required for Deep + ring. Fallback: `docker run --help | grep -i gpu shm`.

**Safetensors (scan without GPU):** https://github.com/huggingface/safetensors — Why: read weights header on CPU, never load full weights. Fallback: `config.json` alone gives layers/hidden/heads; header confirms checkpoint sha.

**YAML / NumPy / DuckDB:**
- https://pyyaml.org/ — watchlist DSL parse + validation.
- https://numpy.org/doc/ — `np.fromfile` decoder for `deep.bin` + reshape `[T, W]`.
- https://duckdb.org/docs/ — Hot Parquet column scan, no server. Fallback: JSONL shards alone work without DuckDB (slower, still correct).

**Test models:** start TinyLlama 1.1B (CPU, minutes), then Llama-3-8B (GPU, hours). Never 70B in plan. Community adds bigger later.

**Suggestion:** bookmark PyTorch `hook` + `Stream` pages. You will re-open them 20 times in Phase 2. Normal.

---

## 5. MACHINE SETUP (copy-paste, 30 minutes)

**A. Ubuntu Deep machine (rented or lab):**
```bash
# check
uname -a                # kernel 5.15+
docker --version        # 24+
nvidia-smi              # driver 535+, shows A100/H100/B200
python3 --version       # 3.10-3.12
# docker deep run template
docker run --gpus all --shm-size=1g \
  -v $BLACKBOX_DIR:/blackbox -v /weights:/weights \
  pytorch/pytorch:2.4.0-cuda12.1-runtime nvidia-smi
```
- `--shm-size=1g` is REQUIRED (ring lives in `/dev/shm/bb.ring`). Without it sidecar starves.
- `BLACKBOX_DIR` default `./blackbox_store`. Hot SSD MBs, Cold local dir or S3 mount under same root.

**B. Laptop (Mac/Windows, Outer + viewer only):**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
BLACKBOX_GPU=cpu pytest -q   # must pass without NVIDIA
blackbox view RUN-000184 --export-html ./run.html  # pure file read, no GPU
```

**Warning:** Windows/Mac Deep day 1 is NOT supported (honesty lock). Outer + scan + search + viewer work there. Deep needs Linux+NVIDIA. Document, don't hack.

---

## 6. REPO SKELETON (create in this order)

1. Folders: `blackbox/tap/ blackbox/gpu_adapters/ blackbox/storage/ blackbox/search/ docs/ adapters/format/ manifests/ blackbox_store/ examples/ tests/ benchmarks/community/`
2. Empty `__init__.py` in each `blackbox/` subfolder (makes imports work).
3. `pyproject.toml` — name `ai-blackbox`, import `blackbox`, deps `torch>=2.4 safetensors>=0.4 pyyaml>=6.0 numpy>=1.26` + optional `pyarrow>=15 or duckdb>=1.0`.
4. `.gitignore` — `manifests/* blackbox_store/* *.bin __pycache__/ .venv/ *.html` (viewer exports).
5. `Dockerfile` + `docker-compose.yml` (app + sidecar share `shm` volume).
6. `benchmark.sh` stub (full content Phase 4, stub prints `not implemented` in Phase 0 so CI passes).

**Method:** `pip install -e .` then `blackbox --help` must list 8 commands before any logic is written. If help doesn't print, stop — CLI wiring is broken, don't add features on broken wiring.

---

## 7. PHASE 0 BOOTSTRAP — LOOPS (no days)

**Loop 1 — config + skeleton:**
- `project-health` sweep → `opencode.json`, `.gitignore`, `pre-commit`. Verify `ruff` green on empty tree.
- Write `pyproject.toml` + `Dockerfile` + `compose` + `benchmark.sh` stub. `pip install -e .` works. `blackbox --help` prints 8 commands (stubs that print `not yet implemented, see plan.md Phase N` + exit 0 — better than crash).
- Check: `git status` clean, no secrets.

**Loop 2 — README + env checks:**
- Write `README.md` from product §10 honest line verbatim + research badge (Linux 100% labs, NVIDIA 75-81%, 78.5% devs Linux). Include 5-min Outer + 15-min Deep quickstart from product §1.
- Add `docs/FORMAT_GUIDE.md` stub (one page: "we document, you edit — export zip gives plain files").
- Verify Ubuntu checks (§5A) + laptop checks (§5B). Record `env.txt` in `benchmarks/community/template/`.

**Loop 3 — CI + DoD:**
- CI: `ruff + mypy + BLACKBOX_GPU=cpu pytest -q` green on GitHub Actions (ubuntu-latest + macos-latest matrix).
- DoD tick (§18 checklist Phase 0). Tag `v0.0-bootstrap`. No code logic yet — correct.

**Advice:** resist coding scan early. Clean skeleton saves rework later.

---

## 8. PHASE 1 OUTER + SCAN (v0.1 SHIPPABLE)

**Goal:** researcher pip-installs, scans Llama-3-8B on CPU in 3s, wraps model, gets sealed `outer.json` 3KB, verifies. Works Mac CPU.

**8.1 `scan.py` — CPU only, 2-10s, KB output:**
- Steps: open `<WEIGHTS>/config.json` → parse layers/hidden/heads/mlp/vocab → open first `*.safetensors` header (first 8 bytes = header length, then JSON) → compute `sha256` of header + config → build `manifest.json` per spec §FILE-manifest → write `manifests/M-001_C-48291/manifest.json` → print `M-001 ... 32L ... torch2.4 cuda12.1 B200`.
- MUST NOT import `torch.cuda`, MUST NOT need prompt, MUST NOT touch GPU. Test by `CUDA_VISIBLE_DEVICES="" python -m blackbox.scan`.
- Errors: missing `config.json` → E02 + hint `--api-model`. Exists same sha → E03 needs `--force`. Different sha → new dir.
- Stub mode: `scan --api-model gpt-4o-2026-03-15` → `{mode: outer_only}`. Deep on stub must fail closed E04.

**8.2 `tap/wrapper.py` — ~20 lines, 0.15ms:**
- Pattern: class `Wrapped(generate_fn)` stores original, on call records `t_start`, calls original, records tokens/logits/timing/seed, writes ring slot 0, returns output unchanged. `detach()` restores original, idempotent (call twice safe).
- Outer fields per spec: prompt/tokens/order/logits?, timing cudaEvents, seed/temp/top_p/sampler, cost, flags. All present or explicit `null` with reason.
- Privacy default raw. `redactor=` hook point only — never ship filter.

**8.3 `controller.py` (Phase 1 stub):** trigger `always` only. Validate watchlist vs manifest (unknown → E04 fail fast, nothing sealed). Estimator dry-run prints GREEN.

**8.4 `storage/layout.py` + `seal.py`:** write `$BLACKBOX_DIR/runs/RUN-xxx/outer.json` + `seal.json` + append `chain.jsonl`. Verify recomputes 5 steps per spec. Any 1-byte edit → FAIL exit 7.

**8.5 CLI wiring:** `argparse` or `click`, 8 subcommands from spec §A.1-A.8. Each prints exact success/error/exit per spec. `--json` prints single-line JSON + newline (for scripts).

**8.6 Tests + DoD:** `test_scan_cpu.py` (fake config + header, no GPU), `test_outer_wrapper.py` (fake model, asserts 3KB + detach restores), `test_seal_verify.py` (tamper 1 byte → FAIL). `benchmark.sh --mode outer --runs 1000` overhead <0.5%. Tag `v0.1`. Publish GitHub-only (no PyPI yet per open decision).

**Warnings:** don't start Deep early. Don't add Postgres. Don't add React. Ship small, honest.

---

## 9. PHASE 2 DEEP + SIDECAR + GPU (v0.2)

**Goal:** 1k×500 → 1MB exact, ~5% overhead, kill -9 safe, CPU fallback green.

**9.1 `watchlist.py`:** parse YAML per spec grammar, expand 7 presets + `extends` + `add:` patch, validate every address vs manifest (`N-24-0001..0100` inclusive ranges), seal patch into `outer.json:watchlist_ref`. Unknown → `UnknownAddressError` naming first bad + valid range.

**9.2 `tap/hooks.py` (hardest file — loop until gates PASS):**
- Only `register_forward_hook` on parent modules in watchlist (not every neuron). Per token: `gathered = activation.index_select(dim=-1, idx)` on GPU → `pinned.copy_(gathered, non_blocking=True)` on dedicated `copyStream` → record Event. Default stream never `synchronize()`.
- Banned: `tensor.cpu()` (0.8ms → 2x slowdown). Required: overlap copy with next MatMul (~80% hidden).
- Ring full → `ring_overflow=true`, drop Deep payload, keep Outer + seal. Hook throw → auto `detach()`, set `hook_detached`, continue run.

**9.3 `tap/sidecar.py`:** separate process/container polls `/dev/shm/bb.ring` mmap triple buffer → WAL append → `fsync` per token → hash `hash_i=...` → write `runs/RUN-xxx/`. Main SIGKILL → sealed prefix still verifies. LD_PRELOAD/eBPF rejected (document why).

**9.4 `gpu_adapters/`:** implement `base.py` 6 funcs, then `nvidia.py` (CUDA Stream+pin), `cpu.py` (`clone()` queue 0.005ms). `amd.py`/`apple.py` stubs that raise `NotImplementedError` with pointer to `ADAPTER_GUIDE.md`. Group watchlist by module, idx to GPU once, double-buffered pinned, Event per group. int8 → dequant on GPU, store logical + `dtype_meta`.

**9.5 Estimator + Dockerfile:** `bytes=W*T*R*B*1.15`, table GREEN <50 YELLOW 50-1000 RED >1000 requires `--force`. `deep.bin` row-major `[T,W]` LE + `deep_index.json` offsets. Docker `--gpus all --shm-size=1g`.

**DoD:** bench log committed, `verify` after `kill -9` passes, `BLACKBOX_GPU=cpu pytest` green, `deep.bin` decoder `np.fromfile(...).reshape(T,W)` asserts size.

**Advice:** get 10-token run exact first, then 500. Debug small. Profile with `nsys` only if overhead >10% — usually a stray `synchronize()`.

---

## 10. PHASE 3 SEARCH + VIEWER (v0.3)

**Goal:** 10k query <2s laptop, viewer Token17 0.92 with calm style.

**10.1 `search/index.py`:** on each seal, append Hot `meta.json` + `shard-*.jsonl` (5MB/10k: min/max/mean/p95/max_window[10]/firing_rate/tags/sha). Optional Parquet via DuckDB (default, no server). Cold stays `deep.bin` mmap. Document `search-adapters/` contract (`hot_filter` + `cold_verify`) for SQLite/MySQL community ports. Presets unaffected.

**10.2 `search/query.py`:** EBNF parser per spec, validate vs manifest before Hot scan (typo → E02 with valid range, never silent empty). Two-phase: Hot parallel threadpool → candidates → Cold mmap exact → rows with `hash_seal_verified`. `COMPARE GROUPS` runs two FINDs then aggregates mean/p95/firing_rate + histogram.

**10.3 Viewer CLI + `to_dataframe`:** CLI table (default 32 tokens × 8 addrs, `--` grey unwatched, red badge on tamper) + `--json` exact. Dataframe columns/dtypes per spec, float32 logical, unwatched `NaN`, broken seal → warning + `seal_ok=False`.

**10.4 HTML:** single self-contained file, stdlib `http.server` for `--open`, no npm. Layout header/badge → PROMPT chips → Token strip click `#token=17` → DETAIL values + sparkline + co-occurrence (label not causality) → OUTPUT → Prev/Next + ←/→. Calm style: white 960px system-ui 14-16px lh1.5 flat borders. Outer-only → notice, no Deep. Tamper → red banner but render.

**DoD:** conformance queries parse, 1k-run search measured (Hot ms + slices), viewer export 41KB opens double-click no server.

---

## 11. PHASE 4 POLISH + PUBLISH (v1.0)

1. Docs: `FORMAT_GUIDE.md` (export zip → edit to your schema), `PRESETS.md`, `ADAPTER_GUIDE.md` (nvidia→amd HIP streams / apple MPS blit, keep `index_select`).
2. `project-docs` → `ARCHITECTURE.md`. Coverage ≥80% Outer. `security-threat-model` sign-off.
3. `benchmark.sh` full: `--mode outer/deep`, prints `mode,W,T,R,overhead_pct,ms_baseline,ms_tapped,GB,flag,torch,cuda,gpu,os` JSON + appends `benchmarks/community/RESULTS.md`. CI gates only Outer CPU pass, never numbers.
4. Examples 01/02/03 exact I/O per spec. README honest limits + footer privacy line + viewer footer same.
5. Publish: GitHub tag `v1.0`, Docker Hub image, PyPI `ai-blackbox` (GitHub-only v0.1 before). Test clean Ubuntu 24.04 `pip install ai-blackbox` + `docker pull` + `pytest` + `verify` green.
6. Open decisions to lock with owner: MIT, `ai-blackbox`, CLI+HTML, GitHub→PyPI. One-line each, no long debate.

---

## 12. METHODS EXPLAINED SIMPLY

**FSST (wrapper + hooks-async + sidecar):** outer shim notes prompt/answer at door (0.15ms). Deep hooks photocopy only watched numbers on GPU, slide copy to side tray on separate belt while next math runs (hidden). Sidecar in next room files + seals each page, survives fire (kill) because pages already filed.

**Estimator:** `W watched × T tokens × R runs × B bytes × 1.15 overhead`. 1k×500×10k×2×1.15 ≈ 11.5GB GREEN. Table in §1. RED>1000 needs `--force` — intentional friction for 1TB accidents.

**Hash chain:** `seal = SHA(prev + SHA(outer) + SHA(deep) + replay)`. Change 1 byte → recompute mismatch → FAIL. Chain links runs in order. Genesis prev = 64 zeros. Verify walks chain.

**Two-phase search:** index cards (5MB) → candidates (100) → open only those files (100MB not 10GB). 100-1000x less reading.

---

## 13. TESTING PLAYBOOK

- **Phase 0:** `ruff + mypy + pytest collects` green. No logic.
- **Phase 1:** `BLACKBOX_GPU=cpu pytest -q` all pass Mac. Tamper test must FAIL verify. Outer overhead <0.5%.
- **Phase 2:** `pytest -m gpu` Linux+NVIDIA. Kill test: `run & kill -9 main; verify` passes prefix. Overhead 1k ~5-7%. `cpu` fallback passes without GPU.
- **Phase 3:** 1k-run search <1s, 10k <2s laptop. Viewer opens no server. Dataframe dtypes assert `int32/float32`.
- **Always:** `python -m json.tool` every example JSON. `grep -rn synchronize blackbox/tap/hooks.py` must be empty (except comment banning it).

---

## 14. TROUBLESHOOTING

| Symptom | Cause | Fix |
|---|---|---|
| `E04 unknown address` before run | typo or old manifest after weight change | open `manifest.json` inventory, fix name or re-`scan` (old watchlists invalid on new sha) |
| `E05/flag RED require --force` | estimate >1000GB | add `--force` intentionally OR reduce W / `every_n:10` / R |
| `ring_overflow` flag | ring 1g too small for burst | raise `--shm-size=2g`, or narrower window |
| `CUDA out of memory` main | hooks hold ref | `detach()` in `finally`, double-buffered pinned not GPU; re-run estimator smaller W |
| `verify FAIL deep sha` | edited file or partial copy | `verify` names file; restore from backup or re-run with same seed |
| `viewer blank` | tried to embed full deep.bin | viewer embeds window slice only per spec; use `--tokens/--addrs` slice |
| `mypy hook error` | untyped hook | type as `Tensor` → `Tensor`, no `Any` |
| `docker no GPU` | forgot `--gpus all` | re-run with flags in §5A |
| `benchmark slow >20%` | stray `synchronize()` or `tensor.cpu()` | grep + remove, keep `copyStream` async |

---

## 15. RISKS REGISTER

| Risk | Likelihood | Damage | Mitigation |
|---|---|---|---|
| 100k watchlist 20% slow | High if user greedy | Medium | docs `every_n:10`, estimator RED>1000, bench published |
| API models never Deep | Certain | Low | honest README two modes, stub manifest fails closed |
| Disk fills (TBs) | Medium | High | `retention_state.json` flag, user policy, never auto-delete |
| CUDA 12/13 drift | Medium | Medium | only `torch.cuda.*` stable, adapter detects version |
| Sidecar dies | Low | Medium | WAL fsync per token, verify prefix, resume hint E1001 |
| Privacy misuse on user traffic | Low but severe | Severe | README + viewer footer warning, `redactor=` hook only, synthetic/consented sets |
| Scope creep (React, Postgres, per-user schema) | High | High | REJECT per plan — CLI+HTML, DuckDB default + adapter guide, FORMAT_GUIDE not builds |

---

## 16. WARNINGS + ADVICE + SUGGESTIONS

**Warnings (from 10 promises, must appear in README + viewer footer):**
- User data private/illegal unless law allows. Box for synthetic/consented eval sets. Don't run on private traffic. Raw save for research.
- No official 70B/B200 numbers day 1. `benchmark.sh` + community dir ARE the report.
- Deep Linux+NVIDIA day 1. Windows/Mac Outer only + CPU fallback.
- Community-maintained, no SLA/PR promise unless funded. MIT.
- Never auto-delete/redact. Flag only. Export zip plain files, adapt via guide.

**Advice:** commit bench logs; review AI diffs line-by-line; demo Outer to owner each loop (keeps honesty); keep `benchmarks/community/` tidy (one dir per run `date-gpu-model-W`).

**Suggestions:** good first issues for contributors: `amd.py` HIP port, `search-adapters/sqlite.py`, `mistral.json` format, calm-viewer a11y pass (keyboard + contrast). Label `good-first-issue`, no SLA promise.

---

## 17. RULES FOR CODERS + AI WORKERS

1. Read file's `spec.md` section before coding. Cite section in PR (`Implements §FILE-outer`).
2. One file per worker. Never overlap edits in parallel.
3. MUST: args table + example + error + exit. MUST NOT: new promises, new deps without owner ok.
4. JSON must parse. YAML must parse (`yamllint`). HTML must open double-click.
5. `ruff + mypy` green. No `print` debug left. No secret.
6. 2 fix tries max, 3rd → escalate with evidence.
7. `product.md` NEVER edited. `spec.md` edited only with owner `apply` word + re-validate checklist 38/38.

---

## 18. CHECKLISTS (tick before tagging)

**Phase 0:** `pip install -e .` OK / `blackbox --help` 8 cmds / `ruff+mypy` green / `.gitignore` has manifests+store / no secrets / CI ubuntu+mac green.
**Phase 1:** `scan` 3s CPU / `outer.json` 3KB / `verify` OK + tamper FAIL / Mac CPU pass / overhead <0.5% / v0.1 GitHub tag.
**Phase 2:** 1k×500 1MB exact / overhead ~5% log / kill-9 verifies / cpu fallback green / RED>1000 needs force / v0.2 tag.
**Phase 3:** 10k query <2s / viewer Token17 0.92 blue + grey + badge / dataframe dtypes / export html 41KB / v0.3 tag.
**Phase 4:** docs 3 guides / coverage 80% Outer / threat sign-off / examples I/O exact / clean Ubuntu green / PyPI + Docker Hub / v1.0.

---

## 19. OPEN DECISIONS (lock with one word each)

1. License MIT vs Apache-2.0 → **proposed MIT** (say `lock MIT`).
2. Package `ai-blackbox` import `blackbox` vs `ai-flight-recorder` → **proposed ai-blackbox**.
3. Viewer v1 CLI+static HTML vs React → **proposed CLI+HTML** (locked in spec, confirm).
4. Publish GitHub-only v0.1 → PyPI v1.0 → **proposed yes**.

---

## 20. APPENDIX

**Size math (T=500 fp16 B=2):** 1k×10k ≈ 11.5GB GREEN; 10k×10k ≈ 115GB YELLOW; 100k×10k ≈ 1.15TB RED needs force. Formula `W·T·R·B·1.15`.

**End-to-end demo (copy-paste when built):**
```bash
pip install ai-blackbox
blackbox scan /weights/Llama-3-8B
blackbox run --watch watchlists/pilot.yaml --tag cohort=good --seed 42 --dry-run
blackbox run --watch watchlists/pilot.yaml --tag cohort=good --seed 42
blackbox find "N-24-0001 AT token=17 > 0.8" --verify --limit 5
blackbox compare --good tag:cohort=good --bad tag:cohort=bad --neurons N-24-0001,A-24-07
blackbox verify RUN-000184 && blackbox view RUN-000184 --export-html ./RUN-000184.html
blackbox size --by-model && blackbox export --zip --run RUN-000184 --with-manifest
```

**Glossary:** Manifest = seat inventory before flight. Run ID = flight number. Outer = path+voice. Deep = engine sensors watched only. Watchlist = sensors picked before flight. Hash-seal = tamper seal. FSST = wrapper+async hooks+sidecar. Hot/Cold = index cards / full files. Adapter = plug for hardware/DB.

*End plan.md — build in order, keep promises, measure everything.*
