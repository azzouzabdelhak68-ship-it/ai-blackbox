# AGENTS.md — AI Black Box (flight recorder for AI)

> README for agents. Standard: https://agents.md/ (60k+ repos, Linux Foundation).
> Stack: Python 3.10-3.12 + PyTorch 2.4 + Docker + NVIDIA CUDA. Viewer v1 = CLI + static HTML, no React, no npm.

## Truth files (read before any task)

| File | What it is | Rule |
|---|---|---|
| `product.md` | Idea + 10 supervisor promises, locked | READ ONLY — NEVER EDIT |
| `spec.md` | Exact CLI/API/schemas, RED>1000GB, calm viewer | READ; edit only on owner `apply` |
| `plan.md` | Loops-only build manual (no timeline) | Follow phase sections 7-11 |

## What to read before modifying each area

| Area | Read first |
|---|---|
| CLI + Python API | `spec.md` §A.1-A.13 |
| Storage/schemas/watchlist | `spec.md` §FILE-*/§WATCH-*/§CTRL-* |
| Hooks/sidecar/GPU | `spec.md` §TAP-*/§GPU-*, `plan.md` §9 |
| Search/viewer | `spec.md` §SEARCH-*/§VIEW-*, `plan.md` §10 |
| Docs/examples | `plan.md` §11, `docs/FORMAT_GUIDE.md` |

## Setup commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
blackbox --help   # must list 8 cmds before any logic change
docker run --gpus all --shm-size=1g -v $BLACKBOX_DIR:/blackbox -v /weights:/weights pytorch/pytorch:2.4.0-cuda12.1-runtime nvidia-smi
```

## Test commands

```bash
ruff check . && ruff format --check . && mypy blackbox/
BLACKBOX_GPU=cpu pytest -q            # must pass everywhere (no GPU)
pytest -m gpu -q                      # Linux+NVIDIA only
./benchmark.sh --mode outer --runs 1000
./benchmark.sh --mode deep --watch watchlists/pilot-sparse-1k.yaml --tokens 500 --runs 100
python -m json.tool runs/RUN-000184/outer.json   # every example JSON must parse
grep -rn "synchronize" blackbox/tap/hooks.py     # must be empty (async only)
```

## Code style (correct pattern)

```python
# Correct: typed, one file per concern, GPU code only in gpu_adapters/
def watch(watchlist: str, add: list[str] | None = None) -> ContextManager:
    ...
# Correct hook: gather watched only, async copy, never sync
gathered = activation.index_select(dim=-1, idx)  # GPU
pinned.copy_(gathered, non_blocking=True)        # copyStream, default stream never synchronize()
```

- Hooks live in `blackbox/tap/hooks.py` only. GPU code only in `blackbox/gpu_adapters/`. Types required, `py.typed`, line length 100.
- One task = one file. Two parallel workers MUST NOT touch the same file.

## Loop workflow (no timeline, repeat until PASS, no stop)

Each phase: Loop ×3 → draft → 4 gates → skeptic → fix → next round.
- Gates (each file must contain `PASS`): `gate_consistency/honesty/implementability/completeness.md`
- Skeptic must contain `kill attempt` lines + fixes.
- Estimator first: `bytes=W*T*R*B*1.15`, GREEN <50GB, YELLOW 50-1000GB, RED >1000GB needs `--force`.
- Hash verbatim: `hash_i=SHA256(prev||SHA256(outer)||SHA256(deep)||canonical(replay))`.
- Max 2 fixes per gate; 3rd fail → stop, show evidence, ask owner.

## Boundaries

✅ Always do
- Validate watchlist vs `manifest.json` before GPU work (unknown → E04 fail fast, nothing sealed)
- Flag `ring_overflow` / auto-`detach()` on hook throw — never stall or kill the job
- Add/update a test for every file changed; keep `outer.json` ~3KB, `deep.bin` row-major `[T,W]` LE
- Run `ruff+mypy+pytest` before every commit; conventional commits `phaseN: file + what`

⚠️ Ask first
- Change public CLI flags / API signatures / file schemas (spec change needs owner `apply`)
- Add a dependency (default rejects `triton/postgres/eBPF/React`)
- Touch `spec.md` / `plan.md` / publish (PyPI/Docker Hub)

🚫 Never do
- Edit `product.md` — locked source of truth
- Claim "records every neuron", publish day-1 70B numbers, auto-delete, auto-redact, mandate a redactor filter
- Commit secrets; push to main; run Deep on Windows/Mac (Outer only there); run on private user traffic without legal basis
- Use `tensor.cpu()` / `synchronize()` in hooks; open full `deep.bin` in viewer (slice only)

## Personas (what each produces, what it must not touch)

- `@drafter` — writes one section draft (A/B/C). Must not edit other sections or `product.md`.
- `@gate` — writes one `gate_*.md` with PASS/FAIL + evidence. Must not edit drafts, only flag.
- `@skeptic` — writes `skeptic_report.md` with kill attempts. Must not edit code, only report.
- `@docs` — writes `docs/` + examples. Never modifies `blackbox/`.

## Nested files (nearest wins)

- Root `AGENTS.md` (this file) — global rules above.
- `blackbox/tap/AGENTS.md` — hook async rules, banned sync calls.
- `blackbox/gpu_adapters/AGENTS.md` — 6-func interface, port guide.
- `blackbox/search/AGENTS.md` — Hot/Cold two-phase, DuckDB default + `search-adapters/` contract.

## Legacy symlinks (for Claude/Copilot/Cursor/Gemini)

```bash
ln -s AGENTS.md CLAUDE.md
ln -s AGENTS.md GEMINI.md
mkdir -p .github && ln -s ../AGENTS.md .github/copilot-instructions.md
```

*End AGENTS.md — keep under 150 lines; details live in spec/plan via tables above.*
