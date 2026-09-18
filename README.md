# AI Black Box — Flight Recorder for AI

> **Don't tell researchers what the AI did. Give them the evidence needed to determine what the AI did.**

Like an aircraft black box: it doesn't decide why the plane crashed — it preserves enough evidence for investigators to reconstruct what happened, searchable, reproducible, hash-sealed.

![research](https://img.shields.io/badge/research-Linux_100%25_labs-NVIDIA_75--81%25-78.5%25_devs_Linux-blue)
<!-- Research badge (product §12): 100% of top-10 AI labs on Linux, NVIDIA 75–81% AI accelerator revenue, 78.5% devs use Linux (Stack Overflow 2025, 49k devs). -->

## Honest line (product §10, verbatim)

> Try in 5 min (pip, Outer), Deep in 15 min (Docker, Linux+NVIDIA). Outer <0.5%, Deep sampled ~5% — verify yourself with `benchmark.sh`; community numbers are the report. Never kills your job, never deletes without you, never traps your data (export zip + guide to adapt — we won't build your schema). Hash-verifiable, raw for research (don't run on private user data unless legal basis — plug your redactor if you must). Community-maintained, no SLA unless funded. Replay via seed+settings.

## Quickstart

### 5-min Outer (any OS, any model — pip, no Docker)

```bash
pip install ai-blackbox
with BlackBox.watch():   # context manager for research
    model.generate("hello")
# or: model = blackbox.torch.wrap(model)  # wrapper/decorator for prod
blackbox run --watch watchlists/pilot.yaml --tag cohort=good --seed 42
blackbox verify RUN-000184
```

### 15-min Deep (Linux + Docker + NVIDIA + PyTorch, open weights only)

```bash
blackbox scan /weights/Llama-3-8B          # 2-10s on CPU, once per checkpoint
blackbox run --watch watchlists/pilot.yaml  # CLI and Python do same
blackbox verify RUN-000184                  # hash-seal check
blackbox view RUN-000184 --open            # local HTML
df = blackbox.to_dataframe(RUN-000184)     # pandas in Jupyter — primary
```

Full flow: `scan → run → find → compare → verify → view → size → export --zip`.
See `spec.md` §A.1–A.13 for exact CLI/API contracts.

## Two modes

| Mode | When | What you get | Overhead |
|------|------|--------------|----------|
| **Outer — always** | Any model: OpenAI/Claude/Gemini via API, or open weights locally | Prompt, output tokens, token order, logits where available, timing, cost, seed/settings, error/safety flags | **<0.5%**, ~3 KB/run |
| **Deep — only if you have weights** | Open weights locally (Llama, Mistral, etc.) or closed-lab running inside own infra | **Exact value per token** for *only* neurons/heads you declared before run, plus attention/precision you chose | **~5–7% for 1k watched**, ~20% for 100k watched (flagged RED, allowed) |

Without weights, Deep is useless by design — no invention can see inside a locked API.
Deep day 1 = **Linux + Docker + NVIDIA + PyTorch only**. **Windows/Mac = Outer only**
(+ CPU fallback, scan, search, viewer — no Deep perf claims there).

## Honesty limits (what we don't promise)

- Records only what you **declared before the run** (exact value per token for watched). Never "every neuron of any model".
- **No official 70B/B200 numbers day 1.** Ship `benchmark.sh` + live estimator (`GREEN <50GB / YELLOW 50–1000GB / RED >1000GB`); `benchmarks/community/` numbers **are** the report.
- **Never auto-deletes, never auto-redacts.** Estimator + `retention_state.json` flag only (`GREEN/YELLOW/RED`); you own `retention_policy.json` and deletion.
- **Never traps your data.** `blackbox export --zip` gives plain files (`outer.json`, `deep.bin`, `deep_index.json`, `seal.json`); we document via `docs/FORMAT_GUIDE.md`, you edit — we won't build your schema.
- **Community-maintained, no SLA / no PR-review promise unless funded** (sponsor/grant → SLA). MIT.
- Hash chain is local v1 (no HSM/Sigstore). `blackbox verify` fails on edit.
- Viewer v1 = CLI + static HTML only (no React, no npm, no Tailwind).

## Privacy warning

**User data is private / illegal to access unless law says otherwise.** This Box is for
synthetic/consented eval sets. **Don't run on private user traffic** unless you have a legal
basis. Raw save for research; if law *does* allow prod logging, plug your own `redactor=` hook
(we provide the hook point, not a mandated filter).

## Retention: flag only

Box **never auto-deletes**. `blackbox size --by-model` shows `used_gb` + `GREEN/YELLOW/RED`;
`retention_policy.json` is user-owned (`"auto_delete": "never"`). You decide, you delete.

## Repo structure

```text
ai-blackbox/
├── README.md                 # this file (from product.md)
├── LICENSE (MIT)
├── pyproject.toml            # pip install ai-blackbox
├── Dockerfile                # pytorch/pytorch:2.4.0-cuda12.1-runtime
├── blackbox/                 # SDK: scan / watchlist / controller / tap / gpu_adapters / storage / search
├── docs/                     # FORMAT_GUIDE.md, PRESETS.md, ADAPTER_GUIDE.md
├── adapters/format/          # llama3.json, mistral.json
├── examples/                 # 01_outer_only_api.py, 02_deep_llama_watchlist.py, 03_compare_good_bad.py
├── benchmarks/community/     # RESULTS.md + template/env.txt (community numbers are the report)
└── spec.md / product.md (locked) / plan.md (build manual)
```

See `product.md` (locked source of truth), `spec.md` (exact behavior), `plan.md` (phased build manual).

## License

MIT — see `LICENSE`. Community-maintained, no SLA unless funded.

---

*Raw save for research — don't run on private user traffic unless legal basis; plug your redactor= if you must.*
