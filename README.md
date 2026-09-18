# AI Black Box — a flight recorder for AI

I'm one person building this. The whole idea in one sentence:

> **Don't tell researchers what the AI did. Give them the evidence needed to determine what the AI did.**

Like an aircraft black box: it doesn't decide why the plane crashed — it just preserves enough evidence for investigators to reconstruct what happened. Searchable, reproducible, hash-sealed.

![research](https://img.shields.io/badge/research-Linux_100%25_labs-NVIDIA_75--81%25-78.5%25_devs_Linux-blue)
<!-- 100% of top-10 AI labs run Linux, NVIDIA holds 75–81% of AI accelerator revenue, 78.5% of devs use Linux (Stack Overflow 2025). That's why Deep targets Linux+NVIDIA first — it's where the models actually live. -->

## The deal, straight

> Try in 5 min (pip, Outer), Deep in 15 min (Docker, Linux+NVIDIA). Outer <0.5%, Deep sampled ~5% — verify yourself with `benchmark.sh`; community numbers are the report. Never kills your job, never deletes without you, never traps your data (export zip + guide to adapt — we won't build your schema). Hash-verifiable, raw for research (don't run on private user data unless legal basis — plug your redactor if you must). Community-maintained, no SLA unless funded. Replay via seed+settings.

That's the promise. Everything below is just details.

## Try it

**5 minutes, Outer** — any OS, any model, pip, no Docker:

```bash
pip install ai-blackbox
with BlackBox.watch():   # context manager for research
    model.generate("hello")
# or: model = blackbox.torch.wrap(model)  # wrapper/decorator for prod
blackbox run --watch watchlists/pilot.yaml --tag cohort=good --seed 42
blackbox verify RUN-000184
```

**15 minutes, Deep** — Linux + Docker + NVIDIA + PyTorch, and you need the actual model weights:

```bash
blackbox scan /weights/Llama-3-8B          # 2-10s on CPU, once per checkpoint
blackbox run --watch watchlists/pilot.yaml  # CLI and Python do same
blackbox verify RUN-000184                  # hash-seal check
blackbox view RUN-000184 --open            # local HTML
df = blackbox.to_dataframe(RUN-000184)     # pandas in Jupyter — primary
```

Full flow: `scan → run → find → compare → verify → view → size → export --zip`.
Exact contracts for every flag: `spec.md` §A.1–A.13.

## Two modes — don't mix them up

| Mode | When | What you get | Overhead |
|------|------|--------------|----------|
| **Outer — always** | Any model: OpenAI/Claude/Gemini via API, or open weights locally | Prompt, output tokens, token order, logits where available, timing, cost, seed/settings, error/safety flags | **<0.5%**, ~3 KB/run |
| **Deep — only if you have weights** | Open weights locally (Llama, Mistral, etc.) or closed-lab running inside own infra | **Exact value per token** for *only* neurons/heads you declared before run, plus attention/precision you chose | **~5–7% for 1k watched**, ~20% for 100k watched (flagged RED, allowed) |

No weights, no Deep — by design, not by bug. Nobody can see inside a locked API, me included. Deep day 1 is **Linux + Docker + NVIDIA + PyTorch only**; on Windows/Mac you get Outer + CPU fallback + scan/search/viewer, and I won't claim Deep numbers there.

## What this is NOT (read before you complain)

- It records only what you **declared before the run** — exact value per token for watched neurons. Anyone promising "every neuron of any model" is selling petabytes.
- **No official 70B numbers day 1.** I don't have a 70B fleet. I ship `benchmark.sh` + a live estimator (`GREEN <50GB / YELLOW 50–1000GB / RED >1000GB`) and the `benchmarks/community/` numbers **are** the report. Run it yourself, post your numbers.
- **Never auto-deletes, never auto-redacts.** The estimator and `retention_state.json` only flag; `retention_policy.json` is yours, deletion is yours.
- **Never traps your data.** `blackbox export --zip` hands you plain files (`outer.json`, `deep.bin`, `deep_index.json`, `seal.json`). `docs/FORMAT_GUIDE.md` shows how to reshape them — but I won't build your schema for you.
- **Community-maintained, no SLA, no PR-review promise unless funded.** Sponsor/grant → we talk SLA. Otherwise it's me plus whoever shows up. MIT.
- Hash chain is local v1 (no HSM/Sigstore). `blackbox verify` fails loudly on edit — that's the whole tamper story.
- Viewer v1 is CLI + static HTML. No React, no npm. It opens with a double-click and works offline, which is the point.

## One serious thing: privacy

**User prompts and tokens are private — accessing them without legal basis is illegal, full stop.** This box is for synthetic/consented eval sets: you write the prompts, you run them, you study the traces. **Do not point it at private user traffic** unless the law where you operate says you can. Raw save for research; if the law *does* allow prod logging, plug your own `redactor=` hook (I provide the hook point, not the filter — your law, your filter).

## Your disk, your call

The box **never auto-deletes**. `blackbox size --by-model` shows usage + `GREEN/YELLOW/RED`; `retention_policy.json` says `"auto_delete": "never"` and means it. You decide, you delete.

## Who maintains this? Me. Possibly nobody, eventually.

Solo project, MIT, no lock-in by construction: plain files, documented formats, export anytime. If I go quiet, everything you need to fork it, port it (`gpu_adapters/` is one file per backend), or adapt the format is already in the repo. That's deliberate — software that only works while its author is awake isn't infrastructure.

## What's where

```text
ai-blackbox/
├── README.md                 # this file
├── LICENSE (MIT)
├── pyproject.toml            # pip install ai-blackbox
├── Dockerfile                # pytorch/pytorch:2.4.0-cuda12.1-runtime
├── blackbox/                 # SDK: scan / watchlist / controller / tap / gpu_adapters / storage / search
├── docs/                     # FORMAT_GUIDE.md, PRESETS.md, ADAPTER_GUIDE.md, ARCHITECTURE.md, THREAT_MODEL.md
├── adapters/format/          # llama3.json, mistral.json
├── examples/                 # 01_outer_only_api.py, 02_deep_llama_watchlist.py, 03_compare_good_bad.py
├── benchmarks/community/     # RESULTS.md + template/env.txt (community numbers are the report)
└── spec.md / product.md (locked) / plan.md (build manual)
```

`product.md` is the locked source of truth, `spec.md` is exact behavior, `plan.md` is the build manual. I don't edit `product.md`. Ever.

## License

MIT — see `LICENSE`. Community-maintained, no SLA unless funded.

---

*Raw save for research — don't run on private user traffic unless legal basis; plug your redactor= if you must.*
