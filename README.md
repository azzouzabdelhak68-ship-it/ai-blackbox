# AI Black Box

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![PyTorch 2.4+](https://img.shields.io/badge/pytorch-2.4%2B-ee4c2c)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status: v0.1](https://img.shields.io/badge/status-v0.1--alpha-lightgrey)]()

**Flight recorder for LLMs.** Don't tell researchers what the model did — give them tamper-evident evidence to figure it out themselves.

Like an aircraft black box: it doesn't judge the crash, it preserves the trace so investigators can reconstruct it. Searchable, reproducible, hash-sealed.

### Why this exists

Most LLM observability stops at prompts and outputs. If you want to know *what happened inside* — which heads fired, which neurons spiked at token 17 — you're wiring hooks by hand for every experiment. This box does it once, externally, without killing your job or your disk.

---

### Features

- **Two modes, one API** — `Outer` (any model via API or weights, ~3KB/run) and `Deep` (exact values for watched neurons/heads, only with weights)
- **Declare before you run** — pick neurons in a `watchlist.yaml` against a CPU-scanned manifest; validated before any GPU work
- **Zero-stall tap** — async `copyStream` gather, never blocks inference, never traps on `synchronize()`
- **Hash-sealed** — `SHA256(prev || outer || deep || replay)`, `blackbox verify` fails on any edit
- **Search, don't grep** — Hot/Cold two-phase search over thousands of runs (`find`, `compare`)
- **View anywhere** — CLI table, JSON, or self-contained HTML (no server, no npm)

---

### Installation

```bash
# Outer — any OS, no GPU needed
pip install ai-blackbox

# From source
git clone https://github.com/azzouzabdelhak68-ship-it/ai-blackbox
cd ai-blackbox
pip install -e ".[dev]"

# Deep — Linux + Docker + NVIDIA (weights on disk)
docker run --gpus all --shm-size=1g \
  -v $BLACKBOX_DIR:/blackbox -v /weights:/weights \
  pytorch/pytorch:2.4.0-cuda12.1-runtime nvidia-smi
```

Requires Python 3.10–3.12, PyTorch 2.4+, Docker 24+ for Deep.

---

### Quickstart

**Outer — works with any model (API or local):**

```python
from blackbox import BlackBox

with BlackBox.watch("watchlists/outer_only.yaml"):
    model.generate("Explain why the sky is blue", seed=42)
```

```bash
blackbox run --watch watchlists/outer_only.yaml --tag cohort=good --seed 42
blackbox verify RUN-000001
blackbox view RUN-000001 --export-html ./run.html
```

**Deep — open weights on your machine:**

```bash
blackbox scan /path/to/weights          # CPU scan, 2–10s, once per checkpoint
blackbox run --watch watchlists/pilot.yaml
blackbox find "N-24-0001 AT token=17 > 0.8" --verify
blackbox compare --good tag:cohort=good --bad tag:cohort=bad --neurons N-24-0001,A-24-07
```

```python
import blackbox
df = blackbox.to_dataframe("RUN-000184", tokens=(15, 20))  # pandas, float32 logical values
```

> Works with any checkpoint that has `config.json` + `safetensors` — tested with Llama, Mistral and similar families. Closed APIs (OpenAI, Claude, Gemini) get Outer only, by design.

---

### How it works

1. **Scan** — CPU reads `config.json` + weights header once → `manifest.json` (inventory of addressable components)
2. **Watch** — you declare a `watchlist.yaml` (presets like `pilot-sparse-1k` or explicit addresses); validated before the run
3. **Seal** — sidecar writes `outer.json` + `deep.bin` + `seal.json` + `chain.jsonl`; any 1-byte edit breaks `verify`

```
weights → manifest → watchlist → run (Outer + Deep) → sealed run → search / view
```

---

### CLI

| Command | What it does |
|---|---|
| `scan` | Inventory a checkpoint on CPU |
| `run` | Execute with a watchlist and seal the run |
| `find` | Search runs (prompts, tags, neuron predicates) |
| `compare` | Good vs bad cohort stats + histogram |
| `verify` | Check hash chain integrity |
| `view` | CLI table, JSON, or static HTML |
| `size` | Disk usage with GREEN/YELLOW/RED flags |
| `export` | Flat zip with plain files — no lock-in |

Full flag reference: [`spec.md`](spec.md) · Build manual: [`plan.md`](plan.md) · Format guide: [`docs/FORMAT_GUIDE.md`](docs/FORMAT_GUIDE.md)

---

### Notes

- Records only what you declared, not "every neuron" — exact value per token for watched addresses.
- Benchmarks are yours to run: `benchmark.sh` + estimator. No day-1 fleet numbers, community results are the report.
- Never auto-deletes or auto-redacts — flags only (`GREEN <50GB / YELLOW / RED >1000GB` requires `--force`).
- Raw traces for research on consented/synthetic data. Don't run on private user traffic without legal basis — plug your own `redactor=` hook if you must.

---

### License

MIT — see [LICENSE](LICENSE). Community-maintained, no SLA unless funded. Contributions welcome.
