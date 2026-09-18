# AI Black Box spec - ROUND 3 (merged A+B+C)

Source: product.md READ-ONLY. Checklist 38 items. Gates PASS. Round 3 of 3, NO-STOP.


# spec.md Section A — CLI + Python API Exact Behavior (Round 3 Draft, FINAL)

Scope: checklist items 1–12 → spec §CLI-scan/run/find/compare/verify/view/size/export +
§API-watch/wrap/scan/dataframe. Sources: product.md §1, §8 Phase0–2, §14, §16.
Normative keywords: MUST / MUST NOT / SHALL follow RFC 2119 sense within this section.
Honesty locks (normative): NO every-neuron promise (declare-before-run only, exact value per token
for watched); NO Day-1 70B fleet numbers (estimator + `benchmark.sh`; community dir is the report);
flag-only retention (MUST NOT auto-delete); raw save for research + caller-owned `redactor=` hook
point only (MUST NOT mandate filter); Deep Day 1 = Linux+Docker+NVIDIA+PyTorch only, Outer = any OS;
local hash chain v1 (no HSM/Sigstore); community-maintained, no SLA unless funded.
Store root `$BLACKBOX_DIR` (default `./blackbox_store`). Docker Deep needs `--gpus all --shm-size=1g`.
IDs: `RUN-000184`, `M-001`, `C-48291` (`sha256:abc…`). Exit codes §A.13 (normative).

---

## A.1 CLI `scan` (item 1 → §CLI-scan)

MUST read only `config.json` + weights header on CPU (MUST NOT touch GPU, MUST NOT need a prompt).

```bash
blackbox scan <WEIGHTS_PATH> [--output DIR] [--force] [--json] [--api-model NAME]
```

| Arg | Req | Default | Exact behavior |
|-----|-----|---------|----------------|
| `WEIGHTS_PATH` | yes | — | Dir containing `config.json` + `*.safetensors` header. Time 2–10 s. Output KB. |
| `--api-model NAME` | no | — | Alternate mode: `blackbox scan --api-model gpt-4o-2026-03-15` → stub manifest `{model: NAME, mode: outer_only}`. Deep SHALL fail closed on stub. |
| `--output DIR` | no | `manifests/M-001_C-48291/` | MUST create parents; MUST NOT overwrite without `--force`. |
| `--force` | no | off | Overwrite same-sha manifest. Different sha SHALL use a new dir, never overwrite. |
| `--json` | no | off | Stdout exactly `{"manifest":PATH,"model_id":"M-001","checkpoint_sha256":"abc…"}` + newline. |

Success exit 0:

```text
$ blackbox scan /weights/Llama-3-8B
Scan CPU 3.1s: config.json + safetensors header only (no GPU, no prompt).
Manifest → manifests/M-001_C-48291/manifest.json
M-001 Llama-3-8B | C-48291 sha256:abc123… | 32L hidden4096 32H MLP14336 | torch2.4 cuda12.1 | NVIDIA B200
```

Errors (stderr single line + exit):

| Case | stderr | Exit |
|------|--------|------|
| Missing `config.json` | `Error E02: no config.json in /weights/X. Hint: --api-model NAME for Outer-only stub.` | 2 |
| Exists same sha | `Error E03: manifest exists sha256:abc…. Use --force.` | 3 |
| Hash drift (at `run`) | `Error E04: checkpoint mismatch expected abc… got def…. Re-scan; old watchlists invalid.` | 4 |

## A.2 CLI `run` (item 2 → §CLI-run)

```bash
blackbox run --watch FILE [--tag K=V]... [--seed N] [--limit N] [--trigger SPEC]
             [--precision fp32|fp16|bf16] [--force] [--dry-run] [--json]
```

| Arg | Default | Exact behavior |
|-----|---------|----------------|
| `--watch FILE` | required | YAML validated vs manifest BEFORE first token; unknown address → fail fast E04, nothing sealed. |
| `--tag K=V` | none, repeatable | SHALL match regex `[a-z0-9_]+=…`; stored `outer.json:tags`; `compare` matches verbatim. |
| `--seed N` | 42 int | Sealed `replay.seed`; replay bit-identical on same HW or warn (full §REPLAY). |
| `--limit N` | all (N≥1) | `--limit 0` → E02. |
| `--trigger SPEC` | file `trigger:` | Override; sealed as patch. Allowed: `always`, `on_flag:bad_output`, `window:5s..10s`, `window:10..50`. Anything else → E02. |
| `--precision` | file `precision:` | Logical value; int8 SHALL be dequantized on GPU pre-copy (+`dtype_meta`). |
| `--force` | off | Admits RED. Without it RED SHALL NOT run (exit 5). Emission: `WARNING RED-flagged … proceeding --force (user-owned disk).` |
| `--dry-run` | off | Estimator only, exit 0, seals nothing. `--json` → `{"W":…,"T":…,"R":…,"bytes":…,"flag":"GREEN"}`. |
| `--json` | off | Per-run `{"run":"RUN-000184","hash":"9f2c…","outer_kb":3.1,"deep_mb":1.0,"flag":"GREEN"}`. |

Success exit 0:

```text
$ blackbox run --watch watchlists/pilot.yaml --tag cohort=good --seed 42
Estimator: W=1000 T=500 R=10000 B=2B(fp16) → 10.0GB ×1.15 = 11.5GB → GREEN (<50GB). Proceeding.
RUN-000184 sealed | outer 3.1KB deep 1.0MB | hash 9f2c… | chain OK
```

RED gate exit 5:

```text
Estimator: W=100000 T=500 R=10000 B=2B → 1.0TB ×1.15 = 1.15TB → RED (>1000GB). Require --force. Nothing ran.
```

## A.3 CLI `find` (item 3 → §CLI-find)

```bash
blackbox find "<QUERY>" [--limit N] [--json | --table] [--verify] [--tokens A..B]
```

Locked grammar: `ADDR AT token=N OP FLOAT` (`OP ∈ {>,>=,<,<=,==}`) | `ADDR AT tokens A..B AGG AGGF`
(`AGGF ∈ {mean,p95,max,max_window}`) | `prompt CONTAINS "literal"` | `checkpoint="sha256:…"`
joined by `AND`. `ADDR ∈ {N-LL-NNNN, A-LL-HH, LOGITS, EMBED}` validated vs manifest, else E04.
`--tokens A..B` ≡ inline range (narrows cold I/O; same result).

| Arg | Default |
|-----|---------|
| `--limit N` | 20 (1..1000) |
| `--json/--table` | `--table` |
| `--verify` | off → per-row `hash_seal_verified` via chain recompute |
| `--tokens` | none |

Success exit 0:

```text
$ blackbox find "N-24-0001 AT token=17 > 0.8 AND prompt CONTAINS \"hello\"" --limit 2 --verify
RUN-ID      N-24-0001@t17  SEAL
RUN-000184  0.92           verified=true
RUN-000191  0.85           verified=true
hot 8ms + cold 2 slices (two-phase: hot filter → cold verify)
```

JSON row: `{"run":"RUN-000184","address":"N-24-0001","token":17,"value":0.92,"hash_seal_verified":true}`.
Errors: E04 bad address (`Valid: N-01-0001..N-32-14336, A-01-01..A-32-32, LOGITS, EMBED.`); E02 grammar with caret.

## A.4 CLI `compare` (item 4 → §CLI-compare)

```bash
blackbox compare --good EXPR --bad EXPR --neurons LIST [--stat LIST] [--tokens A..B] [--limit N] [--json]
```

| Arg | Default | Exact behavior |
|-----|---------|----------------|
| `--good/--bad` | required | `tag:k=v` or `checkpoint=sha…`. SHALL be quoted if value has spaces. |
| `--neurons` | required CSV | Each validated vs manifest; first unknown → E04 naming it. |
| `--stat` | `mean,p95,firing_rate` | Subset of `{mean,p95,max,firing_rate}`; `firing_rate ≜ P(value>0.8)` locked v1. |
| `--tokens` | `15..20` incl | Stats window. |
| `--limit` | 10000/group | 1..100000. |
| `--json` | off | `[{"address":…,"tokens":"15..20","good":{"n":…,"mean":…,"p95":…,"firing_rate":…},"bad":{…}}]`. |

Success exit 0:

```text
$ blackbox compare --good tag:cohort=good --bad tag:cohort=bad --neurons N-24-0001,A-24-07 --tokens 15..20
N-24-0001 | good n=100 mean 0.21 p95 0.44 fire 0.04 | bad n=100 mean 0.63 p95 0.97 fire 0.41
A-24-07   | good n=100 mean 0.10 p95 0.20 fire 0.00 | bad n=100 mean 0.31 p95 0.70 fire 0.12
histogram good [####......] bad [......####] | hot ~10MB, 12 cold slices
```

Empty group → `Error E06: no runs with tag:cohort=good.` exit 6.

## A.5 CLI `verify` (item 5 → §CLI-verify)

```bash
blackbox verify [RUN-ID]... [--all] [--json] [--chain PATH]
```

`--chain` default `$BLACKBOX_DIR/chain.jsonl`. Single: block output. Batch/`--all`: per-run lines +
`SUMMARY: 3 OK, 1 FAIL`. ANY fail → exit 7 (even if some OK). Formula
`hash_i = SHA256(prev || SHA256(outer.json) || SHA256(deep.bin) || canonical(replay))` (schema §FILE-seal).

Success exit 0:

```text
$ blackbox verify RUN-000184
RUN-000184 chain OK | outer 9f2c… deep 44aa… | replay{seed 42,temp,top_p,sampler,torch2.4,cuda12.1,C-abc…} | prev 71ab… linked
```

FAIL exit 7:

```text
$ blackbox verify RUN-000184 RUN-000191
RUN-000184 FAIL: outer.json mismatch stored 9f2c… recomputed 11de… (edited after seal).
RUN-000191 chain OK
SUMMARY: 1 OK, 1 FAIL → exit 7
```

Unknown run → `Error E06: RUN-999999 not found in $BLACKBOX_DIR/runs/.` exit 6.

## A.6 CLI `view` (item 6 → §CLI-view)

```bash
blackbox view RUN-ID [--open] [--port N] [--export-html PATH] [--no-gpu] [--json]
```

| Arg | Default | Exact behavior |
|-----|---------|----------------|
| `--open` | off | Open `http://127.0.0.1:PORT/RUN-ID` in browser; SHALL NOT block (server persists). |
| `--port` | 8137 (1024..65535) | Bad port → E02. |
| `--export-html PATH` | off | Static file (e.g. 41KB); works with no server, no GPU. |
| `--no-gpu` | off | Documented no-op (viewer is CPU-only); kept for script portability. |
| `--json` | off | `{"run":"RUN-000184","tokens":501,"watched":1000,"seal":"OK"}`. |

Success exit 0: `Viewer http://127.0.0.1:8137/RUN-000184 | export ./RUN-000184.html 41KB | badge seal OK`.
Missing → E06 exit 6. Graph colors/semantics normative in §VIEW (blue watched+value, grey `not_watched`,
red broken seal); this CLI only launches/exports.

## A.7 CLI `size` (item 7 → §CLI-size)

```bash
blackbox size [--by-model | --by-run] [--json] [--dir PATH]
```

`--dir` default `$BLACKBOX_DIR`. Thresholds (projected bytes incl. 1.15 overhead):
GREEN <50GB, YELLOW 50–1000GB, RED >1000GB — same as estimator, SHALL match exactly.

Success exit 0:

```text
$ blackbox size --by-model
MODEL         RUNS  HOT    COLD   TOTAL  FLAG
M-001/C-48291  184  0.6MB  184MB  185MB  GREEN (<50GB)
retention: retention_policy.json (user-owned) — flag only, MUST NOT auto-delete.
```

`--by-run` first lines: `RUN-000184 3.1KB + 1.0MB GREEN`. `--json`: `[{"scope":"M-001/C-48291","runs":184,…}]`.

## A.8 CLI `export --zip` (item 8 → §CLI-export)

```bash
blackbox export --zip --run RUN-ID [--out PATH] [--with-manifest] [--with-index]
```

| Arg | Default | Exact behavior |
|-----|---------|----------------|
| `--run` | required | Exported run. Missing → E06. |
| `--out` | `./RUN-000184.zip` | Overwrite? SHALL require `--force`? — NO: v1 overwrites with `WARNING overwriting …` (explicit; retention lock covers deletes, not user-named export targets). |
| `--with-manifest` | off | Adds `manifest.json` + `FORMAT_GUIDE.md` pointer file. |
| `--with-index` | off | Adds `deep_index.json` even when deep absent (stub `{watched:[]}`). Default: included iff deep exists. |

Contents (flat zip, no proprietary DB): `outer.json`, `deep.bin`?, `deep_index.json`, `seal.json`
(+ `manifest.json` if flagged). Adapt via `FORMAT_GUIDE.md` — we document, you edit (no per-user builds).
Success exit 0: `RUN-000184 → ./RUN-000184.zip 1.0MB 4 files (sha 7c1d…).`

---

## A.9 API `BlackBox.watch` (item 9 → §API-watch)

```python
from blackbox import BlackBox
ctx = BlackBox.watch(
    watchlist: str,                       # e.g. "watchlists/cohort-A.yaml" — REQUIRED
    add: list[str] | None = None,         # default None
    window: dict | None = None,           # default None; {"start_token":int,"end_token":int,"every_n":int=1}
    trigger: str | None = None,           # default None→file; "always"|"on_flag:bad_output"|"window:5s..10s"|"window:A..B"
    precision: str | None = None,         # default None→file; "fp32"|"fp16"|"bf16"
    force: bool = False,                  # default False; admits RED, else SizeFlagError
    tag: dict[str, str] | None = None,    # default None; e.g. {"cohort":"good"}
)
with ctx:                                 # __enter__ validates + estimates; __exit__ seals
    model.generate(prompt, seed=42)
```

Precedence (normative): call kwargs > watchlist file > preset expansion > built-ins
(`trigger=always`, `window=all tokens every 1`, `precision=fp16`).
Merge SHALL be sealed as `outer.json:watchlist_ref{id, sha256, patch{add,window,trigger,precision}, tags}`.
Same file across 10k = comparable cohorts; per-run `add=`/`window=` = sealed patch (options lock).
`__enter__` MUST validate addrs vs manifest (raise `UnknownAddressError` naming first bad + valid range),
MUST run estimator (RED + not force → `SizeFlagError`, nothing sealed), MUST check checkpoint sha
(drift → `CheckpointMismatchError`). MUST NOT start streaming before validation passes.

Example (10k good vs 10k bad + probe):

```python
from blackbox import BlackBox
good, bad = load_sets()  # 10000 + 10000 prompts
for p in good:
    with BlackBox.watch("watchlists/cohort-A.yaml", tag={"cohort": "good"}):
        model.generate(p, seed=42)
for p in bad:
    with BlackBox.watch("watchlists/cohort-A.yaml", tag={"cohort": "bad"}):
        model.generate(p, seed=42)
with BlackBox.watch("watchlists/cohort-A.yaml", add=["N-24-9999"],
                     window={"start_token": 10, "end_token": 50, "every_n": 1},
                     tag={"cohort": "probe"}):
    model.generate("Probe prompt.", seed=42)
```

## A.10 API `bb.wrap` (item 10 → §API-wrap)

```python
import blackbox.torch as bb
wrapped = bb.wrap(
    model,
    redactor: Callable[[str], str] | None = None,  # default None → raw
    capture_logits: bool = True,                   # default True; False when unavailable
    seed: int | None = None,                       # default None → caller seed; sealed either way
)
wrapped.detach()  # MUST restore model exactly; MUST be idempotent
```

Outer fields per run (~3KB, SHALL all be present or explicit `null` with reason):
`prompt, output_tokens[], token_order[], logits/probs? (where available), timing{cudaEvents_ms},
seed/settings{seed,temp,top_p,sampler}, cost{tokens,ms}, error/safety_flags[]`.
Perf: ~0.15 ms/req, <0.5%. Resilience (normative): ring-full SHALL flag `ring_overflow` (MUST NOT stall);
hook exception SHALL auto-`detach()`; sidecar WAL fsync'd per token SHALL survive SIGKILL/OOM of main.
Privacy (normative): default raw (research on synthetic/consented sets); private user traffic ONLY with
legal basis; `redactor` is a hook point only — MUST NOT ship a mandated filter.

Example:

```python
import blackbox.torch as bb
m = bb.wrap(model)                       # 5-min Outer, any OS
m.generate("Hello", seed=42)
m.detach(); m.detach()                   # idempotent
own = lambda p: "[REDACTED]" if "ssn" in p.lower() else p   # caller-owned, only if law allows
m2 = bb.wrap(model, redactor=own, capture_logits=True, seed=7)
m2.generate("Account …", seed=7); m2.detach()
```

## A.11 API `BlackBox.scan` (item 11 → §API-scan)

```python
from blackbox import BlackBox
manifest: dict = BlackBox.scan(path: str, output: str | None = None, force: bool = False)
stub: dict = BlackBox.scan("gpt-4o-2026-03-15", api_model: bool = True)
```

Defaults: `output=None` → `manifests/<MODEL>_<C>/manifest.json`; `force=False` (same-sha exists →
`ManifestExistsError`; different sha → new dir, never overwrite).
Return dict keys (normative): `model, checkpoint_sha256, architecture{layers,hidden,heads,mlp_per_layer},
inventory{layers[],heads[],neurons_per_layer,embed,logits}, runtime{torch,cuda}, hardware`.
CPU-only 2–10 s. One-bit weight change → new sha → old watchlists MUST raise `CheckpointMismatchError`.

Example:

```python
from blackbox import BlackBox
m = BlackBox.scan("/weights/Llama-3-8B")
assert m["architecture"] == {"layers": 32, "hidden": 4096, "heads": 32, "mlp_per_layer": 14336}
print(m["checkpoint_sha256"], m["inventory"]["neurons_per_layer"])
stub = BlackBox.scan("gpt-4o-2026-03-15", api_model=True)
assert stub["mode"] == "outer_only"
```

## A.12 API `to_dataframe` (item 12 → §API-dataframe)

```python
import blackbox
df = blackbox.to_dataframe(
    run_id: str,                          # e.g. "RUN-000184" — REQUIRED
    what: str = "outer+deep",             # "outer"|"deep"|"outer+deep"
    tokens: tuple[int, int] | None = None,# default None=all; inclusive slice e.g. (15,20)
    addresses: list[str] | None = None,   # default None=all watched; validated vs manifest (E04→UnknownAddressError)
)
```

Semantics: `what="outer"` → 1 row (prompt/output/timing/cost/seal cols); `"deep"` → long form
tokens×watched; `"outer+deep"` → deep rows + outer cols broadcast.
Columns+dtypes (normative): `run_id[str] token_pos[int32] token[str] logit[float32] address[str]
value[float32] watched[bool] seal_ok[bool]`. File dtype may be fp16/bf16/fp32 row-major LE
(full §FILE-deepbin); frame SHALL expose float32 logical values. Partial-precise: default omits
unwatched (MUST NOT interpolate); requested-but-unwatched → row with `watched=False, value=NaN`.
Broken seal → `SealBrokenWarning` + `seal_ok=False` (MUST NOT silently drop). Missing run → `RunNotFoundError`.

Example (Jupyter primary path):

```python
import blackbox
df = blackbox.to_dataframe("RUN-000184", tokens=(15, 20), addresses=["N-24-0001", "A-24-07"])
assert str(df.dtypes["token_pos"]) == "int32"
df.pivot(index="token_pos", columns="address", values="value").plot(title="RUN-000184 tokens 15..20")
```

---

## A.13 Exit codes + exception map (normative)

| Code | CLI meaning | Python twin |
|------|-------------|-------------|
| 0 | Success (incl. `--dry-run`) | — (returns) |
| 2 | E02 usage/parse/missing-file/bad-grammar/bad-port/`--limit 0` | `ValueError` / `FileNotFoundError` |
| 3 | E03 manifest exists, needs `--force` | `ManifestExistsError` |
| 4 | E04 unknown address / checkpoint mismatch (fail fast, nothing sealed) | `UnknownAddressError` / `CheckpointMismatchError` |
| 5 | E05 RED estimate without `--force` (nothing ran) | `SizeFlagError` |
| 6 | E06 run/tag not found, empty compare group | `RunNotFoundError` |
| 7 | E07 seal FAIL (tamper; batch: any FAIL) | `SealBrokenWarning` (dataframe) / `verify()→False` |

End-to-end runnable (Outer, any OS / Deep, Linux+Docker+NVIDIA):

```bash
pip install ai-blackbox
blackbox scan /weights/Llama-3-8B                                   # CPU 3s → manifest
blackbox run --watch watchlists/pilot.yaml --tag cohort=good --seed 42 --dry-run
blackbox run --watch watchlists/pilot.yaml --tag cohort=good --seed 42
blackbox find "N-24-0001 AT token=17 > 0.8" --verify --limit 5
blackbox compare --good tag:cohort=good --bad tag:cohort=bad --neurons N-24-0001,A-24-07
blackbox verify RUN-000184 && blackbox view RUN-000184 --export-html ./RUN-000184.html \
  && blackbox size --by-model && blackbox export --zip --run RUN-000184 --with-manifest
```

```python
# Python parity (same behavior as CLI above)
from blackbox import BlackBox
import blackbox, blackbox.torch as bb
BlackBox.scan("/weights/Llama-3-8B")
m = bb.wrap(model)
with BlackBox.watch("watchlists/pilot.yaml", tag={"cohort": "good"}):
    m.generate("Hello", seed=42)
m.detach()
df = blackbox.to_dataframe("RUN-000184", tokens=(15, 20))
```

Honesty (normative restatement): estimator `bytes = W·T·R·B·1.15`; GREEN <50GB / YELLOW 50–1000GB /
RED >1000GB. NO official Day-1 70B/B200 fleet report — `benchmark.sh` + estimator; community
`/benchmarks/community/` IS the report. MUST NOT kill job (flag+detach), MUST NOT auto-delete
(flag only, user-owned policy), MUST NOT trap data (plain-file `--zip` + `FORMAT_GUIDE.md`;
no per-user schema builds). Local chain v1 (no HSM/Sigstore). Community-maintained, no SLA unless funded.
Traceability: A.1–A.8 ↔ checklist 1–8; A.9–A.12 ↔ 9–12.



# spec.md — Section B: Storage + Formats + Watchlist + Control + TAP + GPU

> Covers checklist items 13–25, 31–32. Source: `product.md` §8 (Phase 0–5), §10.5, §10.7.
> Store root: `$BLACKBOX_DIR` = `blackbox_store/`.

```
blackbox_store/
├── manifest/M-001__C-48291/manifest.json
├── runs/RUN-000184/
│   ├── outer.json
│   ├── deep.bin
│   ├── deep_index.json
│   └── seal.json
├── chain.jsonl
├── retention_state.json
├── retention_policy.json          # user-owned, never auto-enforced by delete
└── docs/FORMAT_GUIDE.md
```

---

## §FILE-outer — `runs/<RUN-ID>/outer.json` (item 13)

HOT, 2–5 KB, always written. Raw save (verbatim prompt/tokens); no mandated filter.

| Field | Type | Required | Notes |
|---|---|---|---|
| `run_id` | string | yes | `RUN-000184` |
| `model_id` | string | yes | `M-001` |
| `checkpoint_sha256` | string | yes | hex, must match manifest |
| `watchlist_ref` | object | yes | `{watchlist_id, watchlist_sha256, patch}` |
| `prompt` | object | yes | `{text, tokens:int[], tokenizer:string}` |
| `output` | object | yes | `{text, tokens:int[], finish_reason:string}` |
| `logits_summary` | object | no | `{top_k:int[][], available:bool}` — null for closed APIs |
| `timing` | object | yes | `{t_start_iso, t_end_iso, ms_per_token:float, cuda_event_ms:float}` |
| `sampling` | object | yes | `{seed:int, temp:float, top_p:float, sampler:string}` |
| `versions` | object | yes | `{torch, cuda, blackbox, adapter}` |
| `cost` | object | no | `{input_tokens, output_tokens, usd:float}` |
| `flags` | object | yes | `{bad_output:bool, ring_overflow:bool, truncated:bool}` |
| `tags` | string[] | no | e.g. `["good"]` for compare groups |
| `replay` | object | yes | see §REPLAY |
| `error` | object \| null | yes | `{code, message}` or null |

JSON Schema (abridged, normative for types):

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "outer.json",
  "type": "object",
  "required": ["run_id", "model_id", "checkpoint_sha256", "watchlist_ref", "prompt", "output", "timing", "sampling", "versions", "flags", "replay", "error"],
  "properties": {
    "run_id": { "type": "string", "pattern": "^RUN-[0-9]{6}$" },
    "model_id": { "type": "string" },
    "checkpoint_sha256": { "type": "string", "pattern": "^[0-9a-f]{64}$" },
    "watchlist_ref": {
      "type": "object",
      "required": ["watchlist_id", "watchlist_sha256"],
      "properties": {
        "watchlist_id": { "type": "string" },
        "watchlist_sha256": { "type": "string" },
        "patch": { "type": ["object", "null"] }
      }
    },
    "prompt": { "type": "object", "required": ["text", "tokens", "tokenizer"] },
    "output": { "type": "object", "required": ["text", "tokens", "finish_reason"] },
    "logits_summary": { "type": ["object", "null"] },
    "timing": { "type": "object", "required": ["t_start_iso", "t_end_iso", "ms_per_token"] },
    "sampling": { "type": "object", "required": ["seed", "temp", "top_p", "sampler"] },
    "versions": { "type": "object", "required": ["torch", "cuda", "blackbox", "adapter"] },
    "cost": { "type": ["object", "null"] },
    "flags": { "type": "object", "required": ["bad_output", "ring_overflow", "truncated"] },
    "tags": { "type": "array", "items": { "type": "string" } },
    "replay": { "type": "object" },
    "error": { "type": ["object", "null"] }
  }
}
```

Example (valid JSON):

```json
{
  "run_id": "RUN-000184",
  "model_id": "M-001",
  "checkpoint_sha256": "ab12cd34ab12cd34ab12cd34ab12cd34ab12cd34ab12cd34ab12cd34ab12cd34",
  "watchlist_ref": {
    "watchlist_id": "cohort-A-v3",
    "watchlist_sha256": "ee00ff11ee00ff11ee00ff11ee00ff11ee00ff11ee00ff11ee00ff11ee00ff11",
    "patch": { "add": ["N-24-9999"], "window": { "start_token": 10, "end_token": 50 } }
  },
  "prompt": { "text": "Hello world", "tokens": [15496, 995], "tokenizer": "llama3-bpe" },
  "output": { "text": "Hello back", "tokens": [15496, 2201], "finish_reason": "stop" },
  "logits_summary": { "available": true, "top_k": [[15496, 2201, 13]] },
  "timing": { "t_start_iso": "2026-09-15T10:00:00Z", "t_end_iso": "2026-09-15T10:00:02Z", "ms_per_token": 12.5, "cuda_event_ms": 11.9 },
  "sampling": { "seed": 42, "temp": 0.0, "top_p": 1.0, "sampler": "greedy" },
  "versions": { "torch": "2.4.0", "cuda": "12.1", "blackbox": "0.2.0", "adapter": "nvidia" },
  "cost": { "input_tokens": 2, "output_tokens": 2, "usd": 0.0001 },
  "flags": { "bad_output": false, "ring_overflow": false, "truncated": false },
  "tags": ["good"],
  "replay": {
    "seed": 42,
    "temp": 0.0,
    "top_p": 1.0,
    "sampler": "greedy",
    "cudnn_deterministic": true,
    "checkpoint_sha256": "ab12cd34ab12cd34ab12cd34ab12cd34ab12cd34ab12cd34ab12cd34ab12cd34",
    "versions": { "torch": "2.4.0", "cuda": "12.1" }
  },
  "error": null
}
```

---

## §FILE-deepbin — `runs/<RUN-ID>/deep.bin` (item 14)

COLD, optional (absent when `outer_only` or trigger never fired). Raw little-endian float array.

| Property | Value |
|---|---|
| Layout | row-major `float[T][W]` — token-major: token `t`, watched index `j` at byte `(t*W + j) * B` |
| `T` | actual emitted tokens in window (after `every_n` subsample) |
| `W` | `len(watched_addresses)` in `deep_index.json` order |
| dtypes | `fp32` B=4, `fp16` B=2, `bf16` B=2 (logical value; int8 weights dequantized on GPU before copy, see §GPU-adapters) |
| Endian | little-endian |
| Empty case | trigger never fired → file absent, `deep_index.json:{"triggered": false}` |

Size rule: `bytes = W * T * R * B * 1.15 overhead` (per-run file = `W*T*B`; `R` and 1.15 apply to estimator only).

Example descriptor (values live in `deep_index.json`, not in bin):

```json
{ "run_id": "RUN-000184", "dtype": "fp16", "shape": [500, 1000], "order": "row_major_token_major", "endian": "little", "bytes": 1000000 }
```

Decoder (normative):

```python
import numpy as np
W, T, B = 1000, 500, 2
a = np.fromfile("runs/RUN-000184/deep.bin", dtype=np.float16)  # fp32 -> np.float32
assert a.size == T * W
per_token = a.reshape(T, W)   # row t = token t, col j = deep_index order
```

---

## §FILE-deepindex — `runs/<RUN-ID>/deep_index.json` (item 15)

HOT pointer into `deep.bin`. Addresses validated against manifest at DECLARE time.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "deep_index.json",
  "type": "object",
  "required": ["run_id", "triggered", "dtype", "tokens", "watched"],
  "properties": {
    "run_id": { "type": "string" },
    "triggered": { "type": "boolean" },
    "dtype": { "enum": ["fp32", "fp16", "bf16"] },
    "tokens": { "type": "integer", "minimum": 0 },
    "every_n": { "type": "integer", "minimum": 1 },
    "watched": { "type": "array", "items": { "type": "string" } },
    "columns": {
      "type": "object",
      "additionalProperties": {
        "type": "object",
        "required": ["offset", "dtype", "shape"],
        "properties": {
          "offset": { "type": "integer", "minimum": 0 },
          "dtype": { "enum": ["fp32", "fp16", "bf16"] },
          "shape": { "type": "array", "items": { "type": "integer" } }
        }
      }
    }
  }
}
```

Example (valid JSON):

```json
{
  "run_id": "RUN-000184",
  "triggered": true,
  "dtype": "fp16",
  "tokens": 500,
  "every_n": 1,
  "window": { "start_token": 0, "end_token": 511 },
  "watched": ["A-17-04", "N-24-0001", "N-24-0002", "LOGITS"],
  "columns": {
    "A-17-04": { "offset": 0, "dtype": "fp16", "shape": [500] },
    "N-24-0001": { "offset": 1, "dtype": "fp16", "shape": [500] },
    "N-24-0002": { "offset": 2, "dtype": "fp16", "shape": [500] },
    "LOGITS": { "offset": 3, "dtype": "fp16", "shape": [500, 32000] }
  },
  "dtype_meta": { "stored": "fp16", "source": "bf16_dequantized_on_gpu", "endian": "little", "order": "row_major_token_major" }
}
```

---

## §FILE-manifest — `manifest/<M>__<C>/manifest.json` (item 16)

HOT, once per checkpoint, CPU scan output (KB). Address inventory; value never stored here (Q10 lock).

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "manifest.json",
  "type": "object",
  "required": ["model_id", "checkpoint_sha256", "architecture", "inventory", "runtime", "hardware", "adapter"],
  "properties": {
    "model_id": { "type": "string" },
    "checkpoint_sha256": { "type": "string", "pattern": "^[0-9a-f]{64}$" },
    "architecture": {
      "type": "object",
      "required": ["family", "layers", "hidden", "heads", "mlp_per_layer", "vocab"],
      "properties": {
        "family": { "type": "string" },
        "layers": { "type": "integer" },
        "hidden": { "type": "integer" },
        "heads": { "type": "integer" },
        "mlp_per_layer": { "type": "integer" },
        "vocab": { "type": "integer" }
      }
    },
    "inventory": { "type": "object" },
    "runtime": { "type": "object" },
    "hardware": { "type": "object" },
    "adapter": { "type": "object" }
  }
}
```

Example (valid JSON):

```json
{
  "model_id": "M-001",
  "model_name": "Llama-3-8B",
  "checkpoint_sha256": "ab12cd34ab12cd34ab12cd34ab12cd34ab12cd34ab12cd34ab12cd34ab12cd34",
  "architecture": { "family": "llama3", "layers": 32, "hidden": 4096, "heads": 32, "mlp_per_layer": 14336, "vocab": 128256 },
  "inventory": {
    "layers": ["L-01", "L-32"],
    "attention": "A-24-01..32",
    "neurons": "N-24-0001..14336",
    "specials": ["EMBED", "LOGITS", "NORM"]
  },
  "runtime": { "framework": "PyTorch 2.4.0", "cuda": "12.1" },
  "hardware": { "gpu": "NVIDIA B200", "detected_via": "nvidia-smi" },
  "adapter": { "family": "llama3", "layout": "row_major_float16" },
  "scan": { "source": "config.json + weights header", "gpu_used": false, "duration_s": 3.1 }
}
```

Closed-API scan yields minimal manifest: `{"model_id": "M-EXT", "model_name": "gpt-4o-2026-03-15", "deep": "unavailable_no_weights"}` → Outer only.

---

## §FILE-seal — `seal.json` + `chain.jsonl` (item 17)

Normative hash formula (verbatim, preserved):

```
hash_i=SHA256(prev||SHA256(outer)||SHA256(deep)||canonical(replay))
```

Where `prev` = previous `hash` in `chain.jsonl` (genesis: 64×`0`), `SHA256(outer)` = bytes of `outer.json`, `SHA256(deep)` = bytes of `deep.bin` (empty string hash if Deep absent), `canonical(replay)` = canonical JSON (sorted keys, no whitespace) of `outer.json:replay`.

`seal.json` schema:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "seal.json",
  "type": "object",
  "required": ["run_id", "hash", "prev_hash", "sha_outer", "sha_deep", "algorithm"],
  "properties": {
    "run_id": { "type": "string" },
    "hash": { "type": "string" },
    "prev_hash": { "type": "string" },
    "sha_outer": { "type": "string" },
    "sha_deep": { "type": ["string", "null"] },
    "algorithm": { "type": "string" }
  }
}
```

Example (valid JSON):

```json
{
  "run_id": "RUN-000184",
  "hash": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
  "prev_hash": "0000000000000000000000000000000000000000000000000000000000000000",
  "sha_outer": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "sha_deep": "d4735e3a265e16eee03f59718b9b5d03019c07d8b6c51f90da3a666eec13ab35",
  "algorithm": "hash_i=SHA256(prev||SHA256(outer)||SHA256(deep)||canonical(replay))",
  "sealed_by": "sidecar",
  "sealed_at_iso": "2026-09-15T10:00:03Z"
}
```

`chain.jsonl` — one JSON object per line, append-only global chain:

```json
{ "run_id": "RUN-000184", "hash": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08", "prev_hash": "0000000000000000000000000000000000000000000000000000000000000000" }
```

Verify steps (`blackbox verify <RUN>`): 1) recompute `SHA256(outer.json)` bytes; 2) recompute `SHA256(deep.bin)` bytes (or empty); 3) canonicalize `replay`; 4) recompute `hash_i=SHA256(prev||SHA256(outer)||SHA256(deep)||canonical(replay))`; 5) compare to `seal.json:hash` and walk `chain.jsonl` continuity. Any edit → FAIL.

---

## §FILE-retention — `retention_state.json` + `retention_policy.json` (item 18)

Flag-only retention (P9 lock): Box NEVER auto-deletes. State is computed; policy is user-owned.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "retention_state.json",
  "type": "object",
  "required": ["used_gb", "level", "by_model"],
  "properties": {
    "used_gb": { "type": "number" },
    "level": { "enum": ["green", "yellow", "red"] },
    "by_model": { "type": "object" }
  }
}
```

Example `retention_state.json` (valid JSON):

```json
{
  "used_gb": 420.5,
  "level": "yellow",
  "by_model": { "M-001__C-48291": { "runs": 10000, "gb": 400.0 }, "M-EXT": { "runs": 5000, "gb": 20.5 } },
  "updated_at_iso": "2026-09-15T10:00:00Z"
}
```

Example `retention_policy.json` — user-owned (valid JSON):

```json
{
  "policy_version": 1,
  "owner": "researcher",
  "rules": [
    { "match": { "flag": "bad_output" }, "keep": "forever" },
    { "match": { "tag": "good" }, "keep_runs": 10000 }
  ],
  "auto_delete": "never"
}
```

`blackbox size --by-model` reads state; levels reuse estimator bands (GREEN <50 GB, YELLOW 50–1000 GB, RED >1000 GB). Enforcement is manual: user edits policy, user deletes.

---

## §WATCH-grammar — `watchlist.yaml` (item 19)

Addresses only, no meaning. Controller validates every address against manifest before run; unknown → fail fast.

| Field | Type | Required | Values |
|---|---|---|---|
| `version` | int | yes | `1` |
| `model_id` | string | yes | must match manifest |
| `checkpoint_sha256` | string | yes | must match manifest |
| `watchlist_id` | string | yes | e.g. `cohort-A-v3` |
| `trigger` | string \| object | yes | `always` \| `{on_flag: bad_output}` \| `{window: {after_s, until_s}}` |
| `window` | object | yes | `{start_token, end_token, every_n}` |
| `precision` | string | yes | `fp32` \| `fp16` \| `bf16` |
| `entries` | list | yes | `{preset: name}` \| `{addresses: [...]}` \| `{extends, watch}` |

Ranges: `N-24-0001..0100` inclusive. Single: `A-17-04`, `LOGITS`, `EMBED`, `NORM`.

Example (YAML):

```yaml
version: 1
model_id: M-001
checkpoint_sha256: ab12cd34ab12cd34ab12cd34ab12cd34ab12cd34ab12cd34ab12cd34ab12cd34
watchlist_id: cohort-A-v3
trigger: always
window: { start_token: 0, end_token: 511, every_n: 1 }
precision: fp16
entries:
  - preset: pilot-sparse-1k
  - addresses: [A-17-04, A-24-12, N-24-0001..0100, LOGITS]
```

Per-run patch (Python, sealed into `outer.json:watchlist_ref.patch`):

```python
with BlackBox.watch("cohort-A.yaml", add=["N-24-9999"], window={"start_token": 10, "end_token": 50}):
    model.generate(prompt, seed=42)
```

---

## §WATCH-presets — 7 presets (item 20)

Macros expanding to exact addresses. `extends: <preset> + watch: [...]` mixes without editing the preset.

| Preset | Expands to |
|---|---|
| `outer_only` | no Deep entries; Outer always |
| `last-4-layers-full` | L-29..32 all heads + all MLP neurons + LOGITS |
| `attention_only` | all heads (1024 for 32×32), no MLP |
| `mid-mlp-wide` | L-12..20 every 10th token (`every_n: 10`) |
| `logit-lens` | NORM per layer + LOGITS per layer |
| `pilot-sparse-1k` | 250 seeded neurons each from L08/L16/L24/L31 |
| `ioi-circuit-starter` | `A-17..24-04/07/12` + `N-20..24-0001..0200` (curated, stored neutrally) |

Example using extends (YAML):

```yaml
version: 1
model_id: M-001
checkpoint_sha256: ab12cd34ab12cd34ab12cd34ab12cd34ab12cd34ab12cd34ab12cd34ab12cd34
watchlist_id: cohort-A-plus
trigger: always
window: { start_token: 0, end_token: 511, every_n: 1 }
precision: fp16
entries:
  - extends: pilot-sparse-1k
    watch: [N-24-0001]
```

---

## §CTRL-triggers — controller (item 21)

User is boss (P1 lock). Controller evaluates trigger per run/token before arming Deep hooks.

| Trigger | Semantics |
|---|---|
| `always` | Deep armed for every run in window |
| `on_flag: bad_output` | arm Deep only when Outer safety flag trips (e.g. 1% flagged) |
| `window: {after_s, until_s}` | arm only between elapsed seconds, e.g. `after 5s` |
| `{start_token, end_token, every_n}` | arm tokens `start..end` stepping `every_n` |
| certain elements | explicit address hit, e.g. `N-24-0001 from X to Y` range gate |

Unarmed tokens write nothing to ring slots 1..N; `deep_index.json:triggered=false` if never armed.

---

## §CTRL-estimator — size gate (item 22)

Formula (normative): `bytes=W*T*R*B*1.15`, with `B`: fp32=4, fp16/bf16=2. Prints before run. Never auto-blocks; RED requires `--force`.

| Flag | Band | Behavior |
|---|---|---|
| GREEN | <50 GB | proceed |
| YELLOW | 50–1000 GB | warn, proceed |
| RED | >1000 GB | require `--force`, else abort with exit 2 |

Reference table (T=500, fp16 B=2):

| W | R=10k | R=100k | Flag (R=10k) |
|---|---|---|---|
| 1k | ~11.5 GB | ~115 GB | GREEN |
| 10k | ~115 GB | ~1.15 TB | YELLOW |
| 100k | ~1.15 TB | ~11.5 TB | RED, `--force` required |

CLI example:

```
Estimator: W=1,000 T=500 R=10,000 B=2B(fp16) → total ~10 GB raw
Flag: GREEN (<50GB). Proceeding.
---
Estimator: W=100,000 T=500 R=10,000 → ~1 TB fp16 / 2 TB fp32
Flag: RED (>1000GB). Require --force.
```

---

## §TAP-fsst — Forked-Stream Sidecar Tap (item 23)

Hybrid: wrapper-thin + hooks-async + sidecar-persistent. External witness, zero stalls on default stream.

- **Outer (0% GPU):** ~20-line shim `bb.wrap(model)` at `generate()` boundary captures prompt/tokens/logits/`cudaEvent` timing/seed/cost. Cost ~0.15 ms/request.
- **Deep (weights + trigger armed only):** `register_forward_hook` ONLY on parent modules in watchlist. Per token: `gathered = activation.index_select(dim=-1, idx)` on GPU → `copy_(pinned, non_blocking=True)` on dedicated `copyStream` overlapping next MatMul (~80% hidden). Default stream NEVER `synchronize()`. Naive `tensor.cpu()` (0.8–1.2 ms/hook → 1.8–2.2×) is banned.
- Ring full → `flags.ring_overflow=true`, drop Deep payload, never stall inference. Hook throw → auto `detach()`, model restored exactly.

---

## §TAP-sidecar — WAL ring (item 24)

- Transport: lock-free triple-buffered ring in `/dev/shm/bb.ring` (Docker volume, `--shm-size=1g`), file-backed `mmap`.
- Sidecar process (separate container): polls `mmap` → WAL append → per-token `fsync` → `hash_i=SHA256(prev||SHA256(outer)||SHA256(deep)||canonical(replay))` → writes `runs/RUN-xxx/`.
- Survival: main SIGKILL/OOM → evidence already `fsync`'d; `blackbox verify` still passes for sealed prefix.
- Pure sidecar via LD_PRELOAD/eBPF rejected (sees kernels not semantics, breaks Docker, CUDA ABI brittle).

---

## §GPU-adapters — porting interface (item 25)

One file per backend: `gpu_adapters/nvidia.py` (priority) → `amd.py` (HIP) / `apple.py` (MPS blit) / `cpu.py` (thread-pool fallback). Scan is CPU-only, no port needed.

```python
class GpuAdapter:
    def capabilities(self) -> dict: ...
    def create_context(self, watchlist: dict) -> object: ...
    def attach(self, model, ctx: object) -> None: ...
    def detach(self, model, ctx: object) -> None: ...
    def alloc_pinned(self, nbytes: int) -> object: ...
    def stage_copy(self, gathered, pinned, stream) -> None: ...
```

- Group watchlist by parent module; move `idx` to GPU once; double-buffered pinned host memory; one `copyStream` + `Event` per group.
- int8 quantized: dequantize on GPU (`int8 * scale`) before copy; store logical value + `dtype_meta`.
- Overhead linear in `W`, not model size: `~0.2*k*2B/BW / token_time`. CPU fallback uses `clone()` to queue (~0.005 ms for 1k on 7B CPU inference).

---

## §REPLAY — deterministic replay (item 31)

`outer.json:replay = {seed, temp, top_p, sampler, cudnn_deterministic, versions{torch,cuda}, checkpoint_sha256}` — bit-identical on same hardware/software; otherwise warn.

| Mode | Condition | Output |
|---|---|---|
| `deterministic` | same checkpoint + same versions + same GPU arch + `cudnn_deterministic=true` | bit-identical tokens |
| `reproducible` | same checkpoint, versions differ | same tokens expected, warn versions |
| `warn` | checkpoint mismatch or `temp>0` without seed | warn, replay still attempted |

---

## §REDACTOR — hook point only (item 32)

Raw save for research; for allowed prod logging the user plugs their own filter. Box provides the hook point, never a mandated filter.

```python
def redactor(prompt: str, tokens: list[int]) -> tuple[str, list[int]]:
    """User-owned. Return (possibly redacted) prompt and tokens BEFORE seal."""
    return prompt, tokens

model = blackbox.torch.wrap(model, redactor=redactor)
```

Runs without `redactor=` seal raw. With `redactor=`, sealed bytes are post-hook output; hook identity recorded in `outer.json:versions.redactor`.



# spec.md Section C — Search + Viewer + Env + Errors + Examples + Honesty + Traceability
Covers checklist items 26–30, 33–38. Source: product.md §8 Phase 6, §16–21. Scope: CLI + static HTML only. No React. No new promises.

---

## §SEARCH-tiers (item 26)

Tiered index. Goal: 10k-vs-10k comparison scans megabytes of Hot, not terabytes of Cold.

| Tier | Location | Content | Size ref (10k runs, W=1k, T=500) |
|------|----------|---------|----------------------------------|
| Tier 0 manifest | `blackbox_store/manifest/M-001__C-48291/manifest.json` | KB inventory: model, checkpoint sha256, arch (layers/hidden/heads/MLP), address list `L-01..32 → A-24-01..32 → N-24-0001..14336, EMBED, LOGITS`, runtime, hardware | KB, once per checkpoint |
| Tier 1 Hot | `blackbox_store/runs/RUN-*/meta.json` + `blackbox_store/index/shard-*.jsonl` (+ optional Parquet `index/*.parquet` queried via DuckDB) | Per run × per watched address: `min, max, mean, p95, max_window[10]` (max over sliding 10-token windows), `firing_rate(thresh)`, bloom event flags (`bad_output`, `ring_overflow`, `hash_ok`), `tags`, `checkpoint_sha256`, `watchlist_id` | ~5 MB / 10k runs |
| Tier 2 Cold | `blackbox_store/runs/RUN-*/deep.bin` + `deep_index.json` | Exact float values `[tokens × watched]`, row-major (see §FILE-deepbin in Section B) | ~1 MB/run (W=1k,T=500,fp16) → ~10 GB/10k |

Two-phase query (mandatory implementation):
1. **Hot filter:** scan Hot shards in parallel (thread pool, one shard per worker). Predicate on aggregates only, e.g. `max_window[10..19] >= 0.8`. Produces candidate run list.
2. **Cold verify:** for candidates only, `mmap` the slice `deep.bin[token_lo..token_hi, watched_idx]` via `deep_index.json` offset, check exact predicate. 100–1000× I/O reduction vs full scan.
3. Every result row carries `hash_seal_verified: bool` (recomputed via `seal.json` + `chain.jsonl` at query time; `false` rows are still returned but flagged — never silently dropped).

Columnar rule (default, not lock-in): Hot JSONL shards + Parquet + DuckDB for `watched-column scan` (works everywhere, no server). Naive Postgres/MySQL with one row per token per neuron (5B rows / 400 GB index for 10k×500×1k) is rejected — do not implement that shape. Other DBs (SQLite/liteSQL/MySQL/anything) allowed via `search-adapters/` implementing `hot_filter()` + `cold_verify()` per §SEARCH-tiers; see FORMAT_GUIDE for adapter contract. Presets unaffected (address lists only). Day-1 default is DuckDB; community adapters welcome, no per-user builds promised.

---

## §SEARCH-grammar (item 27)

Query language: SQL-like string passed to `blackbox find "<QUERY>"` and `BlackBox.find(query)`. Case-insensitive keywords. All addresses validated against `manifest.json` before Hot scan.

EBNF (builder-exact):

```
query      := find_stmt | compare_stmt
find_stmt  := "FIND RUNS WHERE" condition ("AND" condition)*
condition  := checkpoint_cond | tag_cond | prompt_cond | neuron_cond | flag_cond
checkpoint_cond := "checkpoint" "=" string
tag_cond   := "tag" "=" string
prompt_cond:= "prompt" "CONTAINS" string
flag_cond  := "flag" "=" ("bad_output" | "ring_overflow" | "hash_fail")
neuron_cond:= address ("AT" token_spec)? comparator number
address    := /[NAL]-.../ | "LOGITS" | "EMBED"   ; e.g. N-24-0001, A-24-07, L-24, N-24-0001..0100 (range expands)
token_spec := "token" "=" int | "token" int ".." int | "tokens" int ".." int
comparator := ">" | ">=" | "<" | "<=" | "=" | "!=" | "MAX" ">" | "MAX" ">=" | "MIN" "<"
compare_stmt := "COMPARE GROUPS" group "vs" group ("WITH STATS" addr_list "AT" token_spec "AGG" agg_list)?
group      := ident ":" ("tag" "=" string) ("LIMIT" int)?
addr_list  := address ("," address)*
agg_list   := agg ("," agg)*   ; agg := "mean" | "p95" | "firing_rate" "(" number ")" | "max" | "min"
```

Canonical examples (must all parse — conformance tests):

```sql
FIND RUNS WHERE checkpoint="sha256:abc..." AND N-24-0001 AT token=17 > 0.8
COMPARE GROUPS good:tag="good" LIMIT 10000 vs bad:tag="bad" LIMIT 10000
WITH STATS N-24-0001, A-24-07 AT tokens 15..20 AGG mean, p95, firing_rate(0.8)
FIND RUNS WHERE prompt CONTAINS "instructions for" AND N-24-1000 AT token 0..10 MAX >1.2
FIND RUNS WHERE tag="pilot" AND LOGITS AT token=17 > 3.5
FIND RUNS WHERE flag="bad_output" AND A-17-04 AT tokens 10..50 MAX >= 0.9
```

Semantics (exact):
- `N-Addr AT token=K > X` → Cold-exact value at (K, addr) > X after Hot pre-filter on `max_window` covering K.
- `AT token A..B MAX > X` → max over exact Cold slice [A..B] > X; Hot pre-filter uses overlapping `max_window` aggregates.
- `WITH STATS ... AGG` → computed on Cold slices of matched runs only; `firing_rate(t)` = fraction of tokens in range with value > t.
- `CONTAINS` → substring match on `outer.json:prompt` (case-sensitive), Hot-side (stored in `meta.json:prompt_prefix` + full prompt in `outer.json`).
- `COMPARE GROUPS` → runs two `FIND` passes (one per group filter), then aggregates; output = per-address `mean, p95, firing_rate` per group + delta histogram (see §CLI-compare owned by Section A; format contract: JSON `{group, n, stats{addr:{mean,p95,firing_rate}}}`).
- Validation: unknown address / unknown checkpoint / malformed token range → error, exit code 2, message `unknown address N-99-9999 not in manifest M-001__C-48291; see manifests/.../manifest.json inventory` (see §ERRORS). Never return empty silently on typo.

CLI contract (owned here for grammar; full flag tables in Section A):
- `blackbox find "<QUERY>" [--json | --table] [--limit N default 50] [--verify/--no-verify default verify]`
- `--json` emits one JSON object per line (JSONL) with `hash_seal_verified` per row. `--table` is human default.

---

## §VIEW-cli (item 28)

`blackbox view RUN-000184 [--json | --table] [--tokens 15..20] [--addrs N-24-0001,A-24-07]` — works with no GPU, no Docker, pure local file read.

Table mode (default, human):

```
RUN-000184  model=M-001  ckpt=sha256:abc…  watchlist=cohort-A-v3 (W=1000)  [hash: VERIFIED ✓ chain OK]
PROMPT (12 tokens): "Explain why the sky is blue"
TOK   TEXT      N-24-0001   A-24-07    LOGITS-top
15    " sky"      0.31      0.55       2.10
16    " is"       0.64      0.71       2.88
17    " blue"     0.92      0.83       3.41
OUTPUT: " because of Rayleigh scattering…"
seal: hash_i=9f2c… prev=41ab… verified=true
unwatched addresses shown as `--` (grey, null, not_watched); hash broken rows shown `!!` red.
```

- Colors (ANSI, disabled with `--no-color` or non-tty): watched+value = default/white numbers; column header blue; unwatched `--` grey (dim); `hash FAIL` banner + affected cells red; `VERIFIED ✓` green.
- Hash badge text (exact, both modes): success `[hash: VERIFIED ✓ chain OK]`; failure `[hash: FAIL ✗ RUN-000184 seal mismatch at deep.bin sha256 — see blackbox verify RUN-000184]`.
- `--tokens` / `--addrs` slice the Cold mmap; omitted = first 32 tokens × first 8 watched addresses (never dump full 500×1k by default).
- JSON mode (`--json`): machine-exact:

```json
{"run_id":"RUN-000184","model_id":"M-001","checkpoint_sha256":"abc...","hash_seal_verified":true,"prompt":"Explain why the sky is blue","tokens":[{"pos":17,"text":" blue","values":{"N-24-0001":0.92,"A-24-07":0.83}},{"pos":18,"text":" because","values":{"N-24-0001":null,"_note":"not_watched"}}],"output":" because of Rayleigh scattering…"}
```

- `null` = unwatched (never 0-fill, never interpolate). Exit codes: 0 ok, 3 hash-fail still prints data + red badge (verify distinguishes), 2 bad run-id/address (see §ERRORS).

---

## §VIEW-html (item 29)

`blackbox view RUN-000184 --open [--port 8321] [--export ./run-184.html]` — static HTML, no server required for `--export`; `--open` serves local-only `127.0.0.1` via stdlib `http.server` (no React, no npm, no Tailwind; vanilla HTML+inline CSS+inline SVG or `<canvas>`; Plotly/Dash optional only if already installed — v1 must work without them).

Wireframe (text-exact; builder reproduces this layout top-to-bottom):

```
+==================================================================+
| RUN-000184  [hash: VERIFIED ✓ chain OK]   M-001 | C-48291 | W=1k |
+==================================================================+
| PROMPT (tokenized chips, horizontal scroll)                       |
| [Explain][ why][ the][ sky][ is][ blue]                           |
+==================================================================+
| Token row 0..N (horizontal strip, click selects; selected=outline)|
| [T15 sky][T16 is][T17 blue *][T18 because]...  (* = max in view)  |
+==================================================================+
| DETAIL for selected token (default T17):                          |
|  Token 17 " blue"  t=4.821s  pos=17                               |
|  +----------------+--------------+-----------------+             |
|  | Watched (blue) | value        | sparkline 10..20  |             |
|  | N-24-0001      | 0.92         | ▁▃▅█             |             |
|  | A-24-07        | 0.83         | ▁▂▄█             |             |
|  +----------------+--------------+-----------------+             |
|  Unwatched (grey outline, no value):                              |
|  [N-24-0002 grey] [N-24-0003 grey] ... L-24 total 14336, 100 shown |
|  "address exists, value null — not_watched"                       |
+==================================================================+
| OUTPUT (final text, below; attention edges overlay toggle)        |
| " because of Rayleigh scattering…"                               |
+==================================================================+
| < Prev token | Next token > | Jump [17___] | back to PROMPT | fwd  |
+==================================================================+
```

Rules (exact):
- Flow is strictly `PROMPT → Token 0..500 → OUTPUT`. No other top-level panels. Left-to-right = time order.
- Color rule: watched+value cell = blue fill (`#1d4ed8` bg, white text) with exact value text (e.g. `0.92`); unwatched = grey outline (`#9ca3af` border, transparent bg, text `--` + tooltip `address exists, value null — not_watched`); hash-broken run = red banner + red cell borders (`#dc2626`) and badge `[hash: FAIL ✗ ...]`; selected token = 2px black outline.
- Click Token17 behavior (exact): clicking chip `T17` (or any token) sets `selected_token=17`, re-renders DETAIL panel to show **all watched addresses at that same token** (values + 10-token sparkline from Cold slice `[T-5..T+5]`), plus co-occurrence edge list `top-5 watched addresses with |corr|>0.5 at this token window` labeled `co-occurrence, not causality`. URL hash updates to `#token=17` so reload/back restores selection. Prev/Next buttons move selection ±1. Backward nav scrolls to PROMPT chips; forward nav scrolls to OUTPUT.
- Nav: PROMPT chips clickable (jump to token), token strip scrolls, OUTPUT static. Keyboard: ←/→ moves token. No routing, no build step; single self-contained `.html` for `--export` (inline CSS/JS, Cold slice embedded as JSON for viewed window only, full `deep.bin` never embedded).
- Fallbacks: no-GPU/CPU-only works (pure file read); missing `deep.bin` (Outer-only run) → DETAIL shows `Outer-only run: no Deep values — prompt/output/timing only`; hash fail → red banner but still renders (never blank).
- Calm style (easy on eye, not fancy — normative): white background, max-width 960px centered, system-ui 14-16px, line-height 1.5, flat borders only (no shadows/gradients/animations). Blue reserved for watched value cells only; rest white/grey. Token strip single horizontal scroll; DETAIL below tokens; OUTPUT below DETAIL. Must remain readable for long sessions without eye strain.

---

## §VIEW-partial (item 30)

Partial-but-precise rule (product.md Q10 lock — builder must not violate):
- If the watchlist has W=100 watched neurons, the graph shows exactly 100 filled (blue, exact value per token) + remaining addresses as grey placeholders (`address exists, value null, not_watched`). Example: L-24 has 14336 MLP neurons, 100 watched → 100 blue cells + 14236 grey outlines (paginated, e.g. 100/page with `showing 100 of 14236 unwatched` label).
- Never interpolate, never impute, never average unwatched into display. `null` in JSON, `--` in CLI, grey chip in HTML. Any aggregation (`mean`, sparkline) uses watched values only and labels `n_watched=W`.
- Addresses come from `manifest.json` inventory; values exist only during flight (Cold). If user asks for an address not in watchlist but in manifest → grey + `not_watched`. If not in manifest at all → error (§ERRORS), not grey.

---

## §BENCH (item 33)

`benchmark.sh` (repo root, committed) is the official methodology. No official 70B/B200 numbers ship day 1 — community runs in `benchmarks/community/` ARE the report (honesty lock).

```bash
# Outer (any machine, no GPU needed):
./benchmark.sh --mode outer --runs 1000
# Deep (Linux+Docker+NVIDIA):
./benchmark.sh --mode deep --watch watchlists/pilot-sparse-1k.yaml --tokens 500 --runs 100
```

Metrics printed (exact keys, JSON to stdout + markdown row appended to `benchmarks/community/RESULTS.md`):
`mode, W, T, R, precision, overhead_pct, ms_per_token_baseline, ms_per_token_tapped, deep_bin_bytes_per_run, total_GB, flag(GREEN/YELLOW/RED), torch, cuda, gpu, os`.
- Overhead = `(tapped − baseline)/baseline × 100`. Baseline = same prompts without `watch`/`wrap`.
- Expected reference (method, not promise): Outer <0.5%; Deep 1k watched ~5–7%; 100k flagged RED ~20%. Estimator flag thresholds: GREEN <50 GB total, YELLOW 50–1000 GB, RED >1000 GB (total = W·T·R·B·1.15).
- Community submission format: one dir per run `benchmarks/community/<date>-<gpu>-<model>-W<w>/` with `result.json` + `env.txt` (`nvidia-smi`, `torch.__version__`, `uname -a`). CI does not gate on numbers, only that `benchmark.sh --mode outer` passes on CPU.

---

## §ENV (item 34)

| Var / Flag | Default | Meaning |
|------------|---------|---------|
| `BLACKBOX_DIR` | `./blackbox_store` | Store root: `manifest/`, `runs/`, `index/`, `chain.jsonl`, `retention_state.json` |
| `BLACKBOX_SHM` | `/dev/shm/bb.ring` | Sidecar ring path (Docker volume) |
| `BLACKBOX_GPU` | `auto` (`cuda` if available else `cpu`) | Force `cpu` for fallback tests: `BLACKBOX_GPU=cpu pytest` |
| Docker Deep | `docker run --gpus all --shm-size=1g -v $BLACKBOX_DIR:/blackbox -v /weights:/weights blackbox` | `--gpus all` required for Deep; `--shm-size=1g` required for ring; Outer pip needs neither |
| CPU fallback | `cpu.py` adapter (`clone()` queue, ~0.005 ms/1k) | Outer + scan + search + viewer work on Windows 11 / Mac M4-M5 / Linux without NVIDIA; Deep values only meaningful where hooks ran |
| Disk layout | Hot (SSD, MBs) + Cold (local dir or S3-compatible mount under `$BLACKBOX_DIR`) | Hot searchable in ms; Cold mmap on drill-down; user owns retention (never auto-delete) |

---

## §ERRORS (item 35)

Fail-fast on unknown address; hooks never block; exit codes stable:

| Code | When | Message / behavior |
|------|------|--------------------|
| 0 | success | normal output; `verify` prints `chain OK` |
| 1 | runtime failure (OOM main, sidecar died, disk full) | `ERROR E1001 <what> — WAL up to <token> already sealed at <path>; rerun resume hint` ; sidecar death never loses sealed tokens |
| 2 | usage / validation (unknown address, bad watchlist, bad query, missing run) | `ERROR E2001 unknown address N-99-9999 not in manifest M-001__C-48291; valid range N-24-0001..14336 — see manifest inventory` ; controller validates watchlist vs manifest before run and aborts before any GPU work |
| 3 | seal/verify fail | `FAIL RUN-000184 seal mismatch: expected <h> got <g> (deep.bin sha256) — chain broken at <link>; data still viewable with red badge` ; `verify` exits 3 |
| 4 | size RED without `--force` | `Flag RED ~1TB — require --force to proceed (flag, never block)` |

Hook rule: hook throw → auto `detach()` that token, set `flag=hook_detached`, continue run (never kill job). Ring full → `flag=ring_overflow`, drop newest Deep payload, keep Outer + seal (never stall inference). `detach()` restores model byte-identical.

---

## §EXAMPLES (item 36)

Three committed examples under `examples/` (exact I/O contracts):

**01 `examples/01_outer_only_api.py`** — Outer on closed API (no weights, no GPU):
```python
import blackbox
with blackbox.watch("watchlists/outer_only.yaml"):   # or bb.wrap(llm_client)
    out = llm_client.generate("Explain why the sky is blue", seed=42)
print(out.text)
```
In: prompt string. Out: `outer.json` (~3 KB: prompt, output tokens+order, timing, seed/settings, `watchlist_ref=outer_only`) + `seal.json`; CLI `blackbox view RUN-xxx` shows prompt/output + `Outer-only run` notice; overhead <0.5%.

**02 `examples/02_deep_llama_watchlist.py`** — Deep on Llama-3-8B (Linux+Docker+NVIDIA):
```python
import blackbox.torch as bb
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("/weights/Llama-3-8B")
model = bb.wrap(model, watch="watchlists/pilot-sparse-1k.yaml")
with bb.controller(model, seed=42):
    out = model.generate("Explain why the sky is blue", max_new_tokens=64)
print(out)
```
In: local weights + watchlist (pilot-sparse-1k + `N-24-0001, A-17-04`). Out: `outer.json` + `deep.bin` (~1 MB for W=1k×T≈64 fp16) + `deep_index.json` + `seal.json`; `blackbox find 'N-24-0001 AT token=17 > 0.8'` returns run; viewer Token 17 shows `N-24-0001:0.92` blue.

**03 `examples/03_compare_good_bad.py`** — 10k good vs 10k bad comparison:
```python
import blackbox
for tag, prompts in [("good", good_ps), ("bad", bad_ps)]:
    for p in prompts:
        with blackbox.watch("watchlists/cohort-A.yaml", tag=tag):
            model.generate(p, seed=42)
# then: blackbox compare --good tag=good --bad tag=bad --neurons N-24-0001,A-24-07
```
In: two prompt lists + shared watchlist. Out: Hot-only scan (~10 MB) + ≤100 Cold slices; CLI prints per-group `mean/p95/firing_rate` + ASCII histogram delta; JSON mode emits exact stats object.

---

## §HONESTY (item 37)

Builder-visible constraints (from product.md §10–11; viewer/search must not imply otherwise):
- Never claim "records every neuron of any model" — records only declared watchlist, exact value per token.
- No official 70B/B200 benchmark day 1 — ship `benchmark.sh` + estimator flags; community dir is the report.
- Deep day 1 = Linux + Docker + NVIDIA + PyTorch only; Windows/Mac = Outer + CPU fallback + viewer/search (no Deep perf claims there).
- No custom schema/redactor/dashboard built per user — provide `FORMAT_GUIDE.md` hook + `export --zip`; user adapts.
- No SLA/PR-review promise — community-maintained unless funded.
- Never auto-delete / never auto-redact — flag only (`retention_state.json` GREEN/YELLOW/RED, user-owned policy).
- Locked-model honesty: API-only models = Outer always; Deep requires weights (or closed lab running inside own infra).
- Privacy line in README + viewer footer: `Raw save for research — don't run on private user traffic unless legal basis; plug your redactor= if you must.`

---

## §TRACE (item 38)

product.md → spec Section C mapping (builder checks each row):

| product.md | spec §C |
|------------|---------|
| §8 Phase 6 tiers + two-phase query + SQL | §SEARCH-tiers, §SEARCH-grammar |
| §8 Phase 6 viewer graph + Q10 partial-precise + Workflow A steps 6–7 | §VIEW-cli, §VIEW-html, §VIEW-partial |
| §10.1 + §20 Phase 3 DoD + §12 research | §BENCH |
| §8 Phase 3–4 Docker/shm + §17 deps + §18 layout | §ENV |
| §10.3/10.5 stability/retention + §13 P9/Q8 | §ERRORS |
| §18 examples/ + §16 Workflows A–C + §20 Phases | §EXAMPLES |
| §10–11 promises + non-promises | §HONESTY |
| §13 locks P4/Q10 + §21 open decisions | this table |

Open decisions (proposed, awaiting lock — Section C assumes these, changes nothing above if flipped):
1. License: MIT (vs Apache-2.0). 2. Package: `ai-blackbox` / import `blackbox`. 3. Viewer v1: CLI + static HTML (this section; React rejected). 4. Publish: GitHub-only v0.1, PyPI at v1.0.
