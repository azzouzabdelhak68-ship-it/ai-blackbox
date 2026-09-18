# FORMAT_GUIDE — we document, you edit

> **Rule: we document, you edit — `blackbox export --zip` gives plain files. We won't build your schema.**

## What export gives you

```bash
blackbox export --zip --run RUN-000184 --with-manifest --out ./RUN-000184.zip
# flat zip, no proprietary DB:
#   outer.json        (HOT, 2-5 KB: prompt, output, timing, seed/settings, replay)
#   deep.bin          (COLD, optional: float LE row-major [T, W], token-major)
#   deep_index.json   (HOT pointer: address -> {offset, dtype, shape})
#   seal.json         (HOT: hash_i chain link)
#   manifest.json     (only with --with-manifest: inventory of addresses)
```

## How to adapt to your format

1. Unzip: `unzip RUN-000184.zip -d ./run184/`.
2. Read schemas in `spec.md`: `§FILE-outer`, `§FILE-deepbin`, `§FILE-deepindex`, `§FILE-seal`, `§FILE-manifest`.
3. Map per-model layout via `adapters/format/*.json`:
   - `adapters/format/llama3.json` (Llama-3 family: row_major_float16).
   - `adapters/format/mistral.json` (Mistral family).
4. Convert: `deep.bin` decoder is `numpy.fromfile(path, dtype).reshape(T, W)` (see `spec.md` `§FILE-deepbin`); frame via `blackbox.to_dataframe(run)` for float32 logical values.
5. Validate: every example JSON must parse (`python -m json.tool`); re-seal is yours after edit (edited runs fail `blackbox verify` by design).

We provide the hook point (`FORMAT_GUIDE.md` + `adapters/format/<family>.json`), not per-user builds.
