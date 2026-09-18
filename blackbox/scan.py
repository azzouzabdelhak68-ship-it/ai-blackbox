"""Scan weights on CPU into a manifest (spec §A.1, §FILE-manifest, plan.md §8.1).

CPU-only: reads ``config.json`` + first ``*.safetensors`` header only.
MUST NOT import ``torch.cuda``, MUST NOT need a GPU or prompt.

Exit mapping for CLI: E02 ``ValueError``/``FileNotFoundError`` (missing
config), E03 ``ManifestExistsError`` (same sha exists, needs ``--force``).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import struct
import time
from typing import Any

from blackbox.exceptions import ManifestExistsError

MODEL_ID_DEFAULT = "M-001"
_API_STUB_MODEL_ID = "M-EXT"


def _read_safetensors_header(path: str) -> bytes:
    """Return raw header JSON bytes of a safetensors file (CPU read only)."""
    with open(path, "rb") as fh:
        prefix = fh.read(8)
        if len(prefix) < 8:
            raise ValueError(f"E02: truncated safetensors header in {path}")
        (header_len,) = struct.unpack("<Q", prefix)
        if header_len > 64 * 1024 * 1024:
            raise ValueError(f"E02: safetensors header too large in {path}")
        header = fh.read(int(header_len))
        if len(header) < int(header_len):
            raise ValueError(f"E02: truncated safetensors header in {path}")
    json.loads(header.decode("utf-8"))  # validate JSON, keep raw bytes
    return header


def _arch_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Map HF config keys to manifest architecture fields."""
    layers = cfg.get("num_hidden_layers", cfg.get("n_layer"))
    hidden = cfg.get("hidden_size", cfg.get("n_embd"))
    heads = cfg.get("num_attention_heads", cfg.get("n_head"))
    mlp = cfg.get("intermediate_size", cfg.get("n_inner"))
    vocab = cfg.get("vocab_size")
    family = str(cfg.get("model_type", "unknown"))
    vals: dict[str, Any] = {
        "layers": layers,
        "hidden": hidden,
        "heads": heads,
        "mlp": mlp,
        "vocab": vocab,
    }
    missing = [k for k, v in vals.items() if v is None]
    if missing:
        raise ValueError(f"E02: config.json missing keys for {missing}")
    layers_i = int(str(vals["layers"]))
    hidden_i = int(str(vals["hidden"]))
    heads_i = int(str(vals["heads"]))
    mlp_i = int(str(vals["mlp"]))
    vocab_i = int(str(vals["vocab"]))
    return {
        "family": family,
        "layers": layers_i,
        "hidden": hidden_i,
        "heads": heads_i,
        "mlp_per_layer": mlp_i,
        "vocab": vocab_i,
    }


def scan(
    path: str,
    output: str | None = None,
    force: bool = False,
    api_model: bool = False,
) -> dict[str, Any]:
    """Scan weights dir on CPU and write ``manifest.json``.

    Args:
        path: weights dir (or API model name when ``api_model=True``).
        output: manifest file or dir; default
            ``manifests/<MODEL>_<SHA8>/manifest.json``.
        force: overwrite same-sha manifest.
        api_model: stub mode -> ``{"mode": "outer_only"}``; Deep fails closed.

    Returns:
        Manifest dict (also written to disk, except stub mode).

    Raises:
        FileNotFoundError: E02 missing ``config.json`` (hint: ``--api-model``).
        ManifestExistsError: E03 same-sha manifest exists without ``--force``.
        ValueError: E02 bad config/header JSON.
    """
    t0 = time.monotonic()
    if api_model:
        name = os.path.basename(path.rstrip("/\\")) or path
        return {
            "model": path,
            "model_id": _API_STUB_MODEL_ID,
            "model_name": name,
            "mode": "outer_only",
            "deep": "unavailable_no_weights",
        }
    weights_dir = path
    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(
            f"E02: no config.json in {weights_dir}. Hint: --api-model NAME for Outer-only stub."
        )
    cfg_path = os.path.join(weights_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(
            f"E02: no config.json in {weights_dir}. Hint: --api-model NAME for Outer-only stub."
        )
    with open(cfg_path, "rb") as fh:
        config_bytes = fh.read()
    try:
        cfg = json.loads(config_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"E02: bad config.json in {weights_dir}: {exc}") from exc
    arch = _arch_from_config(cfg)

    st_files = sorted(glob.glob(os.path.join(weights_dir, "*.safetensors")))
    header_bytes = b""
    if st_files:
        header_bytes = _read_safetensors_header(st_files[0])
    checkpoint_sha256 = hashlib.sha256(config_bytes + header_bytes).hexdigest()

    model_name = os.path.basename(os.path.normpath(weights_dir))
    model_id = str(cfg.get("model_id", MODEL_ID_DEFAULT))
    layers, heads, mlp = arch["layers"], arch["heads"], arch["mlp_per_layer"]
    manifest: dict[str, Any] = {
        "model_id": model_id,
        "model_name": model_name,
        "checkpoint_sha256": checkpoint_sha256,
        "architecture": arch,
        "inventory": {
            "layers": [f"L-{i:02d}" for i in (1, layers)] if layers > 1 else ["L-01"],
            "heads": [f"A-{i:02d}-01..{heads:02d}" for i in (1, layers)],
            "neurons_per_layer": mlp,
            "embed": "EMBED",
            "logits": "LOGITS",
            "attention": f"A-01-01..A-{layers:02d}-{heads:02d}",
            "neurons": f"N-01-0001..N-{layers:02d}-{mlp:04d}",
            "specials": ["EMBED", "LOGITS", "NORM"],
        },
        "runtime": {"framework": "PyTorch (cpu scan)", "cuda": "none"},
        "hardware": {"gpu": "none", "detected_via": "cpu-scan"},
        "adapter": {"family": arch["family"], "layout": "row_major_float16"},
        "scan": {
            "source": "config.json + weights header",
            "gpu_used": False,
            "duration_s": round(time.monotonic() - t0, 2),
        },
    }

    if output is None:
        out_path = os.path.join("manifests", f"{model_id}_{checkpoint_sha256[:8]}", "manifest.json")
    elif output.endswith(".json"):
        out_path = output
    else:
        out_path = os.path.join(output, "manifest.json")
    if os.path.isfile(out_path) and not force:
        try:
            with open(out_path, encoding="utf-8") as fh:
                existing = json.load(fh)
            if existing.get("checkpoint_sha256") == checkpoint_sha256:
                raise ManifestExistsError(
                    f"E03: manifest exists sha256:{checkpoint_sha256}. Use --force."
                )
        except ManifestExistsError:
            raise
        except (OSError, ValueError, UnicodeDecodeError):
            raise ManifestExistsError(f"E03: manifest exists at {out_path}. Use --force.") from None
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    manifest["scan"]["duration_s"] = round(time.monotonic() - t0, 2)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return manifest
