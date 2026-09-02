# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a memory-light LLM2Vec text-encoder checkpoint: merge the LoRA adapters, optionally quantize.

The ARDY text encoder is ``LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised``: the
``meta-llama/Meta-Llama-3-8B-Instruct`` base, the ``-mntp`` LoRA merged in, and the
``-mntp-supervised`` LoRA applied on top of that. At runtime the vendored LLM2Vec code
materializes this with PEFT, which needs the full-precision (16 GB) base model in memory.

That does not fit on a 16 GB Apple Silicon machine, so this script produces the final weights
once, streaming tensor-by-tensor (peak RAM ~2-3 GB):

    W_merged = W_base + scale * (B_mntp @ A_mntp) + scale * (B_sup @ A_sup)

(both adapters: r=16, alpha=32 -> scale 2.0; the supervised adapter was trained on the
mntp-merged weights, so the two deltas are additive), and then

- ``--quantize int4``: weight-only int4 (group size 64, MSE-clipped) -> ~5 GB checkpoint that
  runs on MPS / CPU / CUDA through ``torch._weight_int4pack_mm`` (see ardy/model/llm2vec/quantized.py).
- ``--quantize int8``: weight-only per-channel int8 -> ~9 GB, near-lossless reference.
- ``--quantize none``: plain merged bf16/fp16 checkpoint (16 GB), for machines with the memory.

``lm_head`` is dropped (the encoder only uses ``LlamaModel``). Outputs go to
``~/.cache/ardy/text_encoders/LLM2Vec-Meta-Llama-3-8B-Instruct-<variant>`` and are picked up by
``load_text_encoder()`` via the ``llm2vec-int4`` / ``llm2vec-int8`` / ``llm2vec-merged`` presets.

Usage::

    python scripts/merge_llm2vec_lora.py                          # int4 (default)
    python scripts/merge_llm2vec_lora.py --quantize int4 int8     # both variants in one pass
    python scripts/merge_llm2vec_lora.py --quantize none --dtype float16
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

from ardy.model.llm2vec.quantized import INT4_GROUP_SIZE, QUANT_MARKER, quantize_int4, quantize_int8

BASE_REPO = "meta-llama/Meta-Llama-3-8B-Instruct"
MNTP_REPO = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
SUP_REPO = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"

TEXT_ENCODERS_DIR = Path.home() / ".cache" / "ardy" / "text_encoders"
VARIANT_DIRS = {
    "int4": TEXT_ENCODERS_DIR / "LLM2Vec-Meta-Llama-3-8B-Instruct-int4",
    "int8": TEXT_ENCODERS_DIR / "LLM2Vec-Meta-Llama-3-8B-Instruct-int8",
    "none": TEXT_ENCODERS_DIR / "LLM2Vec-Meta-Llama-3-8B-Instruct-merged",
}
MERGED_MARKER = "llm2vec_merged.json"

# Keep output shards small so the merge never holds more than this in RAM per variant.
SHARD_BYTES = 2 * 1024**3


def _snapshot(repo_id: str, allow_patterns=None) -> Path:
    """Local snapshot dir for ``repo_id`` (cache first, then download)."""
    try:
        return Path(snapshot_download(repo_id, allow_patterns=allow_patterns, local_files_only=True))
    except Exception:
        return Path(snapshot_download(repo_id, allow_patterns=allow_patterns))


def _load_adapter(adapter_dir: Path) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], float]:
    """Return ``{base_weight_key: (A, B)}`` and the LoRA scale for one PEFT adapter dir."""
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    assert cfg["peft_type"] == "LORA", cfg["peft_type"]
    scale = cfg["lora_alpha"] / cfg["r"]
    if cfg.get("use_rslora"):
        scale = cfg["lora_alpha"] / (cfg["r"] ** 0.5)
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), "pt") as f:
        for key in f.keys():
            # base_model.model.layers.0.mlp.down_proj.lora_A.weight -> model.layers.0.mlp.down_proj.weight
            assert key.startswith("base_model.") and ".lora_" in key, key
            module, lora_part = key[len("base_model.") :].split(".lora_", 1)
            which = lora_part.split(".", 1)[0]  # "A" or "B"
            pairs.setdefault(module + ".weight", {})[which] = f.get_tensor(key)
    out = {}
    for base_key, ab in pairs.items():
        assert set(ab) == {"A", "B"}, (base_key, set(ab))
        out[base_key] = (ab["A"], ab["B"])
    return out, scale


def _is_linear_weight(key: str) -> bool:
    """Decoder-block linear weights (attention + MLP projections); everything else stays float."""
    return key.startswith("model.layers.") and key.endswith("_proj.weight")


class _Writer:
    """Accumulates tensors for one output variant and flushes them into ~2 GB safetensors shards."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for old in self.out_dir.glob("model-*.safetensors"):
            old.unlink()
        self.buffer: dict[str, torch.Tensor] = {}
        self.buffer_bytes = 0
        self.shard_idx = 0
        self.weight_map: dict[str, str] = {}

    def add(self, key: str, tensor: torch.Tensor) -> None:
        tensor = tensor.contiguous()
        self.buffer[key] = tensor
        self.buffer_bytes += tensor.numel() * tensor.element_size()
        if self.buffer_bytes >= SHARD_BYTES:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        self.shard_idx += 1
        name = f"model-{self.shard_idx:05d}.safetensors"
        save_file(self.buffer, str(self.out_dir / name), metadata={"format": "pt"})
        for k in self.buffer:
            self.weight_map[k] = name
        print(f"  [{self.out_dir.name}] wrote {name} ({self.buffer_bytes / 1024**3:.2f} GB, {len(self.buffer)} tensors)")
        self.buffer = {}
        self.buffer_bytes = 0

    def finish(self) -> int:
        self.flush()
        total = sum((self.out_dir / f).stat().st_size for f in set(self.weight_map.values()))
        (self.out_dir / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": total}, "weight_map": self.weight_map}, indent=2) + "\n"
        )
        return total


def _write_config(base_dir: Path, out_dir: Path, dtype_name: str) -> None:
    cfg = json.loads((base_dir / "config.json").read_text())
    # The LLM2Vec code keys its prompt template on this exact name; keep it.
    cfg["_name_or_path"] = BASE_REPO
    # We load the encoder as a bare LlamaModel (no lm_head).
    cfg["architectures"] = ["LlamaModel"]
    # transformers >= 4.53 builds a bidirectional mask when config.is_causal is False, which
    # also handles padded batches (the vendored LlamaBiModel only sets is_causal=False on the
    # attention modules, which is enough when the mask is skipped for unpadded inputs).
    cfg["is_causal"] = False
    cfg["dtype"] = dtype_name
    cfg.pop("torch_dtype", None)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"):
        src = base_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)


def build(base_dir: Path, mntp_dir: Path, sup_dir: Path, variants: dict[str, Path], dtype: torch.dtype) -> None:
    adapters = [_load_adapter(mntp_dir), _load_adapter(sup_dir)]
    for i, (deltas, scale) in enumerate(adapters):
        print(f"adapter {i}: {len(deltas)} LoRA targets, scale={scale}")
    expected_merged = len(adapters[0][0])

    writers = {variant: _Writer(out_dir) for variant, out_dir in variants.items()}
    index = json.loads((base_dir / "model.safetensors.index.json").read_text())
    shard_files = sorted(set(index["weight_map"].values()))

    merged_count = 0
    quantized_count = 0
    t0 = time.time()
    for shard in shard_files:
        print(f"reading {shard}")
        with safe_open(str(base_dir / shard), "pt") as f:
            for key in f.keys():
                if key == "lm_head.weight":
                    continue  # encoder-only: the LM head is never used
                w = f.get_tensor(key)
                hit = False
                for deltas, scale in adapters:
                    if key in deltas:
                        A, B = deltas[key]
                        if not hit:
                            w = w.to(torch.float32)
                        w = w + scale * (B.to(torch.float32) @ A.to(torch.float32))
                        hit = True
                if hit:
                    merged_count += 1

                for variant, writer in writers.items():
                    if variant != "none" and _is_linear_weight(key):
                        base = key[: -len(".weight")]
                        q = quantize_int4(w) if variant == "int4" else quantize_int8(w)
                        for suffix, tensor in q.items():
                            writer.add(f"{base}.{suffix}", tensor)
                        quantized_count += 1
                    else:
                        writer.add(key, w.to(dtype))
    dtype_name = str(dtype).removeprefix("torch.")
    for variant, writer in writers.items():
        total = writer.finish()
        out_dir = writer.out_dir
        _write_config(base_dir, out_dir, dtype_name)
        meta = {
            "base": BASE_REPO,
            "adapters": [MNTP_REPO, SUP_REPO],
            "dtype": dtype_name,
            "merged_weights": merged_count,
        }
        (out_dir / MERGED_MARKER).write_text(json.dumps(meta, indent=2) + "\n")
        if variant != "none":
            (out_dir / QUANT_MARKER).write_text(
                json.dumps(
                    {
                        "format": "ardy-weight-only",
                        "bits": 4 if variant == "int4" else 8,
                        "group_size": INT4_GROUP_SIZE if variant == "int4" else None,
                        "name_or_path": BASE_REPO,
                        "storage_dtype": dtype_name,
                    },
                    indent=2,
                )
                + "\n"
            )
        print(f"done -> {out_dir} ({total / 1024**3:.2f} GB)")

    print(f"merged LoRA into {merged_count} weights (expected {expected_merged}); quantized {quantized_count}")
    print(f"total time {time.time() - t0:.0f}s")
    assert merged_count == expected_merged, (merged_count, expected_merged)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--quantize",
        nargs="+",
        choices=list(VARIANT_DIRS),
        default=["int4"],
        help="Variants to write in one pass over the base weights (default: int4).",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16"],
        default="float16",
        help="Storage dtype for non-quantized tensors (embeddings, norms; everything for --quantize none). "
        "Default float16: MPS has no native bf16 math.",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Override the output dir (single variant only).")
    parser.add_argument("--force", action="store_true", help="Rebuild variants that already exist.")
    args = parser.parse_args()

    variants = {v: VARIANT_DIRS[v] for v in dict.fromkeys(args.quantize)}
    if args.out_dir is not None:
        if len(variants) != 1:
            parser.error("--out-dir can only be used with a single --quantize variant")
        variants = {next(iter(variants)): args.out_dir}
    if not args.force:
        variants = {v: d for v, d in variants.items() if not (d / MERGED_MARKER).exists()}
        if not variants:
            print("all requested variants already exist; pass --force to rebuild.")
            return

    base_dir = _snapshot(BASE_REPO, allow_patterns=["*.json", "model-*.safetensors"])
    mntp_dir = _snapshot(MNTP_REPO)
    sup_dir = _snapshot(SUP_REPO)
    print(f"base={base_dir}\nmntp={mntp_dir}\nsup={sup_dir}\nvariants={ {k: str(v) for k, v in variants.items()} }")
    build(base_dir, mntp_dir, sup_dir, variants, getattr(torch, args.dtype))


if __name__ == "__main__":
    main()
