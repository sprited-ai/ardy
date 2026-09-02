# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight-only int4 / int8 quantized linear layers for the LLM2Vec text encoder.

The ARDY text encoder is an 8B-parameter Llama-3 model: 16 GB in bf16, which does not fit
next to the OS on a 16 GB Apple Silicon machine (and MPS caps a single process at roughly
two thirds of unified memory). This module lets the encoder run from a pre-quantized
checkpoint produced by ``scripts/merge_llm2vec_lora.py --quantize int4``:

- **Storage format (device-agnostic, safetensors):** per ``nn.Linear`` weight ``[N, K]``:
    - int4: ``qweight`` uint8 ``[N, K/2]`` (two 4-bit codes per byte, even column in the low
      nibble), ``scales`` / ``zeros`` fp16 ``[N, K/group_size]`` with
      ``w ~= (q - 8) * scale + zero`` per group.
    - int8: ``qweight`` int8 ``[N, K]``, ``scales`` fp16 ``[N]`` with ``w ~= q * scale``.
- **Compute:** PyTorch's built-in weight-only kernels ``torch._weight_int4pack_mm`` /
  ``torch._weight_int8pack_mm`` (available on MPS, CPU and CUDA), with a dequantize +
  ``F.linear`` fallback for anything else. The device-specific packed layout is built
  lazily on first use on the target device.

Only ``nn.Linear`` layers inside the decoder blocks are quantized; embeddings and norms stay
in half precision.
"""

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

QUANT_MARKER = "ardy_quant.json"
INT4_GROUP_SIZE = 64
# Clip ratios searched per group when quantizing to int4 (MSE-optimal clipping cuts the
# quantization error noticeably compared to plain min/max at almost no cost).
INT4_CLIP_RATIOS = (1.0, 0.975, 0.95, 0.925, 0.9, 0.875, 0.85)
# innerKTiles for torch._convert_weight_to_int4pack.
_INT4_INNER_K_TILES = 8


# ---------------------------------------------------------------------------
# Quantization (offline, used by scripts/merge_llm2vec_lora.py)
# ---------------------------------------------------------------------------


def quantize_int8(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    """Symmetric per-output-channel int8 quantization of a ``[N, K]`` weight."""
    w = weight.to(torch.float32)
    scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0
    q = torch.round(w / scale[:, None]).clamp(-128, 127).to(torch.int8)
    return {"qweight": q.contiguous(), "scales": scale.to(torch.float16).contiguous()}


def quantize_int4(weight: torch.Tensor, group_size: int = INT4_GROUP_SIZE) -> dict[str, torch.Tensor]:
    """Asymmetric per-group int4 quantization of a ``[N, K]`` weight with MSE-optimal clipping.

    Codes are stored as uint8 with the even column in the low nibble; ``scales``/``zeros`` are
    fp16 ``[N, K/group_size]`` such that ``w ~= (q - 8) * scale + zero``.
    """
    n, k = weight.shape
    assert k % group_size == 0, f"K={k} must be a multiple of group_size={group_size}"
    assert k % 2 == 0
    w = weight.to(torch.float32).reshape(n, k // group_size, group_size)
    wmin_full = w.amin(dim=-1, keepdim=True)
    wmax_full = w.amax(dim=-1, keepdim=True)

    best_err = None
    best_q = best_scale = best_zero = None
    for ratio in INT4_CLIP_RATIOS:
        wmin = wmin_full * ratio
        wmax = wmax_full * ratio
        scale = ((wmax - wmin) / 15.0).clamp(min=1e-8)
        zero = wmin + scale * 8
        q = torch.round((w - zero) / scale + 8).clamp(0, 15)
        deq = (q - 8) * scale + zero
        err = ((deq - w) ** 2).sum(dim=-1, keepdim=True)  # [N, G, 1]
        if best_err is None:
            best_err, best_q, best_scale, best_zero = err, q, scale, zero
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_q = torch.where(better, q, best_q)
            best_scale = torch.where(better, scale, best_scale)
            best_zero = torch.where(better, zero, best_zero)

    q = best_q.to(torch.uint8).reshape(n, k)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()
    return {
        "qweight": packed,
        "scales": best_scale.squeeze(-1).to(torch.float16).contiguous(),
        "zeros": best_zero.squeeze(-1).to(torch.float16).contiguous(),
    }


def dequantize_int4(qweight: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group_size: int) -> torch.Tensor:
    """Inverse of :func:`quantize_int4` -> float32 ``[N, K]``."""
    n = qweight.shape[0]
    lo = qweight & 0x0F
    hi = qweight >> 4
    q = torch.stack([lo, hi], dim=-1).reshape(n, -1).to(torch.float32)  # even col = low nibble
    q = q.reshape(n, -1, group_size)
    w = (q - 8) * scales.to(torch.float32)[..., None] + zeros.to(torch.float32)[..., None]
    return w.reshape(n, -1)


# ---------------------------------------------------------------------------
# Runtime modules
# ---------------------------------------------------------------------------


def _use_int4_kernel(device: torch.device) -> bool:
    # torch._convert_weight_to_int4pack / _weight_int4pack_mm exist for MPS and CUDA; the CPU
    # build ships different "_for_cpu" variants with another layout, so CPU uses the fallback.
    return device.type in ("mps", "cuda")


def _use_int8_kernel(device: torch.device) -> bool:
    return device.type in ("mps", "cpu", "cuda")


class Int4Linear(nn.Module):
    """``nn.Linear`` replacement computing ``x @ W.T`` from int4 group-quantized weights.

    Memory layout: the portable ``qweight`` (uint8, even column in the low nibble) is kept on the
    CPU exactly as loaded from safetensors (mmap-backed, so it costs no RAM until touched and is
    evictable afterwards) and is *not* a registered buffer, so ``.to(device)`` never copies it.
    The device only ever holds the kernel's packed layout, built in row chunks on first use.
    ``scales`` / ``zeros`` are ordinary buffers and follow ``.to(device / dtype)``.
    """

    # Rows per packing chunk: bounds the transient uint8 copy on the device (~K/2 bytes per row).
    _PACK_ROWS = 2048

    def __init__(self, in_features: int, out_features: int, group_size: int = INT4_GROUP_SIZE, bias: bool = False):
        super().__init__()
        assert not bias, "Llama linears have no bias"
        assert in_features % group_size == 0 and in_features % 2 == 0
        assert out_features % 8 == 0, "int4 packing needs out_features % 8 == 0"
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        n_groups = in_features // group_size
        self.register_buffer("scales", torch.zeros(out_features, n_groups, dtype=torch.float16))
        self.register_buffer("zeros", torch.zeros(out_features, n_groups, dtype=torch.float16))
        self._qweight_cpu: Optional[torch.Tensor] = None  # uint8 [N, K/2], portable nibble order
        self._packed: Optional[torch.Tensor] = None  # device layout for torch._weight_int4pack_mm
        self._scales_and_zeros: Optional[torch.Tensor] = None  # [K/gs, N, 2]
        self._packed_key = None

    # -- state dict plumbing: qweight is loaded/saved like a buffer but never moved by .to()
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        q = state_dict.pop(prefix + "qweight", None)
        if q is not None:
            assert q.dtype == torch.uint8 and tuple(q.shape) == (self.out_features, self.in_features // 2), q.shape
            self._qweight_cpu = q.cpu()
            self._packed = None
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if self._qweight_cpu is not None:
            destination[prefix + "qweight"] = self._qweight_cpu

    @property
    def qweight(self) -> Optional[torch.Tensor]:
        return self._qweight_cpu

    @qweight.setter
    def qweight(self, value: torch.Tensor) -> None:
        self._qweight_cpu = value.cpu()
        self._packed = None

    def _apply(self, fn, recurse=True):
        # .to(device / dtype) moves scales/zeros; the packed weight is rebuilt lazily to match.
        # A no-op .to() (LLM2Vec.encode calls .to(device) on every call) keeps the packed weight.
        super()._apply(fn, recurse)
        if self._packed is not None and self._packed_key != (self.scales.device, self.scales.dtype):
            self._packed = None
        return self

    def _prepare(self) -> None:
        device, dtype = self.scales.device, self.scales.dtype
        if self._packed is not None and self._packed_key == (device, dtype):
            return
        if self._qweight_cpu is None:
            raise RuntimeError("Int4Linear has no weights loaded")
        chunks = []
        for r in range(0, self.out_features, self._PACK_ROWS):
            q = self._qweight_cpu[r : r + self._PACK_ROWS].to(device)
            # torch expects the even column in the HIGH nibble; storage keeps it in the low nibble.
            swapped = ((q & 0x0F) << 4) | (q >> 4)
            chunks.append(torch._convert_weight_to_int4pack(swapped.contiguous(), _INT4_INNER_K_TILES))
            del q, swapped
        self._packed = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
        self._scales_and_zeros = torch.stack([self.scales.T, self.zeros.T], dim=-1).to(dtype).contiguous()
        self._packed_key = (device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, self.in_features)
        if _use_int4_kernel(x2.device) and x2.dtype in (torch.float16, torch.bfloat16):
            self._prepare()
            saz = self._scales_and_zeros
            if saz.dtype != x2.dtype:
                saz = saz.to(x2.dtype)
            out = torch._weight_int4pack_mm(x2, self._packed, self.group_size, saz)
        else:
            w = dequantize_int4(self._qweight_cpu.to(x2.device), self.scales, self.zeros, self.group_size).to(x2.dtype)
            out = F.linear(x2, w)
        return out.reshape(*shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bits=4, group_size={self.group_size}"


class Int8Linear(nn.Module):
    """``nn.Linear`` replacement computing ``x @ W.T`` from per-channel int8 weights."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        assert not bias, "Llama linears have no bias"
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("qweight", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scales", torch.zeros(out_features, dtype=torch.float16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, self.in_features)
        if _use_int8_kernel(x2.device) and x2.dtype in (torch.float16, torch.bfloat16):
            out = torch._weight_int8pack_mm(x2, self.qweight, self.scales.to(x2.dtype))
        else:
            out = F.linear(x2, (self.qweight.to(torch.float32) * self.scales.to(torch.float32)[:, None]).to(x2.dtype))
        return out.reshape(*shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bits=8"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def is_quantized_checkpoint(path) -> bool:
    return Path(path).is_dir() and (Path(path) / QUANT_MARKER).exists()


def read_quant_info(path) -> dict:
    return json.loads((Path(path) / QUANT_MARKER).read_text())


def replace_linears(model: nn.Module, bits: int, group_size: int = INT4_GROUP_SIZE, skip: tuple = ()) -> int:
    """Swap every ``nn.Linear`` (except names in ``skip``) for the matching quantized module."""
    count = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and full not in skip:
                if bits == 4:
                    new = Int4Linear(child.in_features, child.out_features, group_size, bias=child.bias is not None)
                elif bits == 8:
                    new = Int8Linear(child.in_features, child.out_features, bias=child.bias is not None)
                else:
                    raise ValueError(bits)
                setattr(module, child_name, new)
                count += 1
    return count


def load_quantized_llama_encoder(path, dtype: torch.dtype = torch.float16):
    """Build the bidirectional Llama encoder from an ARDY int4/int8 checkpoint dir.

    Returns the model on CPU with quantized linears; non-quantized tensors (embeddings, norms)
    are cast to ``dtype``. Move it with ``.to(device)`` afterwards.
    """
    from safetensors.torch import load_file
    from transformers import AutoConfig

    from .models.bidirectional_llama import LlamaBiModel

    path = Path(path)
    info = read_quant_info(path)
    bits = int(info["bits"])
    group_size = int(info.get("group_size") or INT4_GROUP_SIZE)  # int8 checkpoints store null

    config = AutoConfig.from_pretrained(path)
    with torch.device("meta"):
        model = LlamaBiModel(config)
    n_replaced = replace_linears(model, bits, group_size)

    index = json.loads((path / "model.safetensors.index.json").read_text())
    state: dict[str, torch.Tensor] = {}
    for shard in sorted(set(index["weight_map"].values())):
        state.update(load_file(str(path / shard)))
    # Stored keys carry the "model." prefix of the causal-LM checkpoint; LlamaBiModel is the bare model.
    state = {k.removeprefix("model."): v for k, v in state.items() if k != "lm_head.weight"}
    for k, v in state.items():
        if v.is_floating_point():
            state[k] = v.to(dtype)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    # Rotary inv_freq is a non-persistent buffer computed in __init__ (on meta); rebuild it.
    model.rotary_emb = type(model.rotary_emb)(config=config)
    missing = [m for m in missing if "inv_freq" not in m]
    if missing or unexpected:
        raise RuntimeError(f"quantized checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.config._name_or_path = info.get("name_or_path", config._name_or_path)
    model.eval()
    model._ardy_quant = {"bits": bits, "group_size": group_size, "linears": n_replaced}
    return model
