"""Build the browser text encoder (INT4 MatMulNBits ONNX) from ARDY's own merged LLM2Vec weights.

The graph structure is reused from TREE Industries' export (bidirectional Llama-3 + mean pooling over embed_mask,
fixed 64 tokens); every weight is replaced by ours: the int8 checkpoint written by scripts/merge_llm2vec_lora.py
(base + mntp + supervised adapters merged exactly as ardy/model/load_model.py loads them) is dequantized and
re-quantized to symmetric 4-bit blocks of 32 in onnxruntime's MatMulNBits layout. Embeddings and norms are copied.

Usage: python web/build_text_encoder_onnx.py [out_dir] [int8_checkpoint_dir]
"""

import glob
import json
import os
import sys
import time

import numpy as np
import onnx
import torch
from huggingface_hub import hf_hub_download
from onnx import numpy_helper
from onnx.external_data_helper import set_external_data
from safetensors import safe_open

from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, DefaultWeightOnlyQuantizer

SRC_REPO = "TREEIndustries/Llama-3-ARDY-Text-Encoder-ONNX"
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "encoder-sprited")
ckpt = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/.cache/ardy/text_encoders/LLM2Vec-Meta-Llama-3-8B-Instruct-int8")
os.makedirs(os.path.join(out, "onnx"), exist_ok=True)

# --- our weights (streamed from safetensors) -------------------------------------------------------------
index = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
handles = {}
def tensor(name):
    f = index[name]
    if f not in handles: handles[f] = safe_open(os.path.join(ckpt, f), "pt")
    return handles[f].get_tensor(name)
def dequant_int8(prefix):  # our int8: w ~= q * scale (per output row)
    q = tensor(prefix + ".qweight").float(); s = tensor(prefix + ".scales").float()
    return (q * s[:, None]).contiguous()

# --- source graph ------------------------------------------------------------------------------------------
graph_path = hf_hub_download(SRC_REPO, "onnx/text_encoder_int4.onnx")
m = onnx.load(graph_path, load_external_data=False)
inits = {t.name: t for t in m.graph.initializer}
quantizer = DefaultWeightOnlyQuantizer(DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, accuracy_level=4, bits=4))

t0 = time.time(); done = 0; replaced = set()
for node in m.graph.node:
    if node.op_type != "MatMulNBits":
        continue
    # '/model/layers.7/self_attn/q_proj/MatMul_Q4' -> 'model.layers.7.self_attn.q_proj'
    hf = node.name.split("/MatMul")[0].strip("/").replace("/", ".")
    N = next(a.i for a in node.attribute if a.name == "N"); K = next(a.i for a in node.attribute if a.name == "K")
    w = dequant_int8(hf)  # [N, K] (torch Linear layout: out x in)
    assert tuple(w.shape) == (N, K), (hf, w.shape, N, K)
    packed, scales, zp = quantizer.qbits_block_quant(w.t().contiguous().numpy().astype(np.float32))  # expects [K, N]; returns [N, K/32, 16], [N, K/32]
    scales = scales.reshape(-1)
    q_name, s_name = node.input[1], node.input[2]
    assert list(inits[q_name].dims) == list(packed.shape), (q_name, inits[q_name].dims, packed.shape)
    inits[q_name].CopyFrom(numpy_helper.from_array(packed, q_name)); inits[s_name].CopyFrom(numpy_helper.from_array(scales.astype(np.float16), s_name))
    replaced.update([q_name, s_name]); done += 1
    if done % 32 == 0: print(f"  quantized {done}/224 matmuls ({time.time() - t0:.0f} s)", flush=True)

for t in m.graph.initializer:  # embeddings, norms: straight copy
    if t.name in replaced: continue
    if t.name in index:
        arr = tensor(t.name).to(torch.float16).numpy() if t.data_type == onnx.TensorProto.FLOAT16 else tensor(t.name).numpy()
        assert list(t.dims) == list(arr.shape), (t.name, t.dims, arr.shape)
        t.CopyFrom(numpy_helper.from_array(arr, t.name)); replaced.add(t.name)
    else:
        raise SystemExit(f"initializer {t.name} has no counterpart in the checkpoint")
print(f"replaced {len(replaced)} initializers in {time.time() - t0:.0f} s; saving...", flush=True)
for t in m.graph.initializer:
    set_external_data(t, location="text_encoder_int4.onnx.data")
onnx.save_model(m, os.path.join(out, "onnx", "text_encoder_int4.onnx"), save_as_external_data=True, all_tensors_to_one_file=True, location="text_encoder_int4.onnx.data", size_threshold=0, convert_attribute=False)
for f in ("tokenizer.json", "tokenizer_config.json"):
    os.makedirs(os.path.join(out, "tokenizer"), exist_ok=True)
    import shutil; shutil.copy(os.path.join(ckpt, f), os.path.join(out, "tokenizer", f))
print("wrote", out, {f: round(os.path.getsize(os.path.join(out, 'onnx', f)) / 1e9, 2) for f in os.listdir(os.path.join(out, 'onnx'))}, "GB")
