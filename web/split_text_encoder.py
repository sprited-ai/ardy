"""Split the browser text encoder into two halves that each fit an onnxruntime-web instance (wasm32, 4 GB heap).

Part A: token ids/masks -> hidden state entering layer `cut`; part B: hidden (+ masks) -> text_embedding. The
halves are built by hand from the node graph (the model is too large for onnx's protobuf serializer with data
loaded), each with its own external data file. Writes <out>/onnx/part_{a,b}.onnx(.data) and <out>/encoder.json.

Usage: python web/split_text_encoder.py <encoder_dir> [cut_layers="8,20"]
"""

import json
import os
import re
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
from onnx.external_data_helper import load_external_data_for_tensor

enc_dir = sys.argv[1]; cuts = [int(c) for c in (sys.argv[2] if len(sys.argv) > 2 else "8,20").split(",")]
src_dir = os.path.join(enc_dir, "onnx"); src = os.path.join(src_dir, "text_encoder_int4.onnx")
m = onnx.load(src, load_external_data=False)
inits = {t.name: t for t in m.graph.initializer}
prod = {o: n for n in m.graph.node for o in n.output}
graph_in = {i.name: i for i in m.graph.input}

def layer_of(n):
    r = re.search(r"/model/layers\.(\d+)/", n.name); return int(r.group(1)) if r else None
def boundary(cut):
    for n in m.graph.node:
        if layer_of(n) == cut and "input_layernorm" in n.name and n.op_type != "Constant":
            for i in n.input:
                p = prod.get(i)
                if i not in inits and p is not None and (layer_of(p) or 0) < cut: return i
    raise SystemExit(f"boundary tensor for layer {cut} not found")
H = [boundary(c) for c in cuts]

def closure(outputs, stop):
    seen, nodes, stack = set(), [], list(outputs)
    while stack:
        t = stack.pop()
        if t in seen or t in stop or t in inits or t in graph_in: continue
        seen.add(t); n = prod.get(t)
        if n is None: continue
        nodes.append(n); stack.extend(n.input)
    return nodes
# part k computes from boundary k-1 (or the graph inputs) to boundary k (or text_embedding);
# the mask / rotary prefix has no weights and is simply recomputed inside every part.
targets = H + ["text_embedding"]
parts = []
for k, tgt in enumerate(targets):
    stop = {H[k - 1]} if k > 0 else set()
    nodes = closure([tgt], stop=stop)
    reads = {i for n in nodes for i in n.input if i and i not in inits}
    parts.append({"name": f"part_{chr(97 + k)}", "nodes": nodes, "in_hidden": H[k - 1] if k > 0 else None, "out": tgt, "graph_inputs": sorted(t for t in reads if t in graph_in)})
    print(f"  {parts[-1]['name']}: {len(nodes)} nodes, hidden in {parts[-1]['in_hidden']}, out {tgt}, reads {parts[-1]['graph_inputs']}")

# keep original node order
order = {n.name: i for i, n in enumerate(m.graph.node)}
def build(name, nodes, inputs, outputs):
    nodes = sorted(nodes, key=lambda n: order[n.name])
    used = {i for n in nodes for i in n.input}
    tensors = []
    for t in m.graph.initializer:
        if t.name in used:
            t2 = TensorProto(); t2.CopyFrom(t); load_external_data_for_tensor(t2, src_dir); t2.ClearField("external_data"); t2.data_location = TensorProto.DEFAULT; tensors.append(t2)
    g = helper.make_graph(nodes, name, inputs, outputs, initializer=tensors)
    mp = helper.make_model(g, opset_imports=list(m.opset_import), ir_version=m.ir_version)
    out = os.path.join(src_dir, f"{name}.onnx")
    onnx.save_model(mp, out, save_as_external_data=True, all_tensors_to_one_file=True, location=f"{name}.onnx.data", size_threshold=0)
    print(f"  {name}: {len(nodes)} nodes, {len(tensors)} initializers, data {os.path.getsize(out + '.data') / 1e9:.2f} GB")
t0 = time.time()
vi = lambda name: helper.make_tensor_value_info(name, TensorProto.FLOAT16, [1, 64, 4096])
for pt in parts:
    ins = ([vi(pt["in_hidden"])] if pt["in_hidden"] else []) + [graph_in[k] for k in pt["graph_inputs"]]
    outs = [o for o in m.graph.output] if pt["out"] == "text_embedding" else [vi(pt["out"])]
    build(pt["name"], pt["nodes"], ins, outs)
print(f"split written in {time.time() - t0:.0f} s")
def io(path, which):
    mp = onnx.load(path, load_external_data=False)
    return [{"name": v.name, "dtype": TensorProto.DataType.Name(v.type.tensor_type.elem_type).lower(), "shape": [d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim]} for v in getattr(mp.graph, which)]
manifest = {"seq": 64, "llm_dim": 4096, "cut_layers": cuts, "parts": [{"name": pt["name"], "size_bytes": os.path.getsize(os.path.join(src_dir, pt["name"] + ".onnx.data")), "inputs": io(os.path.join(src_dir, pt["name"] + ".onnx"), "input"), "outputs": io(os.path.join(src_dir, pt["name"] + ".onnx"), "output")} for pt in parts]}
json.dump(manifest, open(os.path.join(enc_dir, "encoder.json"), "w"), indent=1)

# --- verify A∘B against the full graph (CPU) --------------------------------------------------------------
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.path.join(enc_dir, "tokenizer"))
p = "A person is walking."; ids = tok.encode(f"<|start_header_id|>user<|end_header_id|>\n\n{p}<|eot_id|>", add_special_tokens=False); seq = [128000] + ids; L = 64; pad = L - len(seq)
ii = np.full((1, L), 128009, np.int64); am = np.zeros((1, L), np.int64); em = np.zeros((1, L), np.int64); ps = seq.index(128007) + 2
for i, t in enumerate(seq): ii[0, pad + i] = t; am[0, pad + i] = 1; em[0, pad + i] = int(i >= ps)
feed = {"input_ids": ii, "attention_mask": am, "embed_mask": em}
full = ort.InferenceSession(src, providers=["CPUExecutionProvider"]).run(None, feed)[0]
state = dict(feed)
for pt in parts:
    sess = ort.InferenceSession(os.path.join(src_dir, pt["name"] + ".onnx"), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {i.name: state[i.name] for i in sess.get_inputs()})
    state.update(dict(zip([o.name for o in sess.get_outputs()], outs)))
split = state["text_embedding"]
cos = float(np.dot(full[0], split[0]) / (np.linalg.norm(full[0]) * np.linalg.norm(split[0]))); print(f"chained parts vs full: cosine {cos:.6f}, max|d| {np.abs(full - split).max():.3e}")
