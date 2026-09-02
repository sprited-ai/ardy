"""Export one ARDY window (text-only autoregressive step) as a single ONNX graph for the browser.

Usage: python web/export_web_onnx.py [out_dir] [model_name]
Writes <out_dir>/window.onnx plus the small JSON side files the web app needs.
"""

import hashlib
import json
import os
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from ardy.exports.web import WebWindow, draw_noise, export_web_window, reference_window  # noqa: E402
from ardy.model.load_model import load_model  # noqa: E402
from interactive_demo.embedding_cache import DEFAULT_CACHE_DIR, text_encoder_cache_namespace  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
name = sys.argv[2] if len(sys.argv) > 2 else "Ardy-Core-RP-20FPS-Horizon40"
os.makedirs(out, exist_ok=True)


def cached_embedding(text: str) -> torch.Tensor:
    """Prompt embedding from the demo's on-disk cache (encoded once by the int4 LLM2Vec encoder)."""
    ns = os.environ.get("TEXT_ENCODER", "llm2vec-int4")
    path = os.path.join(DEFAULT_CACHE_DIR, ns, hashlib.sha256(text.encode("utf-8")).hexdigest() + ".npy")
    return torch.from_numpy(np.load(path)).float()[None]  # [1, 1, 4096]


model = load_model(name, device="cpu", text_encoder=False)
window = WebWindow(model, history_frames=4, num_denoising_steps=10).eval()
llm_dim = 4096
text = cached_embedding("A person is walking.")
print(f"{name}: history {window.history_frames} + gen {window.gen_frames} frames, {window.num_denoising_steps} steps, token dim {window.token_dim}")

# --- bootstrap history: generate a first window from nothing, keep its last 4 frames -------------------
torch.manual_seed(0)
with torch.no_grad():
    boot = model.autoregressive_step(
        num_frames=model.gen_horizon_len, num_denoising_steps=10, motion_mask=None, observed_motion=None,
        cfg_weight=(3.5, 1.5), text_feat=text, text_pad_mask=torch.ones(1, 1, dtype=torch.bool),
        init_history_sequence=None, init_global_translation=torch.zeros(1, 3), init_first_heading_angle=torch.zeros(1),
    )
history = boot[:, -window.history_frames :].contiguous()

# --- WebWindow vs the original code path, same noise ------------------------------------------------
with torch.no_grad():
    ref = reference_window(model, history, text, seed=1, cfg_weight=(3.5, 1.5))
    noise = draw_noise(window, seed=1, device="cpu")
    motion, joints, rots = window(history, text, noise, torch.tensor([3.5]), torch.tensor([1.5]))
print(f"WebWindow vs autoregressive_step: max|d motion|={(motion - ref).abs().max():.2e}  (|ref| mean {ref.abs().mean():.3f}); joints {tuple(joints.shape)}")

# --- export + onnxruntime check --------------------------------------------------------------------
path = os.path.join(out, "window.onnx")
t0 = time.time(); export_web_window(window, path, llm_dim); print(f"exported {path} in {time.time() - t0:.0f} s: {os.path.getsize(path) / 1e6:.0f} MB")
m = onnx.load(path); onnx.checker.check_model(m)
ops = {}
for n in m.graph.node: ops[n.op_type] = ops.get(n.op_type, 0) + 1
print(f"  {len(m.graph.node)} nodes; ops: " + ", ".join(f"{k}:{v}" for k, v in sorted(ops.items(), key=lambda kv: -kv[1])))
sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
feed = {"history": history.numpy(), "text_feat": text.numpy(), "noise": noise.numpy(), "cfg_weight_text": np.array([3.5], np.float32), "cfg_weight_cstr": np.array([1.5], np.float32)}
t0 = time.time(); o_motion, o_joints, o_rots = sess.run(None, feed); dt = time.time() - t0
print(f"onnxruntime CPU vs torch: max|d motion|={np.abs(o_motion - motion.numpy()).max():.2e}, max|d joints|={np.abs(o_joints - joints.numpy()).max():.2e} m; one window {dt * 1000:.0f} ms")

# --- side files for the web app ----------------------------------------------------------------------
skel = model.motion_rep.skeleton
names = list(skel.bone_order_names)
parents = [names.index(p) if p is not None else -1 for _, p in skel.bone_order_names_with_parents]
json.dump({"name": skel.name, "joint_names": names, "parents": parents}, open(os.path.join(out, "skeleton.json"), "w"))
json.dump({"history_frames": window.history_frames, "gen_frames": window.gen_frames, "fps": model.motion_rep.fps,
           "num_denoising_steps": window.num_denoising_steps, "token_dim": window.token_dim, "motion_dim": model.motion_rep.motion_rep_dim,
           "llm_dim": llm_dim, "model": name, "init_history": history[0].tolist()}, open(os.path.join(out, "window.json"), "w"))
print("wrote skeleton.json, window.json (with init_history)")
