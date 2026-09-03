"""Export the condition-input window graph (cond [1,1,2048] instead of text_feat [1,1,4096]) plus text_proj.onnx.

Usage: python web/export_web_onnx_cond.py [out_dir] [model_name]
Verifies: WebWindowCond(cond=TextProj(feat)) == WebWindow(feat) in torch, then onnxruntime vs torch.
"""
import hashlib, json, os, sys, time
import numpy as np, onnx, onnxruntime as ort, torch
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(REPO, "scripts"))
from ardy.exports.web import WebWindow, WebWindowCond, TextProj, draw_noise, export_web_window_cond, export_text_proj
from ardy.model.load_model import load_model
from interactive_demo.embedding_cache import DEFAULT_CACHE_DIR
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
name = sys.argv[2] if len(sys.argv) > 2 else "Ardy-Core-RP-20FPS-Horizon40"
feat = torch.from_numpy(np.load(os.path.join(DEFAULT_CACHE_DIR, os.environ.get("TEXT_ENCODER", "llm2vec-int4"), hashlib.sha256(b"A person is walking.").hexdigest() + ".npy"))).float()[None]
model = load_model(name, device="cpu", text_encoder=False)
ref_window = WebWindow(model, 4, 10).eval()            # must be built before the cond variant swaps embed_text
cfgj = json.load(open(os.path.join(out, "window.json"))); history = torch.tensor(cfgj["init_history"])[None]
noise = draw_noise(ref_window, seed=1, device="cpu")
with torch.no_grad(): ref = ref_window(history, feat, noise, torch.tensor([3.5]), torch.tensor([1.5]))[0]
proj = TextProj(model).eval()
with torch.no_grad(): cond = proj(feat)
window = WebWindowCond(model, 4, 10).eval()          # swaps embed_text in place (after the reference run)
with torch.no_grad(): got = window(history, cond, noise, torch.tensor([3.5]), torch.tensor([1.5]))[0]
print(f"WebWindowCond(TextProj(feat)) vs WebWindow(feat): max|d motion| = {(got - ref).abs().max():.2e}  cond dim {window.cond_dim}")
t0 = time.time(); export_web_window_cond(window, os.path.join(out, "window_cond.onnx")); print(f"window_cond.onnx: {os.path.getsize(os.path.join(out, 'window_cond.onnx'))/1e6:.0f} MB in {time.time()-t0:.0f} s")
export_text_proj(proj, os.path.join(out, "text_proj.onnx")); print(f"text_proj.onnx: {os.path.getsize(os.path.join(out, 'text_proj.onnx'))/1e6:.1f} MB")
sp = ort.InferenceSession(os.path.join(out, "text_proj.onnx"), providers=["CPUExecutionProvider"]); c = sp.run(None, {"text_feat": feat.numpy()})[0]
print(f"text_proj ORT vs torch: {np.abs(c - cond.numpy()).max():.2e}")
sw = ort.InferenceSession(os.path.join(out, "window_cond.onnx"), providers=["CPUExecutionProvider"])
o = sw.run(None, {"history": history.numpy(), "cond": c, "noise": noise.numpy(), "cfg_weight_text": np.array([3.5], np.float32), "cfg_weight_cstr": np.array([1.5], np.float32)})[0]
print(f"window_cond ORT vs torch reference: {np.abs(o - ref.numpy()).max():.2e}")
cfgj.update({"onnx": "window_cond.onnx", "cond_dim": window.cond_dim, "text_proj": "text_proj.onnx", "inputs": ["history", "cond", "noise", "cfg_weight_text", "cfg_weight_cstr"]})
json.dump(cfgj, open(os.path.join(out, "window.json"), "w")); print("window.json updated (onnx=window_cond.onnx, cond_dim, text_proj)")
