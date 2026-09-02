"""Export ARDY denoiser + decoder to ONNX on CPU and verify with onnxruntime (web-feasibility probe)."""
import os, sys, time, collections, numpy as np, torch, onnx, onnxruntime as ort
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from export_onnx import export_denoiser_onnx, export_decoder_onnx, make_denoiser_dummy_inputs, make_decoder_dummy_inputs
from ardy.model.load_model import load_model
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx"); os.makedirs(out, exist_ok=True)
name = sys.argv[2] if len(sys.argv) > 2 else "Ardy-Core-RP-20FPS-Horizon40"
model, cfg = load_model(name, device="cpu", text_encoder=False, return_config=True)
fps, patch = model.motion_rep.fps, model.denoiser.num_frames_per_token
max_tok = ((10 * fps // patch) * patch) // patch   # same window budget as the demo (10 s)
print(f"model {name}: fps={fps} frames/token={patch} max_tokens={max_tok} horizon={model.gen_horizon_len}")
den_path, dec_path = f"{out}/denoiser.onnx", f"{out}/decoder.onnx"
t0 = time.time(); export_denoiser_onnx(model.denoiser, cfg, den_path, num_tokens=max_tok); print(f"denoiser exported in {time.time()-t0:.0f}s")
t0 = time.time(); export_decoder_onnx(model.autoencoder, dec_path, num_tokens=max_tok); print(f"decoder exported in {time.time()-t0:.0f}s")
for p in (den_path, dec_path):
    m = onnx.load(p); onnx.checker.check_model(m)
    ops = collections.Counter(n.op_type for n in m.graph.node)
    ext = sum(os.path.getsize(f"{out}/{f}") for f in os.listdir(out) if f.startswith(os.path.basename(p)) and f != os.path.basename(p))
    print(f"\n{os.path.basename(p)}: {os.path.getsize(p)/1e6:.0f} MB (+{ext/1e6:.0f} MB external), opset {m.opset_import[0].version}, {len(m.graph.node)} nodes")
    print("  ops:", ", ".join(f"{k}:{v}" for k, v in ops.most_common()))
    print("  inputs:", ", ".join(f"{i.name}[{','.join(str(d.dim_value or d.dim_param) for d in i.type.tensor_type.shape.dim)}]" for i in m.graph.input))
# numerical check vs torch, at a smaller token count than the export (exercises the dynamic axes)
torch.manual_seed(0)
d_in = make_denoiser_dummy_inputs(model.denoiser, num_tokens=12, num_text_tokens=1, device="cpu")
with torch.no_grad(): ref = model.denoiser(**d_in)
sess = ort.InferenceSession(den_path, providers=["CPUExecutionProvider"])
feed = {k: v.numpy() for k, v in d_in.items() if k in {i.name for i in sess.get_inputs()}}
t0 = time.time(); got = sess.run(None, feed)[0]; dt = time.time() - t0
print(f"\ndenoiser ORT-CPU vs torch: max|d|={np.abs(got - ref.numpy()).max():.2e} (|ref| mean {np.abs(ref.numpy()).mean():.3f}), 1 step @12 tokens {dt*1000:.0f} ms (M1 CPU, fp32)")
print("decoder: verified separately at the export token count (see web/README.md)")
