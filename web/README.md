# ARDY in the browser (WebGPU)

Text-to-motion running entirely client-side: one ONNX graph per autoregressive window executed by
[onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/) on WebGPU, a precomputed prompt-embedding library
instead of the 8B-parameter text encoder, and a three.js stick-figure player. Same stack as the
[GEAR-SONIC live demo](https://nvlabs.github.io/GEAR-SONIC/demo.html).

```bash
python web/export_web_onnx.py    # -> web/models/window.onnx (+ window.json, skeleton.json); needs the model on CPU
python web/build_prompts.py      # -> web/models/prompts.{json,bin}; needs the local LLM2Vec encoder (int4 on MPS)
python web/serve.py 8765         # static server with the COOP/COEP headers onnxruntime-web wants
open http://127.0.0.1:8765/      # pick a prompt, press Start
```

## How it works

- `ardy/exports/web.py` — `WebWindow` re-implements `Ardy.autoregressive_step` for the text-only case with every
  shape fixed (4 history frames, 40 generated frames, 10 denoising steps): history tokenization (autoencoder encoder),
  recentering, the unrolled CFG denoising loop with the DDIM sampler, root translation back to world space, decoding,
  and the motion-representation inverse (forward kinematics). Inputs: `history [1,4,330]`, `text_feat [1,1,4096]`,
  `noise [1,10,148]`, two guidance weights. Outputs: normalized motion features `[1,44,330]`, joint positions
  `[1,44,27,3]`, joint rotations. It matches `autoregressive_step` bit-for-bit given the same noise.
- `web/app.js` — keeps the last 4 output frames as the next window's history, buffers ~2 windows ahead on an async
  producer loop, plays at the model's 20 fps with requestAnimationFrame, prompt changes take effect at the next window.
- `web/build_prompts.py` — encodes a prompt list once with the local encoder (the demo's embedding cache is reused).
  Free-text prompts would need a text-encoder server (`scripts/run_text_encoder_server.py`).

## Measured (M1 Pro 16 GB, Chrome 152, fp32 graph, 813 MB)

| | |
|---|---|
| WebGPU session creation | 5–12 s |
| one window (10 denoising steps + decode + FK, 40 frames = 2 s of motion) | 270–300 ms after warmup, ~0.9 s first window → 7x real time |
| WASM fallback | ~3.6 s per window, not real time |
| onnxruntime vs PyTorch | motion 3e-4, joints 1e-5 m |
| window-to-window seams | joint jumps 0.04–0.21 m, inside the normal per-frame range (median 0.13 m) |

## Known gaps / next

- **Size**: 813 MB fp32. Two shortcuts tried and rejected: `onnxconverter-common`'s fp16 pass yields a graph onnxruntime
  rejects (mismatched Cast types inside attention), and onnxruntime's 8-bit `MatMulNBits` quantizer made the file *larger*
  (877 MB: the 10 unrolled steps share weights it does not dedupe) with a visible error (joints off by 1.7 cm on average,
  16 cm max). Next candidates: export the denoiser in half precision from PyTorch directly (`model.half()` with the
  sampler / FK kept in fp32), or a `Loop`-based graph so the denoiser weights appear once and can be quantized per block.
- Token count is baked into the decoder reshape at export, so the graph is fixed at 4 + 40 frames; a different history
  length or horizon needs a re-export.
- No kinematic constraints / waypoints, no `motion_correction` post-processing (C++ extension) in the browser.
- Hosting: the model must be served from a CORS-enabled origin (e.g. a Hugging Face repo); redistribution of converted
  ARDY weights is subject to the NVIDIA Open Model License.
