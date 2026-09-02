# ARDY in the browser (prototype)

Feasibility probe for running ARDY's denoiser + decoder with onnxruntime-web (WebGPU), the same stack as the
[GEAR-SONIC live web demo](https://nvlabs.github.io/GEAR-SONIC/demo.html) (`ort.min.js` WebGPU/WASM + three.js +
`coi-serviceworker`).

```bash
python web/export_web_onnx.py            # -> web/onnx/denoiser.onnx (685 MB fp32), decoder.onnx (82 MB)
python web/serve.py 8765                 # static server with COOP/COEP headers (needed for WASM threads)
open http://127.0.0.1:8765/bench.html    # runs 10 denoiser steps + decoder on webgpu, then wasm
```

The export reuses `scripts/export_onnx.py` (the TensorRT path) on CPU, at the demo's 10 s window budget
(50 tokens x 4 frames for the 20 fps Core model). The token count is baked into a reshape inside the decoder
graph, so both graphs run at exactly the exported size and callers pad shorter windows with the masks, as the
TensorRT path already does.

## Measured (M1 Pro 16 GB, Chrome 152, fp32, 50-token window)

| backend | denoiser / step | 10-step window (= 2 s of motion) | decoder |
|---|---|---|---|
| WebGPU | 106 ms | 1.06 s (~1.9x real time) | 35 ms |
| WASM (SIMD + threads) | 357 ms | 3.57 s (0.56x, not real time) | - |
| reference: onnxruntime CPU, native | 43 ms @12 tokens | | 8 ms |
| reference: PyTorch MPS, native | | 0.29 s | |

ONNX vs PyTorch outputs agree to 2e-4 (denoiser) / 1e-3 (decoder). Ops used are all standard opset-17 ops
(MatMul, LayerNormalization, Softmax, Erf, Gather/Scatter, Where, Trilu ...).

## What a full web version still needs

- **Text encoder**: Llama-3-8B LLM2Vec cannot run in the browser. Either a small server (`scripts/run_text_encoder_server.py`)
  or precomputed embeddings for a prompt library (4096 x fp16 = 8 KB per prompt).
- **JS port of the loop**: DDIM sampler (21 lines), diffusion schedule (109), `denoising_step` (94), motion representation
  inverse to joint positions (~380) and skeleton FK; or export those as small ONNX graphs too and keep JS thin.
- **Viewer**: three.js skeleton/mesh playback (the SONIC demo page is a template).
- **Size/speed**: fp16 weights halve the 767 MB download; exporting at a smaller fixed window (e.g. 16 tokens) should
  cut the per-step cost roughly 3x on WebGPU.
- Post-processing (`motion_correction` C++ extension) is not available in the browser; skip or port.
