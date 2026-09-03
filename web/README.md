# ARDY in the browser (standalone, WebGPU)

A standalone port of the ARDY interactive demo: the same layout (header, tabbed panel, timeline), the skinned Core
character, prompt switching with "Now playing" / "Compute" labels — with generation running entirely in the browser
through [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/) on WebGPU. No server: the page is static and the
model is fetched from Hugging Face ([sprited/ardy-web-onnx](https://huggingface.co/sprited/ardy-web-onnx)) on first use
and kept in the browser cache. Phones and metered connections are asked before the 813 MB download.

```
web/
  index.html, app/{main,engine,viewer,timeline,models}.js   the standalone demo (static; any web host)
  export_web_onnx.py   ardy/exports/web.py WebWindow -> models/window.onnx + window.json + skeleton.json
  build_prompts.py     prompt library -> models/prompts.{json,bin}   (LLM2Vec encoder, run once)
  build_skin.py        Core skin (LBS) -> models/skin_cskel27.{json,bin}
  hybrid.html, worker.html, worker.js   browser compute worker for the *server* demo (run_demo.py --backend browser)
  serve.py             local static server with COOP/COEP headers
```

## Hosting

Live at **https://ardy.sprited.ai/** as a Cloudflare Worker serving static assets (`web/wrangler.jsonc`, route
`ardy.sprited.ai/*` on the sprited.ai zone; `web/.assetsignore` keeps the local model files out of the upload):

```bash
npx wrangler deploy --config web/wrangler.jsonc      # needs `npx wrangler login`
```

A GitHub Pages mirror is published from the `gh-pages` branch at https://sprited-ai.github.io/ardy/.

URL flags: `?models=local|hf|<prefix/>` (default: `./models/` if present, else Hugging Face), `?backend=webgpu|wasm`,
`?model=core-rp-20fps-h40`, `?ask=1` (always show the download dialog).

## How it works

- `ardy/exports/web.py` — `WebWindow` re-implements `Ardy.autoregressive_step` for the text-only case with every
  shape fixed (4 history frames, 40 generated frames, 10 denoising steps): history tokenization, recentering, the
  unrolled CFG denoising loop with the DDIM sampler, root translation back to world space, decoding and forward
  kinematics. One ONNX graph; inputs `history [1,4,330]`, `text_feat [1,1,4096]`, `noise [1,10,148]`, two guidance
  weights; outputs normalized motion `[1,44,330]`, joints `[1,44,27,3]`, joint rotations. Bit-identical to the
  PyTorch path given the same noise; onnxruntime matches to 3e-4.
- `app/engine.js` — the demo's autoregressive loop: a frame buffer, the last 4 frames as history, replans when
  playback gets within *Replan trigger* frames of the end, prompt changes replan from the next frame, Restart /
  Restart From Now, seeded noise. `app/viewer.js` poses the Core skin with linear blend skinning
  (`ardy/viz/core_skin.py` math) from the graph's joint positions and rotations. `app/timeline.js` is the demo's
  timeline (frame ruler, prompt segments, constraint track rows, draggable cursor).
- Text prompts: the window graph takes the denoiser's text *conditions* (root 1024 + body 1024; `window_cond.onnx`).
  Presets (20 precomputed LLM2Vec features) go through `text_proj.onnx` (the two `embed_text` layers, 34 MB).
  **Free-text prompts** are optional, with two encoders in the Text tab:
  - *small* (default): [intsuc's MiniLM student](https://huggingface.co/intsuc/Llama-3-ARDY-Mini-Core40-Browser)
    (all-MiniLM-L6-v2 distilled to ARDY's conditions, fp32 ONNX ~110 MB gunzipped in the browser), loaded from its own
    Hugging Face repo; cosine 0.98 (root) / 0.96 (body) against ARDY's projections of the real 8B feature. Any device.
  - *exact*: ARDY's own Llama-3 LLM2Vec encoder as INT4 ONNX in three parts (5 GB,
    [sprited/ardy-web-onnx/text-encoder](https://huggingface.co/sprited/ardy-web-onnx/tree/main/text-encoder)) run in
    three Web Workers, one onnxruntime-web instance each (a single instance cannot hold more than ~2 GB of weights).
    Built by `build_text_encoder_onnx.py` (ARDY's merged weights, 4-bit MatMulNBits; the source graph's causal mask
    replaced by a bidirectional one) and `split_text_encoder.py`. Needs a 16 GB+ desktop and pushes it to swap.

## Measured (M1 Pro 16 GB, Chrome 152, fp32 graph)

| | |
|---|---|
| one window (2 s of motion) on WebGPU | 270–310 ms (~7x real time); first window ~0.9 s |
| WASM fallback | ~3.6 s per window, not real time |
| WebGPU session creation | 3–12 s; model download from Hugging Face ~10 MB/s here |
| window-to-window seams | joint jumps within the normal per-frame range |

## Not ported (yet)

- Kinematic constraints (keyframes, waypoints, end-effectors) and constraint sampling from Bones SEED motion files:
  the graph would need `motion_mask` / `observed_motion` inputs plus the constraint-to-feature logic from
  `gen_constraints.py`.
- Free-text prompts (needs a text-encoder service), G1 / SOMA models (separate exports), multiple samples, other
  denoising-step counts or history lengths (baked into the graph), `motion_correction` post-processing (C++).
- Size: 813 MB fp32. `onnxconverter-common` fp16 and onnxruntime's 8-bit `MatMulNBits` both failed on this graph
  (see git history); exporting the denoiser in half precision from PyTorch is the next thing to try.
