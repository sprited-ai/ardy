// Browser compute worker for the viser demo: loads the single-window ONNX graph on WebGPU and serves
// window jobs sent by scripts/interactive_demo/browser_backend.py over a WebSocket.
(async () => {
  const params = new URLSearchParams(location.search);
  const ep = params.get('backend') || 'webgpu';
  const wsUrl = params.get('ws') || `ws://${location.hostname}:${params.get('wsport') || 2334}`;
  const el = document.getElementById('wstatus');
  const status = (s) => { if (el) el.textContent = s; console.log('[worker]', s); };
  const b64ToF32 = (s) => { const bin = atob(s); const u8 = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i); return new Float32Array(u8.buffer); };
  const f32ToB64 = (a) => { const u8 = new Uint8Array(a.buffer, a.byteOffset, a.byteLength); let s = ''; for (let i = 0; i < u8.length; i += 8192) s += String.fromCharCode.apply(null, u8.subarray(i, i + 8192)); return btoa(s); };
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/';
  const cfg = await (await fetch('models/window.json')).json();
  status(`loading ${cfg.onnx || 'window.onnx'} on ${ep}…`);
  const t0 = performance.now();
  const session = await ort.InferenceSession.create('models/' + (cfg.onnx || 'window.onnx'), { executionProviders: [ep], graphOptimizationLevel: 'all' });
  status(`graph ready on ${ep} (${((performance.now() - t0) / 1000).toFixed(1)} s), connecting to ${wsUrl}…`);
  let jobs = 0, ws;
  const connect = () => {
    ws = new WebSocket(wsUrl);
    ws.onopen = () => { ws.send(JSON.stringify({ type: 'hello', backend: ep, agent: navigator.userAgent })); status(`connected (${ep}) · waiting for windows`); };
    ws.onclose = () => { status('server disconnected, retrying…'); setTimeout(connect, 1500); };
    ws.onmessage = async (ev) => {
      const job = JSON.parse(ev.data); if (job.type !== 'job') return;
      try {
        const t1 = performance.now();
        const feeds = {
          history: new ort.Tensor('float32', b64ToF32(job.history), job.history_shape),
          text_feat: new ort.Tensor('float32', b64ToF32(job.text_feat), [1, 1, cfg.llm_dim]),
          noise: new ort.Tensor('float32', b64ToF32(job.noise), job.noise_shape),
          cfg_weight_text: new ort.Tensor('float32', Float32Array.from([job.cfg_text]), [1]),
          cfg_weight_cstr: new ort.Tensor('float32', Float32Array.from([job.cfg_cstr]), [1]),
        };
        const out = await session.run(feeds);
        const ms = Math.round(performance.now() - t1);
        ws.send(JSON.stringify({ type: 'result', id: job.id, shape: out.motion.dims, motion: f32ToB64(out.motion.data), ms }));
        jobs++; status(`connected (${ep}) · ${jobs} windows · last ${ms} ms`);
      } catch (e) { console.error(e); ws.send(JSON.stringify({ type: 'error', id: job.id, message: String(e && e.message || e) })); status('window failed: ' + e); }
    };
  };
  connect();
})();
