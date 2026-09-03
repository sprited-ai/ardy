// Web Worker hosting ONE half of the text encoder in its own onnxruntime-web instance (each wasm32 instance is limited
// to a 4 GB heap, so an 8B-parameter int4 encoder is split in two). Messages: {type:'load', part, base, ep}
// -> {type:'ready'|'progress'|'error'}; {type:'run', feeds:{name:{data,dims,type}}} -> {type:'result', outputs}.
importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/ort.webgpu.min.js');
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/';
const CACHE = 'ardy-web-models-v1';
let session = null;
async function fetchCached(url, onProgress) {
  let cache = null; try { cache = await caches.open(CACHE); } catch (_) {}
  if (cache) { const hit = await cache.match(url); if (hit) { onProgress({ cached: true }); return new Uint8Array(await hit.arrayBuffer()); } }
  const r = await fetch(url); if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const total = +r.headers.get('Content-Length') || 0; const reader = r.body.getReader(); const chunks = []; let got = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); got += value.length; onProgress({ got, total }); }
  const out = new Uint8Array(got); let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; }
  chunks.length = 0;
  if (cache) { try { await cache.put(url, new Response(out, { headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': String(got) } })); } catch (e) { console.warn('cache put failed', e); } }
  return out;
}
self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'load') {
      const progress = (p) => self.postMessage({ type: 'progress', part: msg.part, ...p });
      const graph = await fetchCached(`${msg.base}onnx/${msg.part}.onnx`, () => {});
      const data = await fetchCached(`${msg.base}onnx/${msg.part}.onnx.data`, progress);
      self.postMessage({ type: 'progress', part: msg.part, stage: 'session' });
      session = await ort.InferenceSession.create(graph, { executionProviders: [msg.ep], graphOptimizationLevel: 'all', externalData: [{ path: `${msg.part}.onnx.data`, data }] });
      // ORT holds its own copies now; detach ours immediately (a dropped ArrayBuffer waits for a rare major GC otherwise)
      for (const u8 of [graph, data]) { try { u8.buffer.transfer(0); } catch (_) {} }
      self.postMessage({ type: 'ready', part: msg.part, inputs: session.inputNames, outputs: session.outputNames });
    } else if (msg.type === 'run') {
      const feeds = {}; for (const [k, v] of Object.entries(msg.feeds)) if (session.inputNames.includes(k)) feeds[k] = new ort.Tensor(v.type, v.data, v.dims);
      const t0 = performance.now(); const out = await session.run(feeds); const outputs = {};
      for (const [k, t] of Object.entries(out)) outputs[k] = { type: t.type, dims: t.dims, data: t.data };
      self.postMessage({ type: 'result', id: msg.id, outputs, ms: performance.now() - t0 });
    }
  } catch (err) { self.postMessage({ type: 'error', id: msg.id, part: msg.part, message: String(err && err.message || err) }); }
};
