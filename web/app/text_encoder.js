// Free-text prompts in the browser. Two encoders produce the denoiser's text conditions (root 1024 + body 1024):
//  - "small": intsuc's MiniLM student (all-MiniLM-L6-v2 distilled to ARDY's conditions), fp32 ONNX ~110 MB, any device.
//  - "exact": ARDY's own Llama-3-8B LLM2Vec encoder as INT4 ONNX in three parts (~5 GB) run in three Web Workers
//    (one onnxruntime-web instance each); it yields the 4096-d feature, projected to conditions by text_proj.onnx.
import { confirmDownload, fetchModel } from './models.js';

const params = new URLSearchParams(location.search);
export const ENCODERS = {
  small: { id: 'minilm-ardy-student', kind: 'small', label: 'MiniLM student (intsuc/Llama-3-ARDY-Mini-Core40-Browser)', size_mb: 110,
           hf: 'https://huggingface.co/intsuc/Llama-3-ARDY-Mini-Core40-Browser/resolve/main/fp32/' },
  exact: { id: 'llama3-llm2vec-int4', kind: 'exact', label: 'Llama-3-8B LLM2Vec text encoder (INT4 ONNX, in parts)', size_mb: 4980, seq: 64,
           hf: 'https://huggingface.co/sprited/ardy-web-onnx/resolve/main/text-encoder/' },
};
export const ENCODER = ENCODERS.exact;   // kept for the download dialog descriptor
function base(kind) { const p = params.get('encoder'); if (p === 'local' && kind === 'exact') return new URL('models/encoder-sprited/', location.href).href; return ENCODERS[kind].hf; }

let state = { kind: null, backend: null, loaded: false, label: '' };
let tokenizer = null, manifest = null, smallSession = null; const workers = {};
export const textEncoderState = () => ({ ...state });

async function transformers() { return import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2/dist/transformers.min.js'); }
async function loadJSON(url, gz) { if (!gz) return (await fetch(url)).json(); const bytes = await fetchModel(url, null, { gunzip: true }); return JSON.parse(new TextDecoder().decode(bytes)); }

// ------------------------------------------------------------------ small (MiniLM student)
async function loadSmall(onProgress) {
  const b = base('small'); const tf = await transformers();
  onProgress && onProgress({ stage: 'tokenizer' });
  const [tokJSON, tokCfg] = await Promise.all([loadJSON(b + 'tokenizer/tokenizer.json.gz', true), loadJSON(b + 'tokenizer/tokenizer_config.json.gz', true)]);
  tokenizer = new tf.PreTrainedTokenizer(tokJSON, tokCfg);
  const bytes = await fetchModel(b + 'text_encoder.onnx.gz', (p) => onProgress && onProgress({ part: 'text_encoder', ...p }), { gunzip: true });
  for (const ep of (params.get('backend') ? [params.get('backend')] : navigator.gpu ? ['webgpu', 'wasm'] : ['wasm'])) {
    try { onProgress && onProgress({ stage: 'session', ep }); smallSession = await ort.InferenceSession.create(bytes, { executionProviders: [ep] }); state.backend = ep; return; } catch (e) { console.warn('small encoder', ep, e); }
  }
  throw new Error('small encoder: no backend');
}
async function encodeSmall(prompt) {
  const enc = tokenizer(prompt, { add_special_tokens: true, return_tensor: false });
  const ids = BigInt64Array.from(enc.input_ids.map(BigInt)), L = ids.length;
  const feeds = { input_ids: new ort.Tensor('int64', ids, [1, L]), attention_mask: new ort.Tensor('int64', BigInt64Array.from(enc.attention_mask.map(BigInt)), [1, L]) };
  if (smallSession.inputNames.includes('token_type_ids')) feeds.token_type_ids = new ort.Tensor('int64', new BigInt64Array(L), [1, L]);
  const t0 = performance.now(); const out = await smallSession.run(feeds);
  return { cond: Float32Array.from(out.text_conditions.data), ms: performance.now() - t0 };
}

// ------------------------------------------------------------------ exact (8B in parts)
function startWorker(part, b, ep, onProgress) {
  return new Promise((resolve, reject) => {
    const w = new Worker(new URL('./encoder_worker.js', import.meta.url)); workers[part] = { w, ready: false, pending: new Map(), next: 0 };
    w.onmessage = (e) => { const m = e.data;
      if (m.type === 'progress') onProgress && onProgress({ part, ...m });
      else if (m.type === 'ready') { workers[part].ready = true; resolve(); }
      else if (m.type === 'result' || m.type === 'error') { const p = workers[part].pending.get(m.id); if (p) { workers[part].pending.delete(m.id); m.type === 'result' ? p.resolve(m) : p.reject(new Error(m.message)); } else if (m.type === 'error') reject(new Error(m.message)); } };
    w.onerror = (e) => reject(new Error(e.message || 'worker error'));
    w.postMessage({ type: 'load', part, base: b, ep });
  });
}
function runWorker(part, feeds) { const st = workers[part]; const id = ++st.next; return new Promise((resolve, reject) => { st.pending.set(id, { resolve, reject }); st.w.postMessage({ type: 'run', id, feeds }); }); }
async function loadExact(onProgress) {
  const b = base('exact'); const tf = await transformers();
  manifest = await loadJSON(b + 'encoder.json');
  onProgress && onProgress({ stage: 'tokenizer' });
  const [tokJSON, tokCfg] = await Promise.all([loadJSON(b + 'tokenizer/tokenizer.json'), loadJSON(b + 'tokenizer/tokenizer_config.json')]);
  tokenizer = new tf.PreTrainedTokenizer(tokJSON, tokCfg);
  const ep = params.get('backend') || (navigator.gpu ? 'webgpu' : 'wasm'); state.backend = ep;
  for (const part of manifest.parts) await startWorker(part.name, b, ep, onProgress);
}
export function tokenizePrompt(prompt) {   // exact encoder: the LLM2Vec / ARDY recipe, left-padded to 64
  const L = ENCODERS.exact.seq;
  const ids = tokenizer.encode(`<|start_header_id|>user<|end_header_id|>\n\n${prompt.trim()}<|eot_id|>`, { add_special_tokens: false });
  const bos = tokenizer.model.tokens_to_ids.get('<|begin_of_text|>') ?? 128000, eot = tokenizer.model.tokens_to_ids.get('<|eot_id|>') ?? 128009, headerEnd = tokenizer.model.tokens_to_ids.get('<|end_header_id|>') ?? 128007;
  let seq = [bos, ...ids]; if (seq.length > L) seq = seq.slice(0, L - 1).concat([eot]);
  const pad = L - seq.length, input_ids = new BigInt64Array(L).fill(BigInt(eot)), attention_mask = new BigInt64Array(L), embed_mask = new BigInt64Array(L);
  const hdr = seq.indexOf(headerEnd), promptStart = hdr >= 0 ? hdr + 2 : 0;
  seq.forEach((t, i) => { input_ids[pad + i] = BigInt(t); attention_mask[pad + i] = 1n; if (i >= promptStart) embed_mask[pad + i] = 1n; });
  return { input_ids, attention_mask, embed_mask };
}
async function encodeExact(prompt) {
  const t = tokenizePrompt(prompt), L = ENCODERS.exact.seq;
  let feeds = { input_ids: { type: 'int64', data: t.input_ids, dims: [1, L] }, attention_mask: { type: 'int64', data: t.attention_mask, dims: [1, L] }, embed_mask: { type: 'int64', data: t.embed_mask, dims: [1, L] } };
  const t0 = performance.now(); let outputs = null;
  for (const part of manifest.parts) { const r = await runWorker(part.name, feeds); outputs = r.outputs; feeds = { ...feeds, ...outputs }; }
  return { feat: Float32Array.from(outputs.text_embedding.data), ms: performance.now() - t0 };
}

// ------------------------------------------------------------------ public API
/** Downloads (with consent, cached) and prepares the chosen encoder. Returns false if the user declined. */
export async function loadTextEncoder(onProgress, kind = 'small') {
  if (state.loaded && state.kind === kind) return true;
  const enc = ENCODERS[kind];
  if (!(await confirmDownload(enc.id, base(kind) + (kind === 'small' ? 'text_encoder.onnx.gz' : 'onnx/part_a.onnx.data'), enc))) return false;
  if (kind === 'small') await loadSmall(onProgress); else await loadExact(onProgress);
  state = { ...state, kind, loaded: true, label: enc.label };
  return true;
}
/** Prompt -> { cond: Float32Array(2048) } (small) or { feat: Float32Array(4096) } (exact), plus timing. */
export async function encodePrompt(prompt) {
  if (!state.loaded) throw new Error('text encoder not loaded');
  return state.kind === 'small' ? encodeSmall(prompt) : encodeExact(prompt);
}
