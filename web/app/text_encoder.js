// Optional free-text prompts: the Llama-3 LLM2Vec text encoder as INT4 ONNX (TREE Industries' export), run in the
// browser with onnxruntime-web. Tokenization mirrors the encoder's validation records:
//   <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>, left-padded with <|eot_id|> to 64,
//   attention_mask on real tokens, embed_mask on the prompt tokens + final <|eot_id|>. Output: [1, 4096] (unnormalized).
// The 8B graph is split in two halves that run in two Web Workers (one onnxruntime-web instance each, 4 GB wasm heap limit).
import { confirmDownload } from './models.js';

const params = new URLSearchParams(location.search);
export const ENCODER = {
  id: 'llama3-llm2vec-int4', label: 'Llama-3-8B LLM2Vec text encoder (INT4 ONNX, two halves)', size_mb: 4980, seq: 64,
  hf: 'https://huggingface.co/sprited/ardy-web-onnx/resolve/main/text-encoder/',
};
function base() { const p = params.get('encoder'); if (p === 'local') return new URL('models/encoder/', location.href).href; if (p && p !== 'hf') return p.endsWith('/') ? p : p + '/'; return ENCODER.hf; }

let tokenizer = null, backend = null, manifest = null; const workers = {};
export const textEncoderState = () => ({ loaded: Object.keys(workers).length > 0 && Object.values(workers).every(w => w.ready), backend });

async function loadTokenizer(b) {
  const tf = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2/dist/transformers.min.js');
  const [tokJSON, tokCfg] = await Promise.all([fetch(b + 'tokenizer/tokenizer.json').then(r => r.json()), fetch(b + 'tokenizer/tokenizer_config.json').then(r => r.json())]);
  tokenizer = new tf.PreTrainedTokenizer(tokJSON, tokCfg);
  return tokenizer;
}
function startWorker(part, b, ep, onProgress) {
  return new Promise((resolve, reject) => {
    const w = new Worker(new URL('./encoder_worker.js', import.meta.url)); workers[part] = { w, ready: false, pending: new Map(), next: 0 };
    w.onmessage = (e) => { const m = e.data;
      if (m.type === 'progress') onProgress && onProgress({ part, ...m });
      else if (m.type === 'ready') { workers[part].ready = true; workers[part].inputs = m.inputs; workers[part].outputs = m.outputs; resolve(); }
      else if (m.type === 'result' || m.type === 'error') { const p = workers[part].pending.get(m.id); if (p) { workers[part].pending.delete(m.id); m.type === 'result' ? p.resolve(m) : p.reject(new Error(m.message)); } else if (m.type === 'error') reject(new Error(m.message)); } };
    w.onerror = (e) => reject(new Error(e.message || 'worker error'));
    w.postMessage({ type: 'load', part, base: b, ep });
  });
}
function runWorker(part, feeds) { const st = workers[part]; const id = ++st.next; return new Promise((resolve, reject) => { st.pending.set(id, { resolve, reject }); st.w.postMessage({ type: 'run', id, feeds }); }); }

/** Downloads (with consent, cached) both halves into two workers. Returns false if the user declined. */
export async function loadTextEncoder(onProgress) {
  if (textEncoderState().loaded) return true;
  const b = base();
  manifest = await (await fetch(b + 'encoder.json')).json();
  if (!(await confirmDownload(ENCODER.id, b + 'onnx/part_a.onnx.data', ENCODER))) return false;
  onProgress && onProgress({ stage: 'tokenizer' });
  await loadTokenizer(b);
  const ep = params.get('backend') || (navigator.gpu ? 'webgpu' : 'wasm'); backend = ep;
  for (const part of manifest.parts) await startWorker(part.name, b, ep, onProgress);   // sequential: two 2.3 GB downloads
  return true;
}
export function tokenizePrompt(prompt) {
  const ids = tokenizer.encode(`<|start_header_id|>user<|end_header_id|>\n\n${prompt.trim()}<|eot_id|>`, { add_special_tokens: false });
  const bos = tokenizer.model.tokens_to_ids.get('<|begin_of_text|>') ?? 128000, eot = tokenizer.model.tokens_to_ids.get('<|eot_id|>') ?? 128009;
  const headerEnd = tokenizer.model.tokens_to_ids.get('<|end_header_id|>') ?? 128007;
  let seq = [bos, ...ids]; if (seq.length > ENCODER.seq) seq = seq.slice(0, ENCODER.seq - 1).concat([eot]);   // truncate right, keep eot
  const L = ENCODER.seq, pad = L - seq.length;
  const input_ids = new BigInt64Array(L).fill(BigInt(eot)), attention_mask = new BigInt64Array(L), embed_mask = new BigInt64Array(L);
  // embed_mask: tokens after "<|end_header_id|>\n\n" (the prompt itself + final eot)
  const hdr = seq.indexOf(headerEnd); const promptStart = hdr >= 0 ? hdr + 2 : 0;
  seq.forEach((t, i) => { input_ids[pad + i] = BigInt(t); attention_mask[pad + i] = 1n; if (i >= promptStart) embed_mask[pad + i] = 1n; });
  return { input_ids, attention_mask, embed_mask };
}

/** Prompt -> Float32Array(4096) text feature for the ARDY window graph (part A then part B). */
export async function encodePrompt(prompt) {
  if (!textEncoderState().loaded) throw new Error('text encoder not loaded');
  const t = tokenizePrompt(prompt), L = ENCODER.seq;
  const feeds = { input_ids: { type: 'int64', data: t.input_ids, dims: [1, L] }, attention_mask: { type: 'int64', data: t.attention_mask, dims: [1, L] }, embed_mask: { type: 'int64', data: t.embed_mask, dims: [1, L] } };
  const t0 = performance.now();
  const a = await runWorker('part_a', feeds);
  const b = await runWorker('part_b', { ...feeds, ...a.outputs });
  const out = b.outputs.text_embedding;
  return { feat: Float32Array.from(out.data), ms: performance.now() - t0, partMs: [a.ms, b.ms] };
}
