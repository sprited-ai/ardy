// Model registry + download with consent and Cache Storage. Nothing here needs a server: files come from
// Hugging Face (sprited/ardy-web-onnx) or, during development, from ./models next to the page.
const HF = 'https://huggingface.co/sprited/ardy-web-onnx/resolve/main/';
export const MODELS = {
  'core-rp-20fps-h40': { label: 'ARDY-Core-RP-20FPS-Horizon40', skeleton: 'cskel27', size_mb: 813, hf: HF + 'core-rp-20fps-h40/' },
};
export const DEFAULT_MODEL = 'core-rp-20fps-h40';

const params = new URLSearchParams(location.search);
export async function resolveBase(modelId) {
  // ?models=local | hf | https://custom/prefix/
  const pref = params.get('models');
  const local = new URL('models/', location.href).href;
  if (pref && pref !== 'local' && pref !== 'hf') return pref.endsWith('/') ? pref : pref + '/';
  if (pref === 'hf') return MODELS[modelId].hf;
  if (pref === 'local') return local;
  try { const r = await fetch(local + 'window.json', { method: 'HEAD' }); if (r.ok) return local; } catch (_) {}
  return MODELS[modelId].hf;
}

export function deviceInfo() {
  const ua = navigator.userAgent, c = navigator.connection || {};
  const mobile = (navigator.userAgentData && navigator.userAgentData.mobile) || /Mobi|Android|iPhone|iPad/i.test(ua) || (navigator.maxTouchPoints > 1 && innerWidth < 900);
  return { mobile, saveData: !!c.saveData, effectiveType: c.effectiveType || null, memoryGB: navigator.deviceMemory || null, webgpu: !!navigator.gpu };
}

const CACHE = 'ardy-web-models-v1';
export async function cachedSize(url) {
  try { const c = await caches.open(CACHE); const r = await c.match(url); if (!r) return 0; const b = await r.clone().arrayBuffer(); return b.byteLength; } catch (_) { return 0; }
}
export async function isCached(url) { try { const c = await caches.open(CACHE); return !!(await c.match(url)); } catch (_) { return false; } }
export async function clearCache() { try { return await caches.delete(CACHE); } catch (_) { return false; } }

/** Fetch a large file with progress, keeping a copy in Cache Storage so the next visit does not download again. */
export async function fetchModel(url, onProgress, { gunzip = false } = {}) {
  let cache = null; try { cache = await caches.open(CACHE); } catch (_) {}
  if (cache) { const hit = await cache.match(url); if (hit) { onProgress && onProgress({ cached: true }); return new Uint8Array(await hit.arrayBuffer()); } }
  const r = await fetch(url); if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const total = gunzip ? 0 : (+r.headers.get('Content-Length') || 0);
  const body = gunzip ? r.body.pipeThrough(new DecompressionStream('gzip')) : r.body;
  const reader = body.getReader(); const chunks = []; let got = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); got += value.length; onProgress && onProgress({ got, total }); }
  const out = new Uint8Array(got); let o = 0; for (const ch of chunks) { out.set(ch, o); o += ch.length; }
  if (cache) { try { await cache.put(url, new Response(out, { headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': String(got) } })); } catch (e) { console.warn('cache put failed', e); } }
  return out;
}

/** Ask before a large download on phones / metered connections (remembered per model in localStorage). */
export async function confirmDownload(modelId, url, descriptor) {
  const info = deviceInfo(); const m = descriptor || MODELS[modelId];
  if (await isCached(url)) return true;
  const key = 'ardy-web-download-ok:' + modelId;
  let remembered = null; try { remembered = localStorage.getItem(key); } catch (_) {}
  if (remembered === 'yes') return true;
  const risky = info.mobile || info.saveData || (info.effectiveType && /2g|3g/.test(info.effectiveType)) || params.get('ask') === '1';
  if (!risky && remembered !== 'ask') return true;   // desktop on a normal connection: just go
  return new Promise((resolve) => {
    const dlg = document.createElement('div'); dlg.className = 'modal';
    dlg.innerHTML = `<div class="modal-card"><h3>Download the model?</h3>
      <p>${m.label} is <b>${m.size_mb} MB</b> and runs entirely in this browser. It is downloaded once and kept in the browser cache.</p>
      <p class="muted">${info.mobile ? 'This looks like a phone or tablet. ' : ''}${info.saveData ? 'Data saver is on. ' : ''}${info.effectiveType ? 'Connection: ' + info.effectiveType + '. ' : ''}${info.memoryGB ? 'Device memory: ~' + info.memoryGB + ' GB. ' : ''}${info.webgpu ? '' : 'WebGPU is not available here, so generation will be slow (WASM).'}</p>
      <label><input type="checkbox" id="dl-remember"> Don't ask again on this device</label>
      <div class="row"><button id="dl-no" class="secondary">Not now</button><button id="dl-yes">Download ${m.size_mb} MB</button></div></div>`;
    document.body.appendChild(dlg);
    dlg.querySelector('#dl-yes').onclick = () => { try { if (dlg.querySelector('#dl-remember').checked) localStorage.setItem(key, 'yes'); } catch (_) {} dlg.remove(); resolve(true); };
    dlg.querySelector('#dl-no').onclick = () => { dlg.remove(); resolve(false); };
  });
}
