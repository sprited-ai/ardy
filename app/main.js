// Standalone port of the ARDY interactive demo: model download (Hugging Face, cached), generation engine,
// viewer, timeline and the demo's tabs. No server involved.
import { MODELS, DEFAULT_MODEL, resolveBase, fetchModel, confirmDownload, isCached, clearCache, deviceInfo } from './models.js';
import { Engine } from './engine.js';
import { Viewer } from './viewer.js';
import { Timeline, PROMPT_COLORS } from './timeline.js';
import { ENCODER, loadTextEncoder, encodePrompt, textEncoderState } from './text_encoder.js';

const $ = (id) => document.getElementById(id);
const status = (s, ok) => { $('status').textContent = s; $('dot').classList.toggle('ok', !!ok); console.log('[ardy-web]', s); };
const toast = (title, body, color = '', ms = 3000) => { const t = document.createElement('div'); t.className = 'toast ' + color; t.innerHTML = `<b></b><span></span>`; t.querySelector('b').textContent = title; t.querySelector('span').textContent = body; $('toasts').appendChild(t); setTimeout(() => t.remove(), ms); };
const params = new URLSearchParams(location.search);
async function fetchJSON(u) { const r = await fetch(u); if (!r.ok) throw new Error(`${u}: ${r.status}`); return r.json(); }

// ---------------------------------------------------------------- tabs / theme
for (const b of $('tabs').querySelectorAll('button')) b.onclick = () => { for (const x of $('tabs').querySelectorAll('button')) x.classList.toggle('on', x === b); for (const t of document.querySelectorAll('.tab')) t.classList.toggle('on', t.id === 'tab-' + b.dataset.tab); };
const setDark = (d) => { document.documentElement.classList.toggle('dark', d); $('darkMode').checked = d; viewer && viewer.setDark(d); timeline && timeline.draw(); try { localStorage.setItem('ardy-web-dark', d ? '1' : '0'); } catch (_) {} };
$('darkToggle').onclick = () => setDark(!document.documentElement.classList.contains('dark'));
$('darkMode').onchange = () => setDark($('darkMode').checked);

// ---------------------------------------------------------------- model load
const modelId = params.get('model') && MODELS[params.get('model')] ? params.get('model') : DEFAULT_MODEL;
for (const [id, m] of Object.entries(MODELS)) { const o = document.createElement('option'); o.value = id; o.textContent = m.label; o.selected = id === modelId; $('modelSel').appendChild(o); }
$('modelSel').onchange = () => { location.search = '?model=' + $('modelSel').value; };
const base = await resolveBase(modelId);
$('modelSrc').value = base.startsWith('http') && !base.startsWith(location.origin) ? 'Hugging Face' : 'local files';
const cfg = await fetchJSON(base + 'window.json'); const skeleton = await fetchJSON(base + 'skeleton.json'); const promptsMeta = await fetchJSON(base + 'prompts.json');
const promptFeats = new Float32Array((await fetchModel(base + 'prompts.bin')).buffer);
let skin = null;
try { const man = await fetchJSON(base + `skin_${skeleton.name}.json`); const buf = (await fetchModel(base + `skin_${skeleton.name}.bin`)).buffer;
  const view = (k) => { const a = man.arrays[k]; const C = { float32: Float32Array, uint16: Uint16Array, uint32: Uint32Array, uint8: Uint8Array }[a.dtype]; return new C(buf, a.offset, a.length); };
  skin = { V: man.arrays.bind_vertices.shape[0], W: man.arrays.lbs_weights.shape[1], verts: view('bind_vertices'), faces: view('faces'), inv: view('bind_rig_transform_inv'), idx: view('lbs_indices'), w: view('lbs_weights') };
} catch (e) { console.warn('no skin data', e); }
let viewer = new Viewer($('view'), skeleton, skin); let timeline = new Timeline($('tl'), { fps: cfg.fps });
document.documentElement.style.setProperty('--tl-h', timeline.height + 'px'); viewer.resize(); timeline.draw();
try { setDark(localStorage.getItem('ardy-web-dark') === '1'); } catch (_) {}
$('nativeFps').value = cfg.fps; $('hist').value = cfg.history_frames; $('steps').value = cfg.num_denoising_steps;
for (const [i, p] of promptsMeta.prompts.entries()) { const o = document.createElement('option'); o.value = i; o.textContent = p; $('promptSel').appendChild(o); const b = document.createElement('button'); b.textContent = p; b.onclick = () => { $('promptSel').value = i; updatePrompt(); }; $('presets').appendChild(b); }

const modelUrl = base + (cfg.onnx || 'window.onnx');
$('cacheState').value = (await isCached(modelUrl)) ? 'cached' : 'not cached';
if (!(await confirmDownload(modelId, modelUrl))) { status('Download declined. Reload to try again.'); throw new Error('declined'); }
status('downloading model…');
const bytes = await fetchModel(modelUrl, (p) => { if (p.cached) { status('loading cached model…'); $('loading').style.transform = 'scaleX(1)'; } else { $('loading').style.transform = `scaleX(${p.total ? p.got / p.total : 0})`; status(`downloading ${(p.got / 1e6).toFixed(0)} / ${(p.total / 1e6).toFixed(0)} MB`); } });
$('cacheState').value = 'cached';
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/';
let session = null, backend = null;
for (const ep of (params.get('backend') ? [params.get('backend')] : navigator.gpu ? ['webgpu', 'wasm'] : ['wasm'])) {
  try { status(`creating ${ep} session…`); const t0 = performance.now(); session = await ort.InferenceSession.create(bytes, { executionProviders: [ep], graphOptimizationLevel: 'all' }); backend = ep; status(`ready on ${ep} (${((performance.now() - t0) / 1000).toFixed(1)} s)`, true); break; } catch (e) { console.warn(ep, e); }
}
if (!session) { status('no WebGPU/WASM backend available'); throw new Error('no backend'); }
$('loading').style.transform = 'scaleX(0)'; $('backend').value = backend;
const engine = new Engine(session, cfg, skeleton, promptFeats, promptsMeta.prompts); engine.backend = backend;

// ---------------------------------------------------------------- playback state
const state = { frame: 0, playing: false, fpsSamples: [], lastTick: performance.now(), acc: 0, nowPlaying: null, nowCompute: null, computeKind: null };
const G = cfg.gen_frames;
function schedule(kind) {   // plan the next window if playback is close to the generated end
  if (engine.busy) return false;
  const thresh = +$('replanThresh').value || 30;
  if (engine.maxFrame < 0) { engine.schedulePrompt(0, +$('promptSel').value); engine.generateWindow({ historyEnd: -1 }).then(afterWindow); return true; }
  if (kind === 'auto' && !$('autoReplan').checked) return false;
  if (kind === 'auto' && engine.maxFrame - state.frame > thresh) return false;
  engine.generateWindow({ historyEnd: engine.maxFrame }).then(afterWindow); return true;
}
function afterWindow(n) { if (n) { $('frameIdx').max = engine.maxFrame; setFrame(state.frame); } }
function replanFrom(frame, promptIdx) {   // the demo's replan: keep frames up to `frame`, regenerate after it
  const historyEnd = Math.max(-1, Math.min(engine.maxFrame, frame));
  if (promptIdx !== undefined) engine.schedulePrompt(historyEnd + 1, promptIdx);
  engine.generateWindow({ historyEnd }).then(afterWindow);
}
function updatePrompt() {
  const pi = +$('promptSel').value; $('activePrompt').textContent = promptsMeta.prompts[pi];
  if (engine.maxFrame < 0) { engine.schedulePrompt(0, pi); toast('Text prompt updated', promptsMeta.prompts[pi], 'green'); return; }
  const next = state.frame + 1; engine.schedulePrompt(next, pi); toast('Text prompt updated', `New prompt starts at frame ${next}`, ''); replanFrom(state.frame, pi);
}
$('updatePrompt').onclick = updatePrompt;
// free-text prompts (optional 5 GB encoder download)
$('enableEncoder').onclick = async () => {
  $('enableEncoder').disabled = true;
  try {
    const ok = await loadTextEncoder((p) => {
      if (p.got !== undefined) { $('encStatus').textContent = `${p.part || ''}: downloading ${(p.got / 1e6).toFixed(0)} / ${(p.total / 1e6).toFixed(0)} MB`; $('loading').style.transform = `scaleX(${p.total ? p.got / p.total : 0})`; }
      else if (p.cached) $('encStatus').textContent = `${p.part || ''}: loading from cache…`;
      else $('encStatus').textContent = `${p.part ? p.part + ': ' : ''}${p.stage || ''}${p.ep ? ' (' + p.ep + ')' : ''}…`;
    });
    $('loading').style.transform = 'scaleX(0)';
    if (!ok) { $('encStatus').textContent = 'download declined'; $('enableEncoder').disabled = false; return; }
    $('encStatus').textContent = `text encoder ready on ${textEncoderState().backend}`; $('customPrompt').disabled = $('encodePrompt').disabled = false; toast('Text encoder ready', ENCODER.label, 'green');
  } catch (e) { $('encStatus').textContent = 'failed: ' + (e.message || e); $('enableEncoder').disabled = false; console.error(e); }
};
$('encodePrompt').onclick = async () => {
  const text = $('customPrompt').value.trim(); if (!text) return;
  $('encodePrompt').disabled = true; $('encStatus').textContent = 'encoding…';
  try { const { feat, ms } = await encodePrompt(text); const idx = engine.addPrompt(text, feat); promptsMeta.prompts.push(text);
    const o = document.createElement('option'); o.value = idx; o.textContent = text; $('promptSel').appendChild(o); $('promptSel').value = idx;
    $('encStatus').textContent = `encoded in ${ms.toFixed(0)} ms`; updatePrompt();
  } catch (e) { $('encStatus').textContent = 'encode failed: ' + (e.message || e); console.error(e); } finally { $('encodePrompt').disabled = false; }
};
$('restartBtn').onclick = () => { engine.reset(); state.frame = 0; state.nowPlaying = state.nowCompute = null; $('activePrompt').textContent = promptsMeta.prompts[+$('promptSel').value]; viewer.setStartArrow([0, 0, 0], 0); schedule('restart'); };
$('restartNowBtn').onclick = () => replanFrom(state.frame, +$('promptSel').value);
$('cfg').oninput = () => { $('cfgv').textContent = $('cfg').value; engine.cfgText = +$('cfg').value; };
$('seed').onchange = () => engine.setSeed(+$('seed').value);
$('playBtn').disabled = false;
$('playBtn').onclick = () => { state.playing = !state.playing; $('playBtn').textContent = state.playing ? 'Pause' : 'Play'; if (state.playing && engine.maxFrame < 0) schedule('start'); };
$('nextBtn').onclick = () => setFrame(state.frame + 1); $('prevBtn').onclick = () => setFrame(state.frame - 1);
$('frameIdx').onchange = () => setFrame(+$('frameIdx').value);
timeline.onScrub = (f) => setFrame(f);
$('showMesh').onchange = () => { viewer.showMesh = $('showMesh').checked; viewer.setPose(engine.frames[state.frame]); };
$('showSkel').onchange = () => { viewer.showSkeleton = $('showSkel').checked; viewer.setPose(engine.frames[state.frame]); };
$('meshOpacity').oninput = () => { if (viewer.skinMat) viewer.skinMat.opacity = +$('meshOpacity').value; };
$('followCam').onchange = () => { viewer.follow = $('followCam').checked; };
$('clearCache').onclick = async () => { await clearCache(); $('cacheState').value = 'not cached'; toast('Cache cleared', 'The model will be downloaded again next time.'); };
$('exportSession').onclick = () => { const data = { model: modelId, prompts: engine.promptSchedule.map(([f, i]) => ({ frame: f, prompt: promptsMeta.prompts[i], index: i })), frame: state.frame, cfg_text: engine.cfgText, seed: engine.seed, joints: engine.frames.map(f => Array.from(f.joints)) }; download('ardy-session.json', JSON.stringify(data)); };
$('exportMotion').onclick = () => { const rows = ['frame,' + skeleton.joint_names.map(n => `${n}_x,${n}_y,${n}_z`).join(',')]; engine.frames.forEach((f, i) => rows.push(i + ',' + Array.from(f.joints).map(v => v.toFixed(5)).join(','))); download('ardy-motion.csv', rows.join('\n')); };
$('importSession').onclick = () => $('importFile').click();
$('importFile').onchange = async () => { const f = $('importFile').files[0]; if (!f) return; const data = JSON.parse(await f.text()); engine.reset(); for (const p of data.prompts || []) { const idx = promptsMeta.prompts.indexOf(p.prompt); if (idx >= 0) engine.promptSchedule.push([p.frame, idx]); } if (data.joints && data.joints.length) { engine.frames = data.joints.map((j, i) => ({ joints: Float32Array.from(j), rots: null, motion: null, prompt: engine.promptAt(i) ?? 0 })); engine.prehistory = engine.initHistory; } state.frame = 0; toast('Session imported', `${engine.frames.length} frames, ${engine.promptSchedule.length} prompts`, 'green'); };
function download(name, text) { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text])); a.download = name; a.click(); }

function setFrame(f) {
  f = Math.max(0, Math.min(engine.maxFrame < 0 ? 0 : engine.maxFrame, f | 0)); state.frame = f; $('frameIdx').value = f; $('curTime').value = (f / cfg.fps).toFixed(2);
  viewer.setPose(engine.frames[f]); redrawLabels();
}
function redrawLabels() {
  const pi = engine.promptAt(state.frame); const pname = pi === null ? null : promptsMeta.prompts[pi];
  if (pname && pname !== state.nowPlaying) { state.nowPlaying = pname; $('nowPlaying').textContent = pname; toast('Now playing', `${pname} (from frame ${state.frame})`, 'teal'); }
  const c = engine.computeAt(state.frame);
  if (c && c.label !== state.nowCompute) { state.nowCompute = c.label; $('compute').textContent = c.label; }
  timeline.set(state.frame, engine.maxFrame, engine.promptSchedule, promptsMeta.prompts);
}

// ---------------------------------------------------------------- loops
let shownFrames = 0, fpsT = performance.now(), fpsN = 0;
function tick(now) {
  requestAnimationFrame(tick);
  const dt = 1000 / cfg.fps; state.acc += now - state.lastTick; state.lastTick = now; if (state.acc > 4 * dt) state.acc = 4 * dt;
  while (state.acc >= dt) { state.acc -= dt; if (state.playing && state.frame < engine.maxFrame) { setFrame(state.frame + 1); shownFrames++; fpsN++; } }
  if (state.playing) schedule('auto');
  if (now - fpsT > 1000) { $('actualFps').value = (fpsN * 1000 / (now - fpsT)).toFixed(1); fpsN = 0; fpsT = now; }
  viewer.render();
}
requestAnimationFrame(tick);
setInterval(() => {   // Debug folder
  const m = performance.memory; const d = deviceInfo();
  $('debugMem').textContent = (m ? `JS heap ${(m.usedJSHeapSize / 1e9).toFixed(2)} GB (limit ${(m.jsHeapSizeLimit / 1e9).toFixed(1)} GB)\n` : '') + `model ${MODELS[modelId].size_mb} MB in ${backend}` + (d.memoryGB ? ` · device memory ~${d.memoryGB} GB` : '');
  $('debugGen').textContent = `windows ${engine.windows || 0} · last ${engine.lastMs ? engine.lastMs.toFixed(0) + ' ms' : '–'} · frames ${engine.frames.length} · buffer ahead ${Math.max(0, engine.maxFrame - state.frame)}`;
}, 500);
engine.on((ev, data) => { if (ev === 'window' && data.start === 0) viewer.setStartArrow([0, 0, 0], 0); });
$('activePrompt').textContent = promptsMeta.prompts[0];
status(`ready on ${backend}`, true);
window.__ardy = { engine, state, textEncoderState, ENCODER };   // for debugging from the console
toast('Model loaded', `${MODELS[modelId].label} on ${backend}`, 'green');
schedule('initial');
