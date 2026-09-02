// ARDY in the browser: one ONNX graph per autoregressive window (see ardy/exports/web.py), driven by
// onnxruntime-web on WebGPU, prompts from a precomputed embedding library, three.js stick-figure playback.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const MODELS = 'models/';
const $ = (id) => document.getElementById(id);
const status = (s) => { $('status').textContent = s; console.log('[ardy-web]', s); };

// ---------------------------------------------------------------- assets
async function fetchJSON(url) { const r = await fetch(url); if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.json(); }
async function fetchBinary(url, onProgress) {
  const r = await fetch(url); if (!r.ok) throw new Error(`${url}: ${r.status}`);
  const total = +r.headers.get('Content-Length') || 0; const reader = r.body.getReader(); const chunks = []; let got = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); got += value.length; if (onProgress) onProgress(got, total); }
  const out = new Uint8Array(got); let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; } return out;
}

const cfg = await fetchJSON(MODELS + 'window.json');
const skeleton = await fetchJSON(MODELS + 'skeleton.json');
const promptsMeta = await fetchJSON(MODELS + 'prompts.json');
const promptFeats = new Float32Array((await fetchBinary(MODELS + 'prompts.bin')).buffer);
const { history_frames: H, gen_frames: G, fps: FPS, token_dim: TOKEN_DIM, motion_dim: MOTION_DIM, llm_dim: LLM_DIM } = cfg;
const T = H + G, J = skeleton.parents.length, NOISE_TOKENS = G / 4 | 0;
for (const [i, p] of promptsMeta.prompts.entries()) { const o = document.createElement('option'); o.value = i; o.textContent = p; $('prompt').appendChild(o); }

// ---------------------------------------------------------------- onnxruntime session (WebGPU, wasm fallback)
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/';
const modelName = cfg.onnx || 'window.onnx';
const bytes = await fetchBinary(MODELS + modelName, (got, total) => { $('progress').value = total ? 100 * got / total : 0; status(`Downloading model ${(got / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`); });
let session, backend;
for (const ep of navigator.gpu ? ['webgpu', 'wasm'] : ['wasm']) {
  try { status(`Creating ${ep} session…`); const t0 = performance.now();
    session = await ort.InferenceSession.create(bytes, { executionProviders: [ep], graphOptimizationLevel: 'all' });
    backend = ep; status(`Ready on ${ep} (session ${((performance.now() - t0) / 1000).toFixed(1)} s)`); break;
  } catch (e) { console.warn(ep, 'failed', e); }
}
if (!session) { status('No backend available (WebGPU or WASM).'); throw new Error('no backend'); }
$('progress').hidden = true;

// ---------------------------------------------------------------- generation state
const state = { history: Float32Array.from(cfg.init_history.flat()), queue: [], generating: false, producing: false, playing: false, frame: 0, windowMs: 0, windows: 0 };
function randn(n) { const a = new Float32Array(n); for (let i = 0; i < n; i += 2) { const u = 1 - Math.random(), v = Math.random(), r = Math.sqrt(-2 * Math.log(u)); a[i] = r * Math.cos(2 * Math.PI * v); if (i + 1 < n) a[i + 1] = r * Math.sin(2 * Math.PI * v); } return a; }
async function generateWindow() {
  if (state.generating) return; state.generating = true;
  const pi = +$('prompt').value, w = +$('cfg').value;
  const feeds = {
    history: new ort.Tensor('float32', state.history, [1, H, MOTION_DIM]),
    text_feat: new ort.Tensor('float32', promptFeats.subarray(pi * LLM_DIM, (pi + 1) * LLM_DIM), [1, 1, LLM_DIM]),
    noise: new ort.Tensor('float32', randn(NOISE_TOKENS * TOKEN_DIM), [1, NOISE_TOKENS, TOKEN_DIM]),
    cfg_weight_text: new ort.Tensor('float32', Float32Array.from([w]), [1]),
    cfg_weight_cstr: new ort.Tensor('float32', Float32Array.from([1.5]), [1]),
  };
  const t0 = performance.now();
  try {
    const out = await session.run(feeds);
    state.windowMs = performance.now() - t0; state.windows++;
    const motion = out.motion.data, joints = out.joints.data;
    for (let f = H; f < T; f++) state.queue.push(joints.slice(f * J * 3, (f + 1) * J * 3));   // new frames only
    state.history = Float32Array.from(motion.subarray((T - H) * MOTION_DIM, T * MOTION_DIM));  // last H frames (world space)
  } catch (e) { status('Generation failed: ' + (e.message || e)); console.error(e); state.playing = false; }
  state.generating = false;
}

// ---------------------------------------------------------------- three.js viewer
const view = $('view');
const renderer = new THREE.WebGLRenderer({ antialias: true }); renderer.setPixelRatio(devicePixelRatio); view.appendChild(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x101318); scene.fog = new THREE.Fog(0x101318, 12, 30);
const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 100); camera.position.set(3.2, 2.0, 4.0);
const controls = new OrbitControls(camera, renderer.domElement); controls.target.set(0, 0.9, 0); controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x334455, 1.1)); const sun = new THREE.DirectionalLight(0xffffff, 1.2); sun.position.set(3, 6, 2); scene.add(sun);
scene.add(new THREE.GridHelper(40, 40, 0x2a3140, 0x1c212b));
const jointMesh = new THREE.InstancedMesh(new THREE.SphereGeometry(0.035, 12, 10), new THREE.MeshStandardMaterial({ color: 0x8ec5ff }), J); scene.add(jointMesh);
const bonePairs = []; skeleton.parents.forEach((p, i) => { if (p >= 0) bonePairs.push([i, p]); });
const boneGeom = new THREE.BufferGeometry(); boneGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(bonePairs.length * 6), 3));
const bones = new THREE.LineSegments(boneGeom, new THREE.LineBasicMaterial({ color: 0xffffff })); scene.add(bones);
const rootTrail = [];
const m4 = new THREE.Matrix4(), v3 = new THREE.Vector3();
function drawFrame(joints) {
  const pos = boneGeom.attributes.position.array;
  for (let j = 0; j < J; j++) { m4.makeTranslation(joints[j * 3], joints[j * 3 + 1], joints[j * 3 + 2]); jointMesh.setMatrixAt(j, m4); }
  jointMesh.instanceMatrix.needsUpdate = true;
  bonePairs.forEach(([a, b], k) => { pos.set([joints[a * 3], joints[a * 3 + 1], joints[a * 3 + 2], joints[b * 3], joints[b * 3 + 1], joints[b * 3 + 2]], k * 6); });
  boneGeom.attributes.position.needsUpdate = true;
  v3.set(joints[0], 0.9, joints[2]); controls.target.lerp(v3, 0.05); camera.position.add(v3.clone().sub(controls.target).multiplyScalar(0.05));
}
function resize() { const w = view.clientWidth, h = view.clientHeight; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); }
addEventListener('resize', resize); resize();
drawFrame(Float32Array.from(cfg.init_history.length ? Array(J * 3).fill(0) : []));

// ---------------------------------------------------------------- playback loop (FPS frames/s), streaming windows ahead
let last = performance.now(), acc = 0, shown = 0, fpsT = performance.now(), fpsN = 0, fps = 0;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function producer() {   // keeps ~2 windows of frames buffered ahead of playback; independent of rendering
  if (state.producing) return; state.producing = true;
  try { while (state.playing) { if (state.queue.length < 2 * G) await generateWindow(); else await sleep(25); } }
  finally { state.producing = false; }
}
function tick(now) {
  requestAnimationFrame(tick);
  acc += now - last; last = now;
  const dt = 1000 / FPS;
  if (acc > 4 * dt) acc = 4 * dt;  // after a stall, skip ahead at most a few frames instead of draining the buffer
  while (acc >= dt) { acc -= dt; if (state.playing && state.queue.length) { drawFrame(state.queue.shift()); shown++; } }
  fpsN++; if (now - fpsT > 500) { fps = fpsN * 1000 / (now - fpsT); fpsN = 0; fpsT = now; }
  controls.update(); renderer.render(scene, camera);
}
setInterval(() => {   // stats are updated on a timer: requestAnimationFrame is throttled in background tabs
  $('stats').textContent = `backend  ${backend}\nwindow   ${state.windowMs.toFixed(0)} ms for ${G} frames (${(G / FPS).toFixed(1)} s) → ${state.windowMs ? ((G / FPS) * 1000 / state.windowMs).toFixed(1) : '–'}x real time\nbuffer   ${state.queue.length} frames · shown ${shown} · windows ${state.windows}\nrender   ${fps.toFixed(0)} fps`;
}, 250);
requestAnimationFrame(tick);

// ---------------------------------------------------------------- UI
$('cfg').oninput = () => { $('cfgv').textContent = $('cfg').value; };
$('toggle').disabled = $('reset').disabled = false;
$('toggle').onclick = () => { state.playing = !state.playing; $('toggle').textContent = state.playing ? 'Pause' : 'Start'; if (state.playing) producer(); };
$('reset').onclick = () => { state.history = Float32Array.from(cfg.init_history.flat()); state.queue.length = 0; };
status(`Ready on ${backend}. Pick a prompt and press Start.`);
