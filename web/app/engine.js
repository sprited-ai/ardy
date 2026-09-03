// Generation engine: the demo's autoregressive loop (history window, replans, prompt schedule) on top of the
// single-window ONNX graph, all in the browser.
export class Engine {
  constructor(session, cfg, skeleton, promptFeats, prompts) {
    this.session = session; this.cfg = cfg; this.skeleton = skeleton; this.prompts = prompts.slice();
    this.condDim = cfg.cond_dim || 2048;
    this.conds = promptFeats;   // Float32Array(condDim) per prompt (text conditions = root+body embed_text of the LLM2Vec feature)
    this.H = cfg.history_frames; this.G = cfg.gen_frames; this.T = this.H + this.G; this.J = skeleton.parents.length;
    this.D = cfg.motion_dim; this.tokenDim = cfg.token_dim; this.llmDim = cfg.llm_dim; this.fps = cfg.fps;
    this.initHistory = Float32Array.from(cfg.init_history.flat());
    this.reset();
    this.cfgText = 3.5; this.cfgCstr = 1.5; this.seed = 42; this.busy = false; this.listeners = new Set();
    this.rng = mulberry32(this.seed);
  }
  on(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  emit(ev, data) { for (const fn of this.listeners) fn(ev, data); }
  reset() {
    // frames[i] = { joints: Float32Array(J*3), rots: Float32Array(J*9), motion: Float32Array(D) }
    this.frames = []; this.promptSchedule = []; this.computeSchedule = []; this.prehistory = this.initHistory;
  }
  get maxFrame() { return this.frames.length - 1; }
  setSeed(s) { this.seed = s; this.rng = mulberry32(s); }
  addPrompt(text, cond) { this.prompts.push(text); this.conds.push(cond); return this.prompts.length - 1; }
  promptAt(frame) { let p = null; for (const [f, idx] of this.promptSchedule) if (frame >= f) p = idx; return p; }
  computeAt(frame) { let c = null; for (const e of this.computeSchedule) if (frame >= e.start) c = e; return c; }
  schedulePrompt(startFrame, promptIdx) { this.promptSchedule = this.promptSchedule.filter(([f]) => f < startFrame); this.promptSchedule.push([startFrame, promptIdx]); }

  /** History for a window that regenerates everything after `historyEnd` (inclusive end frame index). */
  historyFor(historyEnd) {
    if (historyEnd < 0 || this.frames.length === 0) return { hist: this.prehistory, start: 0 };
    const hist = new Float32Array(this.H * this.D);
    for (let k = 0; k < this.H; k++) { const i = Math.max(0, historyEnd - (this.H - 1) + k); hist.set(this.frames[i].motion, k * this.D); }
    return { hist, start: historyEnd + 1 };
  }
  randn(n) { const a = new Float32Array(n); for (let i = 0; i < n; i += 2) { const u = 1 - this.rng(), v = this.rng(), r = Math.sqrt(-2 * Math.log(u)); a[i] = r * Math.cos(2 * Math.PI * v); if (i + 1 < n) a[i + 1] = r * Math.sin(2 * Math.PI * v); } return a; }

  /** Generate one window whose frames replace everything from `start` on. Returns the frames added. */
  async generateWindow({ historyEnd, promptIdx }) {
    if (this.busy) return null; this.busy = true;
    const { hist, start } = this.historyFor(historyEnd);
    const pi = promptIdx ?? this.promptAt(start) ?? 0;
    const t0 = performance.now();
    try {
      const out = await this.session.run({
        history: new ort.Tensor('float32', hist, [1, this.H, this.D]),
        cond: new ort.Tensor('float32', this.conds[pi], [1, 1, this.condDim]),
        noise: new ort.Tensor('float32', this.randn((this.G / 4 | 0) * this.tokenDim), [1, this.G / 4 | 0, this.tokenDim]),
        cfg_weight_text: new ort.Tensor('float32', Float32Array.from([this.cfgText]), [1]),
        cfg_weight_cstr: new ort.Tensor('float32', Float32Array.from([this.cfgCstr]), [1]),
      });
      const ms = performance.now() - t0, J = this.J, D = this.D;
      const motion = out.motion.data, joints = out.joints.data, rots = out.joint_rotations.data;
      this.frames.length = start;   // drop the old future
      for (let f = this.H; f < this.T; f++) this.frames.push({ joints: joints.slice(f * J * 3, (f + 1) * J * 3), rots: rots.slice(f * J * 9, (f + 1) * J * 9), motion: motion.slice(f * D, (f + 1) * D), prompt: pi });
      this.computeSchedule = this.computeSchedule.filter(e => e.start < start); this.computeSchedule.push({ start, label: `browser ${this.backend} · ${ms.toFixed(0)} ms`, ms });
      this.lastMs = ms; this.windows = (this.windows || 0) + 1;
      this.emit('window', { start, count: this.G, ms, promptIdx: pi });
      return this.G;
    } finally { this.busy = false; }
  }
}
export function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
