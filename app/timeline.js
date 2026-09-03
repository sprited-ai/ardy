// Canvas timeline in the demo's layout: frame ruler, "Prompts" row with coloured segments, constraint track rows
// (kept for parity; constraints are not part of the browser graph yet) and a draggable frame cursor.
export const PROMPT_COLORS = ['#2864c8', '#c85028', '#289640', '#b42896', '#967828', '#3c78b4', '#b43c50', '#643cb4', '#288c78', '#a03c78'];
export class Timeline {
  constructor(canvas, opts) {
    this.c = canvas; this.ctx = canvas.getContext('2d'); this.opts = Object.assign({ before: 20, after: 200, fps: 20 }, opts);
    this.tracks = ['Full-Body', '2D Root', 'Left Hand', 'Right Hand', 'Left Foot', 'Right Foot']; this.frame = 0; this.maxFrame = 0; this.prompts = []; this.promptNames = [];
    this.onScrub = null; this.labelW = 70; this.rowH = 16; this.rulerH = 26; this.promptH = 24;
    let dragging = false;
    const pick = (e) => { const r = this.c.getBoundingClientRect(); const x = (e.clientX - r.left) - this.labelW; const f = Math.round(this.start + x / this.pxPerFrame); if (this.onScrub) this.onScrub(Math.max(0, f)); };
    canvas.addEventListener('pointerdown', (e) => { dragging = true; canvas.setPointerCapture(e.pointerId); pick(e); });
    canvas.addEventListener('pointermove', (e) => { if (dragging) pick(e); });
    canvas.addEventListener('pointerup', () => { dragging = false; });
    addEventListener('resize', () => this.draw());
  }
  get height() { return this.rulerH + this.promptH + this.tracks.length * this.rowH + 6; }
  set(frame, maxFrame, prompts, promptNames) { this.frame = frame; this.maxFrame = maxFrame; this.prompts = prompts; this.promptNames = promptNames; this.draw(); }
  draw() {
    const dpr = Math.min(devicePixelRatio, 2), W = this.c.clientWidth, H = this.height;
    if (this.c.width !== W * dpr || this.c.height !== H * dpr) { this.c.width = W * dpr; this.c.height = H * dpr; this.c.style.height = H + 'px'; }
    const ctx = this.ctx; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    const dark = document.documentElement.classList.contains('dark');
    const fg = dark ? '#cfd3da' : '#333', grid = dark ? '#2d3340' : '#e3e6ec', bg = dark ? '#161a20' : '#ffffff';
    ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H);
    this.start = Math.max(0, this.frame - this.opts.before); const span = this.opts.before + this.opts.after; this.pxPerFrame = (W - this.labelW) / span;
    const x = (f) => this.labelW + (f - this.start) * this.pxPerFrame;
    // ruler
    ctx.font = '10px system-ui, sans-serif'; ctx.fillStyle = fg; ctx.textAlign = 'center';
    const step = 9; for (let f = Math.ceil(this.start / step) * step; f <= this.start + span; f += step) { ctx.fillStyle = grid; ctx.fillRect(x(f), this.rulerH, 1, H - this.rulerH); ctx.fillStyle = fg; ctx.fillText(String(f), x(f), 16); }
    // rows
    let y = this.rulerH; ctx.textAlign = 'right'; ctx.font = '11px system-ui, sans-serif';
    ctx.fillStyle = dark ? '#1c2129' : '#f3f4f7'; ctx.fillRect(0, y, W, this.promptH); ctx.fillStyle = fg; ctx.fillText('Prompts', this.labelW - 8, y + 16);
    // prompt segments: from start frame to the next one (or the visible end)
    for (let i = 0; i < this.prompts.length; i++) { const [f0, pi] = this.prompts[i]; const f1 = i + 1 < this.prompts.length ? this.prompts[i + 1][0] : this.start + span + 1;
      const x0 = Math.max(this.labelW, x(f0)), x1 = Math.min(W, x(f1)); if (x1 <= x0) continue;
      ctx.fillStyle = PROMPT_COLORS[i % PROMPT_COLORS.length]; ctx.fillRect(x0, y + 3, x1 - x0, this.promptH - 6);
      ctx.fillStyle = '#fff'; ctx.textAlign = 'left'; ctx.font = 'bold 11px system-ui, sans-serif';
      const label = (this.promptNames[pi] || '') + (f0 > 0 ? '' : ''); ctx.save(); ctx.beginPath(); ctx.rect(x0, y, x1 - x0, this.promptH); ctx.clip(); ctx.fillText(label, x0 + 6, y + 16); ctx.restore(); ctx.textAlign = 'right'; ctx.font = '11px system-ui, sans-serif'; }
    y += this.promptH;
    for (const t of this.tracks) { ctx.fillStyle = grid; ctx.fillRect(this.labelW, y + this.rowH - 1, W - this.labelW, 1); ctx.fillStyle = dark ? '#9aa1ad' : '#666'; ctx.fillText(t, this.labelW - 8, y + 12); y += this.rowH; }
    // generated extent + cursor
    ctx.fillStyle = dark ? 'rgba(80,120,255,.12)' : 'rgba(40,100,200,.08)'; ctx.fillRect(x(0), this.rulerH, Math.max(0, x(this.maxFrame + 1) - x(0)), H - this.rulerH);
    const cx = x(this.frame); ctx.fillStyle = '#2864c8'; ctx.fillRect(cx - 1, this.rulerH, 2, H - this.rulerH);
    ctx.fillRect(cx - 16, 4, 32, 18); ctx.fillStyle = '#fff'; ctx.textAlign = 'center'; ctx.font = 'bold 11px system-ui, sans-serif'; ctx.fillText(String(this.frame), cx, 17);
  }
}
