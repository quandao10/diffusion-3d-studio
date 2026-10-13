// Tiny canvas charts (line w/ crosshair tooltip, bar w/ hover tooltip). No dependencies.
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt = (v) => (v === 0 ? "0" : Math.abs(v) >= 1e4 || Math.abs(v) < 1e-2 ? v.toExponential(2) : +v.toPrecision(4) + "");

function setup(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function tooltip(canvas) {
  let tip = canvas.parentElement.querySelector(".tip");
  if (!tip) { tip = document.createElement("div"); tip.className = "tip"; tip.hidden = true; canvas.parentElement.appendChild(tip); }
  return tip;
}

export class LineChart {
  constructor(canvas, { logY = false, xLabel = "step" } = {}) {
    this.c = canvas; this.logY = logY; this.xLabel = xLabel; this.series = []; this.hover = null;
    this.tip = tooltip(canvas);
    canvas.addEventListener("pointermove", (e) => { const r = canvas.getBoundingClientRect(); this.hover = e.clientX - r.left; this.draw(); });
    canvas.addEventListener("pointerleave", () => { this.hover = null; this.tip.hidden = true; this.draw(); });
    new ResizeObserver(() => this.draw()).observe(canvas);
  }
  set(series) { this.series = series; this.draw(); }
  draw() {
    const { ctx, w, h } = setup(this.c);
    const pad = { l: 46, r: 8, t: 6, b: 18 };
    const pts = this.series.flatMap((s) => s.data);
    ctx.font = "10.5px " + css("--mono");
    if (!pts.length) { ctx.fillStyle = css("--text-muted"); ctx.fillText("waiting for data…", pad.l, h / 2); return; }
    const ty = (v) => (this.logY ? Math.log10(Math.max(v, 1e-12)) : v);
    let x0 = Math.min(...pts.map((p) => p[0])), x1 = Math.max(...pts.map((p) => p[0]));
    let ys = pts.map((p) => ty(p[1])).filter(Number.isFinite);
    let y0 = Math.min(...ys), y1 = Math.max(...ys);
    if (this.logY) { y0 = Math.floor(y0); y1 = Math.ceil(y1); } else { const m = (y1 - y0) * 0.08 || 1; y0 = Math.max(0, y0 - m); y1 += m; }
    if (x1 === x0) x1 = x0 + 1;
    if (y1 === y0) y1 = y0 + 1;
    const X = (x) => pad.l + ((x - x0) / (x1 - x0)) * (w - pad.l - pad.r);
    const Y = (y) => h - pad.b - ((ty(y) - y0) / (y1 - y0)) * (h - pad.t - pad.b);
    // grid + axes
    ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1; ctx.fillStyle = css("--text-muted");
    const ticks = this.logY ? Array.from({ length: y1 - y0 + 1 }, (_, i) => y0 + i) : [0, 0.5, 1].map((f) => y0 + f * (y1 - y0));
    const step = Math.ceil(ticks.length / 4);
    ticks.forEach((t, i) => {
      if (i % step) return;
      const yy = h - pad.b - ((t - y0) / (y1 - y0)) * (h - pad.t - pad.b);
      ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(w - pad.r, yy); ctx.stroke();
      ctx.fillText(this.logY ? "1e" + t : fmt(t), 2, yy + 3);
    });
    ctx.fillText(`${this.xLabel} ${Math.round(x0)}`, pad.l, h - 4);
    const lab = String(Math.round(x1));
    ctx.fillText(lab, w - pad.r - ctx.measureText(lab).width, h - 4);
    // lines
    for (const s of this.series) {
      ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.beginPath();
      let started = false;
      for (const [x, y] of s.data) {
        if (!Number.isFinite(ty(y))) continue;
        if (!started) { ctx.moveTo(X(x), Y(y)); started = true; } else ctx.lineTo(X(x), Y(y));
      }
      ctx.stroke();
    }
    // crosshair + tooltip
    if (this.hover != null && this.hover >= pad.l) {
      const xv = x0 + ((this.hover - pad.l) / (w - pad.l - pad.r)) * (x1 - x0);
      ctx.strokeStyle = css("--text-muted"); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(this.hover, pad.t); ctx.lineTo(this.hover, h - pad.b); ctx.stroke();
      const lines = [`${this.xLabel} ${Math.round(xv)}`];
      for (const s of this.series) {
        let best = null;
        for (const p of s.data) if (!best || Math.abs(p[0] - xv) < Math.abs(best[0] - xv)) best = p;
        if (best) {
          lines.push(`${s.name}: ${fmt(best[1])}`);
          ctx.fillStyle = s.color; ctx.strokeStyle = css("--surface-1"); ctx.lineWidth = 2;
          ctx.beginPath(); ctx.arc(X(best[0]), Y(best[1]), 4, 0, 7); ctx.fill(); ctx.stroke();
        }
      }
      this.tip.textContent = lines.join("\n");
      this.tip.hidden = false;
      this.tip.style.left = Math.min(this.hover + 10, w - 150) + "px";
      this.tip.style.top = "22px";
    }
  }
}

export class BarChart {
  constructor(canvas, { unit = "FPS" } = {}) {
    this.c = canvas; this.unit = unit; this.bars = []; this.hover = -1; this.tip = tooltip(canvas);
    canvas.addEventListener("pointermove", (e) => {
      const r = canvas.getBoundingClientRect(); const x = e.clientX - r.left;
      this.hover = this.geo ? this.geo.findIndex((g) => x >= g.x - 4 && x <= g.x + g.w + 4) : -1; this.draw();
    });
    canvas.addEventListener("pointerleave", () => { this.hover = -1; this.tip.hidden = true; this.draw(); });
    new ResizeObserver(() => this.draw()).observe(canvas);
  }
  set(bars) { this.bars = bars; this.draw(); }
  draw() {
    const { ctx, w, h } = setup(this.c);
    ctx.font = "10.5px " + css("--mono");
    if (!this.bars.length) { ctx.fillStyle = css("--text-muted"); ctx.fillText("no measurements yet", 8, h / 2); return; }
    const pad = { l: 8, r: 8, t: 16, b: 18 };
    const max = Math.max(...this.bars.map((b) => b.value), 1);
    const slot = (w - pad.l - pad.r) / this.bars.length;
    const bw = Math.min(46, slot * 0.6);
    this.geo = [];
    ctx.strokeStyle = css("--grid"); ctx.beginPath(); ctx.moveTo(pad.l, h - pad.b + 0.5); ctx.lineTo(w - pad.r, h - pad.b + 0.5); ctx.stroke();
    this.bars.forEach((b, i) => {
      const x = pad.l + i * slot + (slot - bw) / 2;
      const bh = b.value > 0 ? Math.max(2, (b.value / max) * (h - pad.t - pad.b)) : 0;
      const y = h - pad.b - bh;
      ctx.fillStyle = b.value > 0 ? (i === this.hover ? css("--text-primary") : css("--series-1")) : css("--bad");
      if (bh) { ctx.beginPath(); ctx.roundRect(x, y, bw, bh, [4, 4, 0, 0]); ctx.fill(); }
      else { ctx.fillText("OOM", x + bw / 2 - 10, h - pad.b - 4); }
      ctx.fillStyle = css("--text-secondary");
      const v = b.value > 0 ? (b.value >= 100 ? Math.round(b.value) : b.value.toFixed(1)) : "";
      ctx.fillText(String(v), x + bw / 2 - ctx.measureText(String(v)).width / 2, y - 3);
      ctx.fillStyle = css("--text-muted");
      ctx.fillText(b.label, x + bw / 2 - ctx.measureText(b.label).width / 2, h - 5);
      this.geo.push({ x, w: bw });
    });
    if (this.hover >= 0) {
      const b = this.bars[this.hover], g = this.geo[this.hover];
      this.tip.textContent = b.tip || `${b.label}: ${b.value} ${this.unit}`;
      this.tip.hidden = false; this.tip.style.left = Math.min(g.x, w - 180) + "px"; this.tip.style.top = "26px";
    }
  }
}
