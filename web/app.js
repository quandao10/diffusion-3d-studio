import { SplatViewer, loadPly, replicate } from "./splat.js";
import { LineChart, BarChart } from "./charts.js";

const $ = (s) => document.querySelector(s);
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const DATA = "data/results";
const MULTS = [1, 2, 4, 8, 16];
const GRAD_KEYS = [["xyz", "position"], ["features_dc", "colour"], ["opacity", "opacity"], ["scaling", "scale"], ["rotation", "rotation"]];

const st = { live: false, runs: [], sel: null, base: null, mult: 1, metrics: [], jobName: null, lastSnapStep: -1, polling: false };

// ------------------------------------------------------------------ viewer
let stats = {};
const viewer = new SplatViewer($("#gl"), {
  onStats: (s) => {
    stats = s;
    if (!st.base) return;
    $("#hud").textContent =
      `${s.fps} FPS (rAF)` + (s.gpuMs != null ? `  ·  GPU ${s.gpuMs.toFixed(2)} ms` : "") +
      `\n${s.count.toLocaleString()} Gaussians  ·  ${s.width}×${s.height}` +
      `\nsort ${s.sortMs.toFixed(0)} ms  ·  ${String(s.renderer).replace(/^ANGLE \((.*)\)$/, "$1").slice(0, 46)}`;
  },
});
viewer.autoRotate = 20;
$("#rotate").onchange = (e) => (viewer.autoRotate = e.target.checked ? 20 : 0);
$("#scalemod").oninput = (e) => { viewer.scaleMod = +e.target.value; $("#scalev").textContent = (+e.target.value).toFixed(2); };
$("#bg").onchange = (e) => {
  viewer.bg = e.target.value.split(",").map(Number);
  $(".canvaswrap").style.background = `rgb(${viewer.bg.map((v) => v * 255).join(",")})`;
};
$("#reset").onclick = () => Object.assign(viewer.cam, { az: 0, el: 15, r: 2.5 });
$("#resscale").onchange = (e) => viewer.setResolutionScale(+e.target.value);

// ------------------------------------------------------------------ charts
const cLoss = new LineChart($("#c-loss"), { logY: true });
const cGrad = new LineChart($("#c-grad"), { logY: true });
const cN = new LineChart($("#c-n"));
const cBres = new BarChart($("#c-bres"));
const cBn = new BarChart($("#c-bn"));
$("#l-grad").innerHTML = GRAD_KEYS.map(([, n], i) => `<span><i style="background:var(--series-${i + 1})"></i>${n}</span>`).join("");

function drawMetrics() {
  const m = st.metrics;
  cLoss.set([{ name: "SDS loss", color: css("--series-1"), data: m.map((r) => [r.step, r.sds_loss]) }]);
  cGrad.set(GRAD_KEYS.map(([k, n], i) => ({ name: n, color: css(`--series-${i + 1}`), data: m.filter((_, j) => j % 2 === 0).map((r) => [r.step, r.grad_norm[k]]) })));
  cN.set([{ name: "Gaussians", color: css("--series-3"), data: m.map((r) => [r.step, r.num_gaussians]) }]);
}

// ------------------------------------------------------------------ runs
async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${url}: ${r.status} ${await r.text()}`);
  return r.json();
}

async function loadRuns() {
  st.runs = st.live ? await fetchJSON("api/runs") : await fetchJSON("data/runs.json").catch(() => []);
  const el = $("#runs");
  if (!st.runs.length) { el.innerHTML = `<p class="dim">no runs yet</p>`; return; }
  el.innerHTML = "";
  for (const r of st.runs) {
    const b = document.createElement("button");
    b.className = "run" + (r.name === st.sel?.name ? " sel" : "");
    b.dataset.name = r.name;
    b.innerHTML = `<img alt="" src="${DATA}/${r.name}/views/view_000.png"><div><b title="${r.prompt}">${r.prompt}</b>
      <span>seed ${r.seed} · ${r.iters} it · ${r.train_s}s · ${(r.num_gaussians_final / 1000).toFixed(1)}k G</span></div>`;
    b.onclick = () => selectRun(r.name);
    el.appendChild(b);
  }
}

async function selectRun(name) {
  const r = st.runs.find((x) => x.name === name);
  if (!r) return;
  st.sel = r;
  document.querySelectorAll(".run").forEach((b) => b.classList.toggle("sel", b.dataset.name === name));
  $("#viewname").textContent = `· “${r.prompt}”`;
  $("#hud").textContent = "loading PLY…";
  st.base = await loadPly(`${DATA}/${name}/${r.ply}`);
  setMult(st.mult);
  // training curves
  const txt = await fetch(`${DATA}/${name}/metrics.jsonl`).then((x) => x.text()).catch(() => "");
  st.metrics = txt.split("\n").filter(Boolean).map((l) => JSON.parse(l));
  drawMetrics();
  // snapshot scrubber
  const steps = [0];
  for (let s = 25; s <= r.iters; s += 25) steps.push(s);
  const frames = [...steps.map((s) => `step_${String(s).padStart(4, "0")}`), "final"];
  $("#scrubwrap").hidden = false;
  const sc = $("#scrub");
  sc.max = frames.length - 1; sc.value = frames.length - 1;
  sc.oninput = () => { $("#snap").src = `${DATA}/${name}/progress/${frames[sc.value]}.png`; $("#scrubv").textContent = frames[sc.value].replace("_", " "); };
  sc.oninput();
  await loadBench(r);
}

async function loadBench(r) {
  let url = r?.has_benchmark ? `${DATA}/${r.name}/benchmark.csv` : "data/evidence/performance.csv";
  const txt = await fetch(url).then((x) => (x.ok ? x.text() : "")).catch(() => "");
  const [head, ...lines] = txt.trim().split(/\r?\n/);
  if (!head) return showBench([]);
  const keys = head.split(",");
  showBench(lines.map((l) => { const v = l.split(","); return Object.fromEntries(keys.map((k, i) => [k, k === "notes" ? v.slice(i).join(",") : v[i]])); }));
}

function showBench(rows) {
  const tb = $("#t-bench tbody");
  tb.innerHTML = rows.map((r) => `<tr><td>${r.setting}</td><td>${r.resolution}</td><td>${(+r.gaussians).toLocaleString()}</td><td>${(+r.fps).toFixed(1)}</td><td>${(+r.peak_vram_mb).toFixed(0)}</td><td class="dim">${String(r.notes).split(";").pop().trim()}</td></tr>`).join("");
  const tip = (r) => `${r.setting} ${r.resolution}px, ${(+r.gaussians).toLocaleString()} G\n${(+r.fps).toFixed(1)} FPS, ${(+r.peak_vram_mb).toFixed(0)} MB peak`;
  cBres.set(rows.filter((r) => r.setting === "resolution").map((r) => ({ label: `${r.resolution}²`, value: +r.fps, tip: tip(r) })));
  cBn.set(rows.filter((r) => r.setting === "gaussians").map((r) => ({ label: fmtN(+r.gaussians), value: +r.fps, tip: tip(r) })));
}
const fmtN = (n) => (n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? Math.round(n / 1e3) + "k" : String(n));

// ------------------------------------------------------------------ browser stress test
function setMult(k) {
  st.mult = k;
  document.querySelectorAll("#mults button").forEach((b) => b.classList.toggle("on", +b.dataset.k === k));
  if (st.base) viewer.setGaussians(replicate(st.base, k));
}
$("#mults").innerHTML = MULTS.map((k) => `<button data-k="${k}">×${k}</button>`).join("");
document.querySelectorAll("#mults button").forEach((b) => (b.onclick = () => setMult(+b.dataset.k)));

async function sweep() {
  if (!st.base) return;
  const tb = $("#t-web tbody");
  tb.innerHTML = "";
  for (const k of MULTS) {
    setMult(k);
    await sleep(1000); // first sort + warm-up
    const fps = [], gpu = [], sort = [];
    for (let i = 0; i < 5; i++) { await sleep(400); fps.push(stats.fps); if (stats.gpuMs != null) gpu.push(stats.gpuMs); sort.push(stats.sortMs); }
    const med = (a) => (a.length ? a.slice().sort((x, y) => x - y)[a.length >> 1] : null);
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>×${k}</td><td>${stats.count.toLocaleString()}</td><td>${stats.width}×${stats.height}</td><td>${med(fps)}</td><td>${med(gpu)?.toFixed(2) ?? "n/a"}</td><td>${med(sort).toFixed(0)}</td>`;
    tb.appendChild(tr);
  }
  setMult(1);
}
$("#sweep").onclick = sweep;

// ------------------------------------------------------------------ server jobs
async function generate() {
  const body = { prompt: $("#prompt").value.trim(), seed: +$("#seed").value, iters: +$("#iters").value, guidance_scale: +$("#cfg").value };
  try {
    const r = await fetchJSON("api/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    st.jobName = r.name; st.metrics = []; st.lastSnapStep = -1; drawMetrics();
    $("#log").textContent = "";
    poll();
  } catch (e) { alert(e.message); }
}
$("#go").onclick = generate;

async function bench() {
  if (!st.sel) return alert("select a run first");
  try {
    await fetchJSON("api/benchmark", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: st.sel.name, quick: false }) });
    showBench([]);
    poll();
  } catch (e) { alert(e.message); }
}
$("#bench").onclick = bench;

async function poll() {
  if (st.polling) return;
  st.polling = true;
  try {
    for (;;) {
      const j = await fetchJSON("api/job");
      if (!j.kind) break;
      $("#go").disabled = $("#bench").disabled = j.active;
      $("#log").textContent = j.log.join("\n");
      $("#log").scrollTop = 1e9;
      if (j.kind === "generate") updateGen(j);
      else if (j.rows) { showBench(j.rows); $("#jobstatus").textContent = j.active ? "benchmark" : "bench done"; }
      if (!j.active) {
        if (j.kind === "generate" && j.returncode === 0) { await loadRuns(); await selectRun(j.name); }
        if (j.kind === "benchmark" && j.returncode === 0) { await loadRuns(); st.sel = st.runs.find((r) => r.name === j.name); await loadBench(st.sel); }
        break;
      }
      await sleep(1000);
    }
  } finally { st.polling = false; }
}

async function updateGen(j) {
  const p = j.progress;
  $("#jobbox").hidden = false;
  $("#jobname").textContent = j.name;
  const status = p?.status || (j.active ? "starting" : j.returncode === 0 ? "done" : "failed");
  const b = $("#jobstatus");
  b.textContent = status.replace(/_/g, " "); b.className = "badge " + status;
  if (!p) return;
  $("#jobbar").style.width = `${(100 * p.step) / p.iters}%`;
  $("#k-step").textContent = `${p.step} / ${p.iters}`;
  $("#k-elapsed").textContent = `${j.elapsed_s.toFixed(0)} s`;
  $("#k-loss").textContent = p.sds_loss?.toFixed(2) ?? "–";
  $("#k-t").textContent = p.timestep ?? "–";
  $("#k-res").textContent = p.render_res ? `${p.render_res}²` : "–";
  $("#k-n").textContent = p.num_gaussians?.toLocaleString() ?? "–";
  $("#k-vram").textContent = `${(p.peak_vram_mb / 1024).toFixed(2)} GB`;
  $("#k-gxyz").textContent = p.grad_norm ? p.grad_norm.xyz.toExponential(2) : "–";
  const more = await fetchJSON(`api/metrics/${j.name}?since=${st.metrics.length}`).catch(() => []);
  if (more.length) { st.metrics.push(...more); drawMetrics(); }
  if (p.step !== st.lastSnapStep && (p.step % 25 === 0 || p.status !== "training")) {
    st.lastSnapStep = p.step;
    $("#snap").src = `${DATA}/${j.name}/progress/latest.png?v=${p.step}-${p.status}`;
    $("#scrubv").textContent = `live · step ${p.step}`;
  }
}

async function pollSystem() {
  for (;;) {
    try {
      const s = await fetchJSON("api/system");
      $("#gpu").textContent = `${s.host} · GPU ${s.gpu_index} ${s.name.replace("NVIDIA ", "")} · ${(s.mem_used_mb / 1024).toFixed(1)}/${(s.mem_total_mb / 1024).toFixed(0)} GB · ${s.util}% · ${s.power_w.toFixed(0)} W`;
    } catch { /* ignore */ }
    await sleep(2000);
  }
}

// ------------------------------------------------------------------ video-narration helpers
window.studio = {
  viewer, st, generate, bench, sweep, setMult, selectRun, loadRuns,
  caption(text, sub = "") { const c = $("#caption"); c.hidden = !text; c.innerHTML = text + (sub ? `<small>${sub}</small>` : ""); },
  flash(sel) { const e = $(sel); e.classList.add("flash"); setTimeout(() => e.classList.remove("flash"), 2500); },
};

// ------------------------------------------------------------------ boot
(async () => {
  try {
    await fetchJSON("api/system");
    st.live = true;
    $("#mode").textContent = "live GPU server"; $("#mode").className = "pill live";
    pollSystem();
  } catch {
    $("#mode").textContent = "static build (pre-generated)"; $("#mode").className = "pill static";
    $("#go").disabled = $("#bench").disabled = true; $("#offline").hidden = false;
    try {
      const sys = await fetchJSON("data/system.json");
      $("#gpu").textContent = `generated on ${sys.name}`;
    } catch { $("#gpu").textContent = "GPU: offline"; }
  }
  await loadRuns();
  if (st.runs.length) await selectRun(st.runs[0].name);
  else { drawMetrics(); showBench([]); }
  if (st.live) { const j = await fetchJSON("api/job"); if (j.active) poll(); }
})();
