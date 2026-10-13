import { SplatViewer, loadPly, replicate } from "./splat.js";
import { LineChart, BarChart } from "./charts.js";

const $ = (s) => document.querySelector(s);
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const DATA = "data/results";
const MULTS = [1, 2, 4, 8, 16];
const GRAD_KEYS = [["xyz", "position"], ["features_dc", "colour"], ["opacity", "opacity"], ["scaling", "scale"], ["rotation", "rotation"]];

const st = { galleryTick: 0, live: false, cfg: {}, runs: [], mine: [], sel: null, base: null, mult: 1, metrics: [], jobId: null, lastSnapStep: -1, polling: false };
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const runPath = (r) => r.rel || `${DATA}/${r.name}`;

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
  const gallery = st.live ? await fetchJSON("api/runs") : await fetchJSON("data/runs.json").catch(() => []);
  st.runs = [...st.mine, ...gallery.filter((g) => !st.mine.some((m) => m.name === g.name))];
  const el = $("#runs");
  if (!st.runs.length) { el.innerHTML = `<p class="dim">no runs yet</p>`; return; }
  el.innerHTML = "";
  for (const r of st.runs) {
    const b = document.createElement("button");
    b.className = "run" + (r.name === st.sel?.name ? " sel" : "");
    b.dataset.name = r.name;
    b.innerHTML = `<img alt="" src="${runPath(r)}/views/view_000.png"><div><b title="${esc(r.prompt)}">${r.mine ? "★ " : ""}${esc(r.prompt)}</b><i class="tag ${r.community ? "comm" : "ex"}">${r.mine ? "yours" : r.community ? "community" : "example"}</i>
      <span>seed ${esc(r.seed)} · ${esc(r.iters)} it · ${esc(r.train_s)}s · ${(r.num_gaussians_final / 1000).toFixed(1)}k G</span></div>`;
    b.onclick = () => selectRun(r.name);
    el.appendChild(b);
  }
}

async function selectRun(name) {
  const r = st.runs.find((x) => x.name === name);
  if (!r) return;
  st.sel = r;
  const P = runPath(r);
  document.querySelectorAll(".run").forEach((b) => b.classList.toggle("sel", b.dataset.name === name));
  $("#viewname").textContent = `· “${r.prompt}”`;
  $("#hud").textContent = "loading PLY…";
  st.base = await loadPly(`${P}/${r.ply}`);
  setMult(st.mult);
  // training curves
  const txt = await fetch(`${P}/metrics.jsonl`).then((x) => x.text()).catch(() => "");
  st.metrics = txt.split("\n").filter(Boolean).map((l) => JSON.parse(l));
  drawMetrics();
  // snapshot scrubber
  const steps = [0];
  for (let s = 25; s <= r.iters; s += 25) steps.push(s);
  const frames = [...steps.map((s) => `step_${String(s).padStart(4, "0")}`), "final"];
  $("#scrubwrap").hidden = false;
  const sc = $("#scrub");
  sc.max = frames.length - 1; sc.value = frames.length - 1;
  sc.oninput = () => { $("#snap").src = `${P}/progress/${frames[sc.value]}.png`; $("#scrubv").textContent = frames[sc.value].replace("_", " "); };
  sc.oninput();
  await loadBench(r);
}

async function loadBench(r) {
  let url = r?.has_benchmark ? `${runPath(r)}/benchmark.csv` : "data/evidence/performance.csv";
  const txt = await fetch(url).then((x) => (x.ok ? x.text() : "")).catch(() => "");
  const [head, ...lines] = txt.trim().split(/\r?\n/);
  if (!head) return showBench([]);
  const keys = head.split(",");
  showBench(lines.map((l) => { const v = l.split(","); return Object.fromEntries(keys.map((k, i) => [k, k === "notes" ? v.slice(i).join(",") : v[i]])); }));
}

function showBench(rows) {
  const tb = $("#t-bench tbody");
  tb.innerHTML = rows.map((r) => `<tr><td>${r.setting}</td><td>${r.resolution}</td><td>${(+r.gaussians).toLocaleString()}</td><td>${(+r.fps).toFixed(1)}</td><td>${(+r.peak_vram_mb).toFixed(0)}</td><td class="dim">${esc(String(r.notes ?? "").split(";").pop().trim())}</td></tr>`).join("");
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

// ------------------------------------------------------------------ server jobs (queued; one GPU job at a time)
// One job per browser session: a random id kept in sessionStorage (new tab/visit = new session).
function sessionId() {
  try {
    let s = sessionStorage.getItem("studio.session");
    if (!s) { s = Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, "0")).join(""); sessionStorage.setItem("studio.session", s); }
    return s;
  } catch { return (st._sid ||= Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, "0")).join("")); }
}
// A session may have one job queued or running; the button unlocks again when it finishes.
function lockGenerate() {
  if (st.polling) { $("#go").disabled = true; $("#go").textContent = "Your job is in progress…"; }
  else { $("#go").disabled = false; if (/in progress/.test($("#go").textContent)) $("#go").textContent = "Generate 3D on GPU"; }
}

async function generate() {
  const body = { prompt: $("#prompt").value.trim(), seed: +$("#seed").value, iters: +$("#iters").value, guidance_scale: +$("#cfg").value };
  try {
    const r = await postJSON("api/generate", body, { "X-Studio-Session": sessionId() });
    startJob(r.id);
  } catch (e) {
    showError(e);
  }
}
$("#go").onclick = generate;

async function bench() {
  if (!st.sel) return alert("select a run first");
  try {
    const r = await postJSON("api/benchmark", { name: st.sel.name, quick: false });
    showBench([]);
    startJob(r.id);
  } catch (e) { showError(e); }
}
$("#bench").onclick = bench;

async function postJSON(url, body, headers = {}) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json", ...headers }, body: JSON.stringify(body) });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.detail || `${r.status}`);
  return d;
}

function showError(e) {
  $("#jobbox").hidden = false;
  const b = $("#jobstatus");
  b.textContent = "not started"; b.className = "badge failed";
  $("#jobname").textContent = "";
  $("#k-step").textContent = "–";
  $("#joberr").hidden = false; $("#joberr").className = "err";
  $("#joberr").textContent = e.message;
}

function startJob(id) {
  st.jobId = id; st.metrics = []; st.lastSnapStep = -1; drawMetrics();
  $("#log").textContent = "";
  $("#joberr").hidden = true;
  try { sessionStorage.setItem("studio.job", id); } catch { /* storage unavailable */ }
  poll();
}

async function poll() {
  if (st.polling || !st.jobId) return;
  st.polling = true;
  try {
    for (;;) {
      let j;
      try { j = await fetchJSON(`api/jobs/${st.jobId}`); }
      catch { try { sessionStorage.removeItem("studio.job"); } catch {} break; }
      const active = j.status === "queued" || j.status === "running";
      lockGenerate();
      $("#bench").disabled = active || !st.cfg.admin;
      $("#log").textContent = j.log.join("\n");
      $("#log").scrollTop = 1e9;
      if (j.kind === "generate") await updateGen(j);
      else if (j.rows) { showBench(j.rows); $("#jobbox").hidden = false; $("#jobstatus").textContent = active ? "benchmark" : "bench done"; }
      if (!active) {
        try { sessionStorage.removeItem("studio.job"); } catch {}
        if (j.kind === "generate" && j.status === "done" && j.run) {
          if (!st.mine.some((r) => r.name === j.run.name)) st.mine.unshift({ ...j.run, mine: true });
          await loadRuns(); await selectRun(j.run.name);
        }
        if (j.kind === "benchmark" && j.status === "done") { await loadRuns(); st.sel = st.runs.find((r) => r.name === j.name); await loadBench(st.sel); }
        break;
      }
      await sleep(1000);
    }
  } finally { st.polling = false; lockGenerate(); }
}

async function updateGen(j) {
  const p = j.progress;
  $("#jobbox").hidden = false;
  $("#jobname").textContent = j.prompt ? `“${j.prompt}”` : j.name;
  let status = j.status === "queued" ? "queued" : p?.status || (j.status === "running" ? "starting" : j.status);
  if (j.status === "failed") status = "failed";
  const b = $("#jobstatus");
  b.textContent = status.replace(/_/g, " "); b.className = "badge " + status;
  if (j.status === "queued") {
    $("#jobbar").style.width = "0%";
    $("#k-step").textContent = `#${j.position} in queue`;
    $("#k-elapsed").textContent = `waiting ${j.wait_s.toFixed(0)} s`;
    $("#joberr").hidden = false;
    $("#joberr").className = "info";
    $("#joberr").textContent = j.waiting_for_gpu
      ? `The GPU is busy with another visitor's job. You are #${j.position} in line (about ${j.position} min) - your job starts automatically, keep this page open.`
      : `Starting on a free GPU…`;
    return;
  }
  $("#joberr").className = "err";
  if (j.status === "running") $("#joberr").hidden = true;
  $("#jobname").textContent = (j.prompt ? `“${j.prompt}”` : j.name) + (j.gpu != null ? ` · GPU ${j.gpu}` : "");
  if (j.status === "failed") { $("#joberr").hidden = false; $("#joberr").textContent = "The job failed - see the log below."; }
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
  const more = await fetchJSON(`api/jobs/${j.id}/metrics?since=${st.metrics.length}`).catch(() => []);
  if (more.length) { st.metrics.push(...more); drawMetrics(); }
  if (p.step !== st.lastSnapStep && (p.step % 25 === 0 || p.status !== "training")) {
    st.lastSnapStep = p.step;
    $("#snap").src = `${j.rel}/progress/latest.png?v=${p.step}-${p.status}`;
    $("#scrubv").textContent = `live · step ${p.step}`;
  }
}

// Before submitting: tell the visitor whether their job would start now or wait behind others.
function showQueueHint(q) {
  const h = $("#queuehint"), go = $("#go");
  if (st.polling) { h.hidden = true; return; }  // the visitor's own job box shows their position instead
  const free = q.capacity - q.running;  // job slots open right now = process-free GPUs
  h.hidden = false;
  if (free > 0 && q.queued === 0) {
    h.className = "qhint free";
    h.textContent = `● ${free} GPU${free > 1 ? "s" : ""} free - your job starts right away (about 1 minute).`;
    go.textContent = "Generate 3D on GPU";
  } else {
    const pos = q.queued + 1;
    const rounds = Math.ceil(pos / Math.max(1, q.capacity));
    h.className = "qhint busy";
    h.textContent = q.capacity
      ? `● All ${q.capacity} available GPU${q.capacity > 1 ? "s are" : " is"} busy, ${q.queued} waiting - yours would be #${pos} in line (about ${Math.max(1, Math.round(rounds * 1.2))} min). It starts automatically when a GPU frees up.`
      : `● No GPU is free on the server right now - yours would be #${pos} in line and starts as soon as a GPU frees up.`;
    go.textContent = `Join queue (#${pos})`;
  }
}

async function pollQueue() {
  for (;;) {
    try {
      const q = await fetchJSON("api/queue");
      $("#queue").textContent = `max jobs now: ${q.capacity} · running ${q.running} · free GPUs ${q.gpus_idle} · waiting ${q.queued}`;
      showQueueHint(q);
    } catch { /* ignore */ }
    await sleep(3000);
    if (++st.galleryTick % 10 === 0 && !st.polling) loadRuns().catch(() => {});  // pick up new community models (~30 s)
  }
}

async function pollSystem() {
  for (;;) {
    try {
      const s = await fetchJSON("api/system");
      $("#gpu").textContent = `${s.host} · ${s.gpus_total}× ${s.name.replace("NVIDIA ", "")}`;
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
    st.cfg = await fetchJSON("api/config");
    st.live = true;
    $("#mode").textContent = st.cfg.public ? "live GPU server · public" : "live GPU server"; $("#mode").className = "pill live";
    $("#queue").hidden = false;
    $("#prompt").maxLength = st.cfg.prompt_max;
    if (!st.cfg.admin) {
      $("#iters").max = st.cfg.iters[1]; $("#iters").min = st.cfg.iters[0];
      $("#bench").disabled = true; $("#bench").title = "run by the owner; measured results are shown below";
      $("#publicnote").hidden = false;
      const slots = st.cfg.max_parallel ? `up to ${st.cfg.max_parallel} job${st.cfg.max_parallel > 1 ? "s" : ""} at a time` : "one job on every GPU that nobody else is using (the count above updates live)";
      const quota = st.cfg.per_visitor ? ` Each visitor can generate ${st.cfg.per_visitor} model per ${st.cfg.window_h} h.` : "";
      $("#publicnote").textContent = `Public demo: the server runs ${slots} (about 1 minute each on an A100); other prompts wait in a queue and start automatically.${quota} Each session can have one job in progress at a time, max ${st.cfg.iters[1]} SDS steps, no mesh export. Every finished model is added to the public gallery below.`;
    }
    pollSystem();
    pollQueue();
    lockGenerate();
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
  if (st.live) {
    let id = null;
    try { id = sessionStorage.getItem("studio.job"); } catch { /* storage unavailable */ }
    if (id) { st.jobId = id; poll(); }
  }
})();
