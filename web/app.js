import { SplatViewer, loadPly, replicate, compose } from "./splat.js";
import { LineChart, BarChart } from "./charts.js";

const $ = (s) => document.querySelector(s);
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const DATA = "data/results";
const MULTS = [1, 2, 4, 8, 16];
const GRAD_KEYS = [["xyz", "position"], ["features_dc", "colour"], ["opacity", "opacity"], ["scaling", "scale"], ["rotation", "rotation"]];

const st = { mode: "create", target: null, lastTargetId: null, galleryTick: 0, live: false, cfg: {}, runs: [], mine: [], sel: null, base: null, mult: 1, metrics: [], jobId: null, lastSnapStep: -1, polling: false };
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
      `\nsort ${s.sortMs.toFixed(0)} ms  ·  ${String(s.renderer).replace(/^ANGLE \((.*)\)$/, "$1").slice(0, 46)}` +
      (st.exhibit ? `\nexhibition: ${st.exhibit.k} models · LOD kept ${fmtN(st.exhibit.kept)} of ${fmtN(st.exhibit.of)}` : "");
  },
});
viewer.autoRotate = 20;
$("#rotate").onchange = (e) => (viewer.autoRotate = e.target.checked ? 20 : 0);
$("#scalemod").oninput = (e) => { viewer.scaleMod = +e.target.value; $("#scalev").textContent = (+e.target.value).toFixed(2); };
// Rendering pillar (Beta): studio lighting from per-Gaussian normals, a ground plane with soft shadow, backdrops
const BACKDROPS = {
  dark: { stage: true, theme: { top: [0.17, 0.19, 0.24], bottom: [0.055, 0.062, 0.078], floor: [0.36, 0.39, 0.46] } },
  light: { stage: true, theme: { top: [0.97, 0.97, 0.98], bottom: [0.80, 0.82, 0.86], floor: [0.93, 0.93, 0.95] } },
  white: { stage: false, bg: [1, 1, 1] },
  black: { stage: false, bg: [0.07, 0.08, 0.1] },
};
function setBackdrop(name) {
  const b = BACKDROPS[name] || BACKDROPS.dark;
  $("#backdrop").value = name;
  viewer.stage = b.stage;
  if (b.theme) viewer.theme = b.theme;
  if (b.bg) viewer.bg = b.bg;
}
function setStudioLook(on) { $("#studioLook").checked = on; viewer.lit = on; }
$("#studioLook").onchange = (e) => (viewer.lit = e.target.checked);
$("#backdrop").onchange = (e) => setBackdrop(e.target.value);
$("#lightAz").oninput = $("#lightEl").oninput = () => viewer.setLight(+$("#lightAz").value, +$("#lightEl").value);
$("#reset").onclick = () => Object.assign(viewer.cam, { az: st.sel?.front_azimuth || 0, el: 15, r: st.exhibit ? exhibitRadius(st.exhibit.k) : 2.5 });
$("#resscale").onchange = (e) => viewer.setResolutionScale(+e.target.value);
$("#rmode").onchange = (e) => (viewer.mode = +e.target.value);
$("#shot").onclick = () => {
  const a = document.createElement("a");
  a.href = viewer.screenshot();
  a.download = `${(st.sel?.prompt || "studio").replace(/[^a-z0-9]+/gi, "_").slice(0, 40)}.png`;
  a.click();
};

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
      <span>${esc((r.guidance || "sds").toUpperCase())} · seed ${esc(r.seed)} · ${esc(r.iters)} it · ${esc(r.train_s)}s · ${(r.num_gaussians_final / 1000).toFixed(1)}k G</span></div>`;
    b.onclick = () => selectRun(r.name);
    el.appendChild(b);
  }
}

async function selectRun(name) {
  const r = st.runs.find((x) => x.name === name);
  if (!r) return;
  st.sel = r;
  const P = runPath(r);
  leaveExhibit();
  document.querySelectorAll(".run").forEach((b) => b.classList.toggle("sel", b.dataset.name === name));
  $("#viewname").textContent = `· “${r.prompt}”`;
  $("#hud").textContent = "loading model…";
  try {
    st.base = await loadPly(`${P}/${r.ply}`);
  } catch (e) {
    $("#hud").textContent = "could not load this model (it may have been removed)";
    return;
  }
  $("#dlply").href = `${P}/${r.ply}`; $("#dlply").hidden = false;
  viewer.cam.az = r.front_azimuth || 0;  // MVDream objects face azimuth 90
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
  const cfg = $("#cfg").value.trim();
  const body = { prompt: $("#prompt").value.trim(), seed: +$("#seed").value, iters: +$("#iters").value,
                 guidance: $("#guidance").value, guidance_scale: cfg === "" ? null : +cfg };
  if (!body.prompt) return showError(new Error("Type a prompt first."));
  if (st.mode === "challenge") {
    if (tabooHits(body.prompt).length) return checkTaboo();
    body.target = st.target.id;
  }
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

$("#stop").onclick = async () => {
  if (!st.jobId || !confirm("Stop this job and release the GPU?")) return;
  $("#stop").disabled = true;
  try { await postJSON(`api/jobs/${st.jobId}/cancel`, {}, { "X-Studio-Session": sessionId() }); }
  catch (e) { alert(e.message); }
  finally { $("#stop").disabled = false; }
};

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
  if (st.round) { finishRound(st.round.auto, "auto"); $("#result").hidden = true; }
  st.jobId = id; st.metrics = []; st.lastSnapStep = -1; drawMetrics();
  // reset the job box so values from the previous job are never shown for the new one
  for (const k of ["eta", "step", "elapsed", "loss", "t", "res", "n", "vram", "gxyz"]) $(`#k-${k}`).textContent = "–";
  $("#jobbar").style.width = "0%";
  $("#jobstatus").textContent = "submitted"; $("#jobstatus").className = "badge starting";
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
      catch (e) {
        if (/: 404/.test(e.message)) {  // the server restarted and forgot this job: tell the user instead of spinning forever
          try { sessionStorage.removeItem("studio.job"); } catch {}
          showError(new Error("The server restarted and this job was lost. Please start it again."));
          break;
        }
        await sleep(3000);  // transient network error: keep trying
        continue;
      }
      const active = j.status === "queued" || j.status === "running";
      if (!active) $("#stop").hidden = true;
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
        if (j.kind === "generate") showResult(j);
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
  $("#stop").hidden = !(j.status === "queued" || j.status === "running");
  $("#k-eta").textContent = j.eta_finish_s != null ? `≈ ${clock(j.eta_finish_s)} (${mins(j.eta_finish_s)})` : (j.status === "done" ? "finished" : "–");
  if (j.status === "cancelled") {
    $("#joberr").hidden = false; $("#joberr").className = "info";
    $("#joberr").textContent = "Stopped - the GPU was released. You can start a new job.";
    return;
  }
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
  if (j.status === "failed") { $("#joberr").hidden = false; $("#joberr").textContent = j.error || "The job failed - see the log below."; }
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

// ETA = measured seconds/step x steps + measured overhead for the chosen method (server-side model)
function runSeconds(method, iters) {
  const m = st.cfg.eta_model?.[method] || [0.1, 25];
  return m[1] + m[0] * iters;
}
const clock = (sec) => new Date(Date.now() + sec * 1000).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
const mins = (sec) => (sec < 90 ? `${Math.max(1, Math.round(sec))} s` : `${(sec / 60).toFixed(1)} min`);

// Before submitting: tell the visitor whether their job would start now or wait behind others, and when it would finish.
function showQueueHint(q = st.lastQ) {
  if (!q) return;
  st.lastQ = q;
  const h = $("#queuehint"), go = $("#go");
  if (st.polling) { h.hidden = true; return; }  // the visitor's own job box shows their position instead
  const free = q.capacity - q.running;  // job slots open right now = process-free GPUs
  const method = $("#guidance").value, iters = Math.max(50, Math.round((+$("#iters").value || 500) / 50) * 50);
  const run = runSeconds(method, iters), wait = q.new_job_start_s;
  const eta = wait == null ? "" : ` ETA: finishes ≈ ${clock(wait + run)} (in ~${mins(wait + run)}; ${method.toUpperCase()}, ${iters} steps).`;
  h.hidden = false;
  if (free > 0 && q.queued === 0) {
    h.className = "qhint free";
    h.textContent = `● ${free} GPU${free > 1 ? "s" : ""} free - your job starts right away.${eta}`;
    go.textContent = "Generate 3D on GPU";
  } else {
    const pos = q.queued + 1;
    const rounds = Math.ceil(pos / Math.max(1, q.capacity));
    h.className = "qhint busy";
    h.textContent = q.capacity
      ? `● All ${q.capacity} available GPU${q.capacity > 1 ? "s are" : " is"} busy, ${q.queued} waiting - yours would be #${pos} in line, starting in ~${mins(wait ?? rounds * 70)}.${eta}`
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
      if (!$("#menu").hidden) showMenu();
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

// ------------------------------------------------------------------ guidance comparison (evidence CSVs)
const VERDICT = { sds: "blotchy, Janus corgi", csd: "noisy", nfsd: "natural colours, blurry", ism: "no Janus, off colours", mvdream: "best: coherent, sharp" };
async function csvRows(url) {
  const t = await fetch(url).then((r) => (r.ok ? r.text() : "")).catch(() => "");
  const [head, ...lines] = t.trim().split(/\r?\n/);
  if (!head) return [];
  const k = head.split(",");
  return lines.map((l) => Object.fromEntries(l.split(",").map((v, i) => [k[i], v])));
}
const avg = (rows, key) => rows.reduce((a, r) => a + +r[key], 0) / rows.length;
async function loadComparison() {
  const rows = await csvRows("data/evidence/guidance_comparison.csv");
  if (!rows.length) return;
  $("#t-algo tbody").innerHTML = ["sds", "csd", "nfsd", "ism", "mvdream"].map((m) => {
    const r = rows.filter((x) => x.method === m);
    if (!r.length) return "";
    return `<tr class="${m === "mvdream" ? "best" : ""}"><td>${m.toUpperCase()}</td><td>${avg(r, "clip_score").toFixed(2)}</td><td>${avg(r, "train_s").toFixed(0)} s</td><td>${avg(r, "total_s").toFixed(0)} s</td><td>${(Math.max(...r.map((x) => +x.peak_vram_mb)) / 1024).toFixed(1)} GB</td><td>${VERDICT[m]}</td></tr>`;
  }).join("");
  const steps = [[300, "_it300", "clean, softer"], [500, "", "sharpest (default, cap)"], [800, "_it800", "over-saturated"], [1200, "_it1200", "blown out"]];
  const out = [];
  for (const [n, suf, v] of steps) {
    const r = suf ? await csvRows(`data/evidence/guidance_comparison${suf}.csv`) : rows.filter((x) => x.method === "mvdream");
    if (r.length) out.push(`<tr class="${n === 500 ? "best" : ""}"><td>${n}</td><td>${avg(r, "clip_score").toFixed(2)}</td><td>${avg(r, "total_s").toFixed(0)} s</td><td>${v}</td></tr>`);
  }
  $("#t-steps tbody").innerHTML = out.join("");
}

// ------------------------------------------------------------------ game loop: START (menu) -> ACTIVE (studio) -> END (result)
const loadStats = () => { const z = { rounds: 0, wins: 0, streak: 0, points: 0 }; try { return { ...z, ...JSON.parse(sessionStorage.getItem("studio.stats")) }; } catch { return z; } };
function saveStats(s) { try { sessionStorage.setItem("studio.stats", JSON.stringify(s)); } catch {} showScore(); }
function showScore() {
  const s = loadStats();
  $("#score").hidden = !s.rounds;
  $("#score").textContent = `🎯 ${s.points} pts · ${s.wins}/${s.rounds} wins · streak ${s.streak}`;
  $("#ch-round").textContent = `round ${s.rounds + 1} · ${s.wins} win${s.wins === 1 ? "" : "s"}`;
}

function showMenu() {
  if (st.round) finishRound(st.round.auto, "auto");   // leaving mid-round keeps the auto-camera verdict
  $("#result").hidden = true;
  $("#menu").hidden = false;
  const q = st.lastQ;
  const free = q ? q.capacity - q.running : 0;
  $("#menuStatus").textContent = !st.live ? "" : !q ? "● connecting to the GPU server…"
    : free > 0 ? `● ${free} GPU${free === 1 ? "" : "s"} free · ${q.queued} waiting · about ${mins(runSeconds(st.cfg.default_guidance || "sds", 500))} per model`
    : `● All GPUs are busy right now (${q.running} studio job${q.running === 1 ? "" : "s"}, ${q.queued} waiting). You can still play: your job waits in the queue and starts on the next free GPU.`;
}

function enterMode(m) {
  $("#menu").hidden = true;
  if (m === "gallery") { m = "create"; setTimeout(() => { $("#runs").scrollIntoView({ behavior: "smooth", block: "center" }); studio.flash("#runs"); }, 50); }
  if (m === "how") { setTimeout(() => $(".algo").scrollIntoView({ behavior: "smooth" }), 50); return; }
  if (m === "exhibit") { showExhibit(); setTimeout(() => $(".viewer").scrollIntoView({ behavior: "smooth", block: "start" }), 50); return; }
  st.mode = m;
  $("#modechip").textContent = m === "challenge" ? "🎯 Challenge" : "▶ Create";
  $("#challengeBox").hidden = m !== "challenge";
  $("#genTitle").textContent = m === "challenge" ? "Challenge: describe it, don't name it" : "Generate (text → 3D)";
  if (m === "challenge") { if (!st.target) newTarget(); $("#prompt").value = ""; $("#prompt").placeholder = "Describe the object without saying its name…"; }
  else { $("#prompt").placeholder = "e.g. a ceramic teapot shaped like a pumpkin"; }
  checkTaboo();
  showScore();
  if (m !== "gallery") $("#prompt").focus();
}

function newTarget() {
  const list = st.cfg.challenge || [];
  if (!list.length) return;
  let t;
  do { t = list[Math.floor(Math.random() * list.length)]; } while (list.length > 1 && t.id === st.lastTargetId);
  setTarget(t);
  $("#prompt").value = "";
  checkTaboo();
}
function setTarget(t) {
  st.target = t; st.lastTargetId = t.id;
  $("#ch-label").textContent = t.label;
  $("#ch-taboo").innerHTML = t.taboo.map((w) => `<i>${esc(w)}</i>`).join("");
}
$("#ch-new").onclick = newTarget;

const tabooHits = (text) => (st.mode === "challenge" && st.target
  ? st.target.taboo.filter((w) => new RegExp(`\\b${w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(s|es)?\\b`, "i").test(text)) : []);
function checkTaboo() {
  const hits = tabooHits($("#prompt").value);
  $("#tabooWarn").hidden = !hits.length;
  $("#tabooWarn").textContent = hits.length ? `Challenge rule: you can't say ${hits.map((h) => `“${h}”`).join(", ")}.` : "";
  if (!st.polling) $("#go").disabled = !!hits.length || !st.live;
  return hits;
}
$("#prompt").addEventListener("input", checkTaboo);

function judgeHTML(jd, targetLabel, what = "8 auto-camera views") {
  if (!jd?.top) return `<p class="dim small">The CLIP judge was not available for this run.</p>`;
  const rows = jd.top.map((t) => `<div class="jrow ${t.label === targetLabel ? "hit" : ""}"><span>${esc(t.label.replace(/^an? /, ""))}</span><span class="jb"><i style="width:${Math.max(2, t.p * 100)}%"></i></span><span class="mono">${Math.round(t.p * 100)}%</span></div>`).join("");
  return `<h4>The judge's guesses (CLIP, ${what})</h4>${rows}` +
    (jd.prompt_clip != null ? `<p class="dim small">Prompt match (CLIP): <b>${jd.prompt_clip}</b> <span class="dim">(good results score about 25–30)</span></p>` : "");
}

function resultButtons(phase, ch, done) {   // phase: "preview" (photo shoot pending) or "final"
  $("#r-shoot").hidden = $("#r-keep").hidden = phase !== "preview";
  $("#r-explore").hidden = phase === "preview" || !done;
  $("#r-again").hidden = phase === "preview";
  $("#r-retry").hidden = phase === "preview" || !ch;
  $("#r-ply").hidden = !done;
  $("#r-again").textContent = ch ? "Play again (new target)" : "Create another";
}

function runStatsHTML(run) {
  return `<dt>method</dt><dd>${esc((run.guidance || "sds").toUpperCase())}</dd><dt>steps</dt><dd>${run.iters}</dd>` +
    `<dt>optimisation</dt><dd>${run.train_s} s</dd><dt>total</dt><dd>${run.total_s} s</dd><dt>Gaussians</dt><dd>${(run.num_gaussians_final || 0).toLocaleString()}</dd>`;
}

function showResult(j) {
  const run = j.run, jd = run?.judge, ch = !!j.target;
  const tgt = ch ? (st.cfg.challenge || []).find((t) => t.id === j.target) : null;
  const banner = $("#r-banner"), sub = $("#r-sub");
  $("#r-photos").hidden = true;
  if (j.status !== "done") {
    resultButtons("final", ch, false);
    banner.className = "banner fail";
    banner.textContent = j.status === "cancelled" ? "⏹ Stopped" : "⚠ Generation failed";
    sub.textContent = j.status === "cancelled" ? "The job was stopped and its GPU released." : (j.error || "Something went wrong on the GPU server. Please try again.");
    $("#r-img").hidden = true; $("#r-judge").innerHTML = ""; $("#r-stats").innerHTML = "";
    if (ch) { const s = loadStats(); s.rounds += 1; s.streak = 0; saveStats(s); }
  } else {
    $("#r-img").hidden = false;
    $("#r-img").src = `${run.rel}/progress/final.png`;
    $("#r-ply").href = `${run.rel}/${run.ply}`;
    $("#r-stats").innerHTML = runStatsHTML(run);
    if (ch) {
      // ACTIVE state continues: the auto-camera verdict is only a preview; the photo shoot decides the round
      st.round = { j, run, auto: jd, tgt, photos: [] };
      if (tgt && st.target?.id !== tgt.id) setTarget(tgt);   // the challenge box shows the target being judged
      resultButtons("preview", true, true);
      banner.className = "banner ready";
      banner.textContent = `Model ready. Preview: the judge sees ${jd?.top?.[0]?.label ?? "…"}`;
      sub.innerHTML = `Target: <b>${esc(tgt?.label)}</b> · your prompt: “${esc(j.prompt)}”. ` +
        `This guess comes from 8 automatic camera views. <b>The final verdict comes from your own photos</b>: light and frame the model to make the judge see ${esc(tgt?.label)}.`;
      $("#r-judge").innerHTML = judgeHTML(jd, tgt?.label);
    } else {
      resultButtons("final", false, true);
      banner.className = "banner ready";
      banner.textContent = "✓ Your 3D model is ready";
      sub.textContent = `“${j.prompt}” · it is now in the gallery and in your exhibition.`;
      $("#r-judge").innerHTML = judgeHTML(jd, null);
    }
  }
  $("#result").hidden = false;
}

// ------------------------------------------------------------------ photo shoot (Rendering pillar inside the Challenge loop)
function startShoot() {
  const r = st.round;
  if (!r) return;
  $("#result").hidden = true;
  if (st.sel?.name !== r.run.name) selectRun(r.run.name);
  viewer.autoRotate = 0; $("#rotate").checked = false;
  setStudioLook(true);
  $("#shootTarget").textContent = `· make the judge see ${r.tgt?.label}`;
  $("#shootMsg").textContent = "";
  $("#shoot").hidden = false;
  drawThumbs();
  $(".viewer").scrollIntoView({ behavior: "smooth", block: "start" });
}

// what CLIP sees: a centred square crop of the canvas (its processor centre-crops anyway)
function capturePhoto(size = 448) {
  const c = $("#gl"), s = Math.min(c.width, c.height);
  const o = document.createElement("canvas");
  o.width = o.height = size;
  o.getContext("2d").drawImage(c, (c.width - s) / 2, (c.height - s) / 2, s, s, 0, 0, size, size);
  return o.toDataURL("image/jpeg", 0.9);
}

function drawThumbs() {
  const r = st.round, el = $("#thumbs");
  el.innerHTML = "";
  for (let i = 0; i < 3; i++) {
    if (r?.photos[i]) {
      const im = document.createElement("img");
      im.src = r.photos[i]; im.title = "click to delete this photo";
      im.onclick = () => { r.photos.splice(i, 1); drawThumbs(); };
      el.appendChild(im);
    } else el.insertAdjacentHTML("beforeend", `<div class="slot"></div>`);
  }
  $("#snapBtn").disabled = !r || r.photos.length >= 3;
  $("#submitShots").disabled = !r || !r.photos.length;
  $("#submitShots").textContent = r?.photos.length ? `Send ${r.photos.length} photo${r.photos.length > 1 ? "s" : ""} to the judge` : "Send photos to the judge";
}

$("#snapBtn").onclick = () => {
  const r = st.round;
  if (!r || r.photos.length >= 3) return;
  r.photos.push(capturePhoto());
  $("#gl").animate([{ filter: "brightness(2.2)" }, { filter: "none" }], { duration: 220 });
  drawThumbs();
};
$("#submitShots").onclick = async () => {
  const r = st.round;
  if (!r?.photos.length) return;
  $("#submitShots").disabled = $("#snapBtn").disabled = $("#skipShots").disabled = true;
  $("#shootMsg").textContent = "The judge is looking at your photos…";
  try {
    const jd = await postJSON("api/judge_photos", { target: r.tgt.id, prompt: r.j.prompt, images: r.photos }, { "X-Studio-Session": sessionId() });
    finishRound(jd, "photos");
  } catch (e) {
    $("#shootMsg").textContent = `Could not judge the photos: ${e.message}. Try again or keep the auto verdict.`;
    drawThumbs();
  } finally { $("#skipShots").disabled = false; }
};
$("#skipShots").onclick = () => finishRound(st.round?.auto, "auto");

// END state of a Challenge round
function finishRound(jd, source) {
  const r = st.round;
  if (!r) return;
  st.round = null;
  $("#shoot").hidden = true;
  const win = !!jd?.win, s = loadStats();
  const pts = win ? Math.round(100 * (jd.target_p || 0)) : 0;
  s.rounds += 1; s.wins += win ? 1 : 0; s.streak = win ? s.streak + 1 : 0; s.points += pts; saveStats(s);
  if (win) addTrophy(r.run);
  const banner = $("#r-banner");
  banner.className = "banner " + (win ? "win" : "loss");
  banner.textContent = win ? `🏆 You win! +${pts} pts. The judge saw ${r.tgt?.label}.` : `✗ Not quite. The judge saw ${jd?.top?.[0]?.label ?? "something else"}.`;
  const autoP = r.auto?.target_p, p = jd?.target_p;
  let cmp = "";
  if (source === "photos" && autoP != null && p != null) {
    const d = Math.round(100 * (p - autoP));
    cmp = ` Auto camera ${Math.round(100 * autoP)}% → your photos ${Math.round(100 * p)}% <span class="delta ${d >= 0 ? "up" : "down"}">(${d >= 0 ? "+" : ""}${d} pts of confidence)</span>.`;
  }
  $("#r-sub").innerHTML = `Target: <b>${esc(r.tgt?.label)}</b> · your prompt: “${esc(r.j.prompt)}” · ` +
    (win ? `rank 1 with ${Math.round(100 * p)}%.` : `your target ranked #${jd?.target_rank ?? "?"}.`) + cmp;
  if (source === "photos") {
    $("#r-img").hidden = true;
    $("#r-photos").hidden = false;
    $("#r-photos").innerHTML = r.photos.map((u) => `<img src="${u}" alt="your photo">`).join("");
  }
  $("#r-judge").innerHTML = judgeHTML(jd, r.tgt?.label, source === "photos" ? `your ${r.photos.length} photo${r.photos.length > 1 ? "s" : ""}` : "8 auto-camera views");
  resultButtons("final", true, true);
  $("#result").hidden = false;
}

// ------------------------------------------------------------------ exhibition: trophies + your models on one lit stage
const loadTrophies = () => { try { return JSON.parse(sessionStorage.getItem("studio.trophies")) || []; } catch { return []; } };
function addTrophy(run) {
  const t = loadTrophies().filter((n) => n !== run.name);
  t.unshift(run.name);
  try { sessionStorage.setItem("studio.trophies", JSON.stringify(t.slice(0, 16))); } catch {}
}
const plyCache = new Map();
async function plyOf(r) {
  const url = `${runPath(r)}/${r.ply}`;
  if (!plyCache.has(url)) plyCache.set(url, loadPly(url));
  return plyCache.get(url);
}
// fit the row of models in the horizontal field of view (vertical fov 49.1°, 4:3 canvas)
const exhibitRadius = (k, hw) => (hw ?? (((k <= 4 ? k : Math.ceil(k / 2)) - 1) / 2) * 1.45 + 0.8) / Math.tan((Math.atan(Math.tan((49.1 * Math.PI) / 360) * 4 / 3))) + 0.9;

async function showExhibit() {
  const k = +$("#exhibitN").value, budget = +$("#exhibitBudget").value;
  const trophies = loadTrophies();
  const pick = [], seen = new Set();
  const add = (r, why) => { if (r && !seen.has(r.name) && pick.length < k) { seen.add(r.name); pick.push({ r, why }); } };
  trophies.forEach((n) => add(st.runs.find((r) => r.name === n), "trophy"));
  st.mine.forEach((m) => add(st.runs.find((r) => r.name === m.name), "yours"));
  st.runs.filter((r) => !r.community && r.guidance === "mvdream").forEach((r) => add(r, "example"));
  st.runs.filter((r) => !r.community).forEach((r) => add(r, "example"));
  st.runs.forEach((r) => add(r, "community"));
  if (!pick.length) return;
  $("#hud").textContent = `loading ${pick.length} models…`;
  const gs = await Promise.all(pick.map((p) => plyOf(p.r).catch(() => null)));
  const items = pick.map((p, i) => ({ ...p, g: gs[i], yaw: -(p.r.front_azimuth || 0) })).filter((x) => x.g);
  const scene = compose(items, budget);
  st.exhibit = { k: items.length, kept: scene.count, of: items.reduce((a, x) => a + x.g.count, 0) };
  st.base = scene; st.sel = null; st.mult = 1;
  document.querySelectorAll("#mults button").forEach((b) => b.classList.toggle("on", +b.dataset.k === 1));
  viewer.setGaussians(scene);
  viewer.target = scene.center;
  Object.assign(viewer.cam, { az: 0, el: items.length > 4 ? 32 : 14, r: exhibitRadius(items.length, scene.halfWidth) + (items.length > 4 ? 1.3 : 0) });
  viewer.autoRotate = 0; $("#rotate").checked = false;
  setStudioLook(true);
  document.querySelectorAll(".run").forEach((b) => b.classList.remove("sel"));
  $("#viewname").textContent = `· Exhibition: ${items.map((x) => (x.why === "trophy" ? "🏆 " : x.why === "yours" ? "★ " : "") + x.r.prompt.replace(/^an? /, "")).join("  |  ")}`;
  $("#exhibitCtl").hidden = false;
  $("#dlply").hidden = true;
}
function leaveExhibit() {
  if (!st.exhibit) return;
  st.exhibit = null; viewer.target = [0, 0, 0]; viewer.cam.r = 2.5;
  $("#exhibitCtl").hidden = true;
}
$("#exhibitN").onchange = $("#exhibitBudget").onchange = () => st.exhibit && showExhibit();

$("#r-explore").onclick = () => { $("#result").hidden = true; setStudioLook(true); };
$("#r-shoot").onclick = startShoot;
$("#r-keep").onclick = () => finishRound(st.round?.auto, "auto");
$("#r-again").onclick = () => { $("#result").hidden = true; if (st.mode === "challenge") newTarget(); else $("#prompt").select(); $("#prompt").focus(); };
$("#r-retry").onclick = () => { $("#result").hidden = true; $("#prompt").focus(); };
$("#r-menu").onclick = showMenu;
$("#menuBtn").onclick = showMenu;
document.querySelectorAll("#menu .mode").forEach((b) => (b.onclick = () => enterMode(b.dataset.go)));
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (!$("#result").hidden) { $("#result").hidden = true; if (st.round) startShoot(); }
  else if (!$("#menu").hidden) $("#menu").hidden = true;
  else showMenu();
});

// ------------------------------------------------------------------ video-narration helpers
window.studio = {
  viewer, st, generate, bench, sweep, setMult, selectRun, loadRuns, showMenu, enterMode, newTarget, showResult,
  startShoot, capturePhoto, finishRound, showExhibit, setBackdrop, setStudioLook,
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
    if (st.cfg.default_guidance) $("#guidance").value = st.cfg.default_guidance;
    const cfgHint = () => ($("#cfg").placeholder = `auto (${st.cfg.cfg_default?.[$("#guidance").value] ?? ""})`);
    $("#guidance").onchange = () => { cfgHint(); showQueueHint(); };
    $("#iters").oninput = () => showQueueHint();
    cfgHint();
    if (!st.cfg.admin) {
      $("#iters").max = st.cfg.iters[1]; $("#iters").min = st.cfg.iters[0];
      $("#bench").disabled = true; $("#bench").title = "run by the owner; measured results are shown below";
      $("#publicnote").hidden = false;
      const slots = st.cfg.max_parallel ? `up to ${st.cfg.max_parallel} job${st.cfg.max_parallel > 1 ? "s" : ""} at a time` : "one job on every GPU that nobody else is using (the count above updates live)";
      const quota = st.cfg.per_visitor ? ` Each visitor can generate ${st.cfg.per_visitor} model per ${st.cfg.window_h} h.` : "";
      $("#publicnote").textContent = `Public demo: the server runs ${slots} (about ${mins(runSeconds(st.cfg.default_guidance || "sds", 500))} per model on an A100); other prompts wait in a queue and start automatically.${quota} Each session can have one job in progress at a time, max ${st.cfg.iters[1]} SDS steps, no mesh export. Every finished model is added to the public gallery below.`;
    }
    pollSystem();
    pollQueue();
    lockGenerate();
  } catch {
    $("#mode").textContent = "static build (pre-generated)"; $("#mode").className = "pill static";
    $("#go").disabled = $("#bench").disabled = true; $("#offline").hidden = false;
    $("#menuOffline").hidden = false;
    document.querySelectorAll('#menu .mode[data-go="create"], #menu .mode[data-go="challenge"]').forEach((b) => (b.disabled = true));
    try {
      const sys = await fetchJSON("data/system.json");
      $("#gpu").textContent = `generated on ${sys.name}`;
    } catch { $("#gpu").textContent = "GPU: offline"; }
  }
  loadComparison();
  await loadRuns();
  // open on a curated MVDream example rather than whatever was generated last
  const featured = st.runs.find((r) => !r.community && !r.mine && r.guidance === "mvdream") || st.runs[0];
  if (featured) await selectRun(featured.name);
  else { drawMetrics(); showBench([]); }
  let resume = null;
  if (st.live) {
    try { resume = sessionStorage.getItem("studio.job"); } catch { /* storage unavailable */ }
    if (resume) { st.jobId = resume; poll(); }
  }
  showScore();
  if (!resume) showMenu();   // START state
})();
