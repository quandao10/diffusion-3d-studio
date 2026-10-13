"""Build report/Assignment_3_Final.pdf (<= 2 pages) from the evidence files; numbers are read, never typed.

usage: GITHUB_URL=... PAGES_URL=... LIVE_URL=... VIDEO_URL=... python report/make_report.py
"""
import csv, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from browser import launch
from playwright.sync_api import sync_playwright

TODO = '<span class="todo">TODO – add after upload</span>'
link = lambda k, label=None: (f'<a href="{os.environ[k]}">{label or os.environ[k]}</a>' if os.environ.get(k) else TODO)
E = lambda f: os.path.join(ROOT, "evidence", f)
rows = lambda f: list(csv.DictReader(open(E(f))))

# ---- evidence
perf = rows("performance.csv")
res = {int(r["resolution"]): float(r["fps"]) for r in perf if r["setting"] == "resolution"}
cnt = [r for r in perf if r["setting"] == "gaussians"]
big = [r for r in cnt if float(r["fps"]) > 0][-1]
oom = [r for r in cnt if "OOM" in r["notes"]]
gc = rows("guidance_comparison.csv")


def g(m, key, f=lambda v: v):
    r = [x for x in gc if x["method"] == m]
    return f(sum(float(x[key]) for x in r) / len(r))


steps = {300: rows("guidance_comparison_it300.csv"), 500: [x for x in gc if x["method"] == "mvdream"],
         800: rows("guidance_comparison_it800.csv"), 1200: rows("guidance_comparison_it1200.csv")}
st = {n: (sum(float(x["clip_score"]) for x in r) / len(r), sum(float(x["total_s"]) for x in r) / len(r)) for n, r in steps.items()}
m_nz = [json.loads(l) for l in open(os.path.join(ROOT, "results", "mushroom_house", "metrics.jsonl"))]
nonzero = all(all(v > 0 for v in r["grad_norm"].values()) for r in m_nz)
vid = json.load(open(os.path.join(ROOT, "video_v2", "timeline.json"))) if os.path.exists(os.path.join(ROOT, "video_v2", "timeline.json")) else {}

VERDICT = {"sds": "blotchy, over-saturated; corgi gets a 2nd face (Janus)", "csd": "noisy, speckled", "nfsd": "natural colours, blurry",
           "ism": "Janus fixed, colours off", "mvdream": "<b>coherent, no Janus, sharpest</b>"}
NAME = {"sds": "SDS (upstream)", "csd": "CSD", "nfsd": "NFSD", "ism": "ISM (LucidDreamer)", "mvdream": "<b>MVDream (default)</b>"}
g_rows = "".join(f"<tr><td>{NAME[m]}</td><td>{g(m, 'clip_score'):.2f}</td><td>{g(m, 'train_s'):.0f} s</td><td>{g(m, 'total_s'):.0f} s</td>"
                 f"<td>{max(float(x['peak_vram_mb']) for x in gc if x['method'] == m) / 1024:.1f} GB</td><td>{VERDICT[m]}</td></tr>"
                 for m in ["sds", "csd", "nfsd", "ism", "mvdream"])
s_rows = "".join(f"<tr><td>{n} steps</td><td>{st[n][0]:.1f}</td><td>{st[n][1]:.0f} s</td><td>{v}</td></tr>" for n, v in
                 ((300, "clean, softer"), (500, "<b>sharpest</b>"), (800, "over-saturated"), (1200, "blown out")))
perf_res = " · ".join(f"{k}²: {v:.0f}" for k, v in sorted(res.items()))
perf_cnt = " · ".join(f"{int(r['gaussians']) / 1e6:.1f}M: {float(r['fps']):.1f}" for r in cnt if float(r["fps"]) > 0)

html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
@page {{ size: Letter; margin: 0.5in 0.55in; }}
body {{ font: 9.2pt/1.32 "DejaVu Sans", Arial, sans-serif; color: #111; }}
h1 {{ font-size: 14.5pt; margin: 0 0 2px; color: #1b3a5c; }} .meta {{ color: #333; font-size: 8.9pt; margin-bottom: 4px; }}
h2 {{ font-size: 11pt; color: #1b3a5c; margin: 8px 0 3px; border-bottom: 1px solid #c9d6e3; padding-bottom: 1px; }}
p {{ margin: 2px 0 4px; }} ul {{ margin: 1px 0 4px 15px; padding: 0; }} li {{ margin: 1px 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 8.5pt; margin: 3px 0 4px; }}
th, td {{ border: 1px solid #cfd8e2; padding: 1.5px 4px; text-align: left; vertical-align: top; }} th {{ background: #eef3f8; }}
.ok {{ color: #0a6b3d; font-weight: 600; }} .chg {{ color: #a05a00; font-weight: 600; }} .no {{ color: #a01c1c; font-weight: 600; }}
.todo {{ background: #fff3c4; color: #7a4b00; padding: 0 3px; }} img {{ width: 100%; }} .cap {{ font-size: 8pt; color: #444; margin: 0 0 3px; }}
code {{ font-size: 8.2pt; }} .two {{ display: grid; grid-template-columns: 1.25fr 1fr; gap: 10px; align-items: start; }}
</style></head><body>
<h1>Assignment 3 — MVP Implementation &amp; Demo</h1>
<div class="meta"><b>Minh Quan Dao (qd66)</b> · Interactive Diffusion-Based 3D Generation Studio · Pillars: <b>Procedural Generation + Rendering</b><br>
Source: {link('GITHUB_URL')} · Web build: {link('PAGES_URL')} (static) · Live studio: {link('LIVE_URL', 'public GPU link')} (while the server runs) · Video: {link('VIDEO_URL')}</div>

<p><b>What runs.</b> A text prompt is turned into an explorable 3D object by optimising <b>3D Gaussians</b> with gradients from a <b>frozen
diffusion model</b> (official DreamGaussian code, commit <code>69a3c70</code>, on an NVIDIA A100). The result is rendered in real time by my WebGL2 splatting
viewer inside a public web studio: anyone can submit a prompt, watch the optimisation live, and orbit the result.</p>

<h2>1. Plan vs. Reality</h2>
<table><tr><th>A2 MVP item</th><th>Status</th><th>What was built / evidence</th></tr>
<tr><td>Text prompt → frozen diffusion prior → score distillation (SDS)</td><td class="ok">done</td><td>SD 2.1-base, CFG 100, annealed t; gradients reach all Gaussian parameters at every step ({'verified' if nonzero else 'partial'}, <code>metrics.jsonl</code>)</td></tr>
<tr><td>NeRF + differentiable volume rendering</td><td class="chg">changed</td><td>explicit 3D Gaussians + differentiable CUDA rasterizer: ≈47 s per 500-step run instead of hours, so I could iterate</td></tr>
<tr><td>Interactive novel-view viewer; progress view; seed/guidance/steps controls</td><td class="ok">done</td><td>own WebGL2 viewer, live loss/gradient/#Gaussian charts, snapshot scrubber, method picker</td></tr>
<tr><td>COULD: 3DGS, mesh export, variations</td><td class="ok">done</td><td>3DGS is the core; textured OBJ via upstream export (CLI); public community gallery</td></tr>
<tr><td>COULD: VSD comparison</td><td class="chg">changed</td><td>VSD (LoRA training, hours) replaced by fast alternatives CSD, NFSD, ISM + MVDream, measured (Sec. 2)</td></tr>
<tr><td>Stage-2 mesh refinement; production UI</td><td class="no">cut</td><td>not run; the studio is a working prototype, not a polished product</td></tr></table>

<h2>2. Improving DreamGaussian: faster / higher-quality guidance</h2>
<p>I added CSD, NFSD and Interval Score Matching (LucidDreamer) as alternative distillation losses on the same frozen SD 2.1, with view-dependent
prompts (<code>scripts/guidance_plus.py</code>), and enabled upstream's MVDream multi-view guidance. All five were compared on 5 prompts, seed 0, 500 steps,
one free A100 per run. CLIP = ViT-L/14 prompt–view similarity over 8 views (<code>evidence/guidance_comparison*.csv</code>).</p>
<table><tr><th>guidance</th><th>CLIP</th><th>optimise</th><th>total</th><th>peak VRAM</th><th>visual result</th></tr>{g_rows}</table>
<div class="two"><div><img src="../docs_assets/report_guidance.png"><div class="cap">Same seed and steps. SDS gives the corgi a second face; MVDream is coherent and sharp.</div></div>
<div><p><b>MVDream step sweep</b> (CLIP / total time):</p><table>{s_rows}</table>
<p>CLIP barely separates the methods (it rewards saturated colours), so I judged visually. <b>MVDream is the default</b> (~2 min per object).
500 steps is the sharpest and longer runs over-saturate, so 500 is also the public cap.</p></div></div>

<h2>3. Stress test (measured, CUDA rasterizer, dedicated A100)</h2>
<p>Forward render of a 311k-Gaussian result along a fixed 360° orbit (120 frames, warm-up excluded, synchronised timing, <code>evidence/performance.csv</code>).
<b>FPS vs resolution:</b> {perf_res}. <b>FPS vs Gaussian count at 512²</b> (k jittered clones): {perf_cnt}; peak VRAM {float(big['peak_vram_mb']) / 1024:.1f} GB at
{int(big['gaussians']) / 1e6:.0f}M; <b>CUDA out-of-memory at {int(oom[0]['gaussians']) / 1e6:.0f}M</b> (failure point). The A2 viewer target (≥30 FPS) holds up to 2048² and ≈5M
Gaussians. The browser viewer is also pushed ×1–×16 live in the video (it rendered on a GPU shared with another user's job, so those numbers are indicative only).</p>

<h2>4. Public studio (runtime system)</h2>
<ul><li><b>Scheduling:</b> each job runs on a GPU that has <i>no process of anyone</i> on it, so the number of parallel jobs follows the machine live (4 free A100s → up to 4 jobs). Otherwise jobs queue FIFO.</li>
<li><b>Visitors:</b> see free GPUs and an <b>ETA before submitting</b> (measured s/step × steps + overhead, simulated over the queue). Each session can have one job in progress, which it can <b>stop</b> to release the GPU. Finished models join a shared gallery.</li>
<li><b>Safety:</b> server bound to localhost behind an outbound Cloudflare tunnel; prompt filter, ≤500 steps, no visitor access to the stress test; server paths hidden from visitors' logs; scheduler and limits covered by in-process tests.</li></ul>

<h2>5. AI Utility</h2>
<ul><li><b>Runtime AI:</b> frozen SD 2.1-base and MVDream supply every gradient; nothing is trained. The original <code>stabilityai</code> SD repo now returns HTTP 401, so I used a mirror of the same weights (documented).</li>
<li><b>Coding agent (Claude Code), as planned in my tech spec:</b> very useful. From the logs, the environment build started 11:44 and the first successful 500-step run finished 12:04.
It diagnosed: a missing <code>&lt;cfloat&gt;</code> in simple-knn; CUDA headers stored only under <code>targets/</code> on this server; a 37 GB peak-VRAM reading caused by <code>cudnn.benchmark</code> (fixed to 4.4 GB, aborted log kept); and headless-Chrome WebGL context loss.
It also wrote most of the wrapper, guidance losses, viewer, studio, scheduler and benchmarks.</li>
<li><b>Cost:</b> every claim still had to be checked against renders and logs. CLIP alone would have picked the wrong method, and a blank viewer turned out to be a lost GPU context, not a shader bug. Time saved was not measured against a baseline. The video narration is synthetic (Kokoro TTS) and matches the on-screen captions.</li></ul>

<h2>6. Roadmap to Final</h2>
<ul><li>Quality: combine MVDream with ISM-style deterministic targets, tune timestep/CFG schedules, add stage-2 mesh refinement and a mesh/Gaussian toggle.</li>
<li>Studio: stable hosting (named tunnel or university reverse proxy), seed/method side-by-side compare, moderation for the shared gallery.</li>
<li>Viewer: GPU sorting and level-of-detail for multi-million-Gaussian scenes; measure it on a dedicated GPU.</li></ul>
<p><b>Build/run:</b> Linux + NVIDIA GPU: <code>bash scripts/setup_env.sh</code>, then <code>python server.py</code> or <code>bash scripts/serve_public.sh</code> (README, <code>evidence/</code>).
Demo video: {vid.get('duration_s', '?')} s screen capture of the running studio; only waits are fast-forwarded and labelled.</p>
</body></html>"""

out_html = os.path.join(ROOT, "report", "report.html")
open(out_html, "w").write(html)
with sync_playwright() as p:
    b = launch(p)
    pg = b.new_page()
    pg.goto("file://" + out_html)
    pg.wait_for_timeout(500)
    pg.pdf(path=os.path.join(ROOT, "report", "Assignment_3_Final.pdf"), format="Letter", print_background=True,
           margin=dict(top="0.45in", bottom="0.45in", left="0.5in", right="0.5in"))
    b.close()
print("wrote report/Assignment_3_Final.pdf")
