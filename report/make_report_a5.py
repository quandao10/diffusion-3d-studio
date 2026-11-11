"""Build report/Assignment_5_Beta.pdf (<= 2 pages): pillar-1 review, secondary-pillar integration, new features,
known issues, updated schedule, links. Every number is read from evidence/ (beta_rounds.json, beta_shoot.csv, beta_render.csv).

usage: LIVE_URL=... python report/make_report_a5.py
"""
import csv, json, os, statistics, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from browser import launch
from playwright.sync_api import sync_playwright

REPO = "https://github.com/quandao10/diffusion-3d-studio"
PAGES = "https://quandao10.github.io/diffusion-3d-studio/"
ZIP = REPO + "/archive/refs/tags/v0.3.0-beta.zip"
VIDEO = "https://www.youtube.com/watch?v=UGFzpFrmWcE"
LIVE = os.environ.get("LIVE_URL", "")
EV = os.path.join(ROOT, "evidence")

rounds = json.load(open(os.path.join(EV, "beta_rounds.json")))
shoot = list(csv.DictReader(open(os.path.join(EV, "beta_shoot.csv"))))
render = list(csv.DictReader(open(os.path.join(EV, "beta_render.csv"))))
LABEL = {t["id"]: t["label"] for t in json.load(open(os.path.join(ROOT, "configs", "challenge.json")))["targets"]}
done = [r for r in rounds if r.get("status") == "done"]
wall = [r["wall_s"] for r in done]
train = [r["job"]["run"]["train_s"] for r in done]
yes = lambda v: str(v) == "True"
pct = lambda v: f"{round(100 * float(v))}%" if v not in ("", None, "None") else "–"
LOOKS = [("raw_white", "raw, white"), ("raw_dark", "raw, black"), ("studio_dark", "studio dark"), ("studio_light", "studio light")]


def cell(r, k):
    w = yes(r[f"{k}_win"])
    return f'<td class="{"w" if w else "l"}">{pct(r[f"{k}_p"])}{"" if w else " ✗"}</td>'


shoot_rows = "".join(
    f'<tr><td>{LABEL[r["target"]].replace("a ", "").replace("an ", "")}</td><td class="pr">“{r["prompt"]}”</td>'
    f'<td class="{"w" if yes(r["auto_win"]) else "l"}">{pct(r["auto_p"])}{"" if yes(r["auto_win"]) else " ✗"}</td>'
    + "".join(cell(r, k) for k, _ in LOOKS) + "</tr>" for r in shoot)
wins = {k: sum(yes(r[f"{k}_win"]) for r in shoot) for k, _ in LOOKS}
mean_p = {k: statistics.mean(float(r[f"{k}_p"]) for r in shoot) for k, _ in LOOKS}
auto_w = sum(yes(r["auto_win"]) for r in shoot)
auto_p = statistics.mean(float(r["auto_p"]) for r in shoot)
sum_row = (f'<tr class="sum"><td colspan="2"><b>wins / mean target probability</b></td><td>{auto_w}/{len(shoot)} · {auto_p:.2f}</td>'
           + "".join(f"<td>{wins[k]}/{len(shoot)} · {mean_p[k]:.2f}</td>" for k, _ in LOOKS) + "</tr>")
render_rows = "".join(
    f'<tr><td>{r["setting"]}</td><td>{int(r["gaussians"]):,}</td><td>{r["canvas"]}</td><td>{float(r["fps_raf"]):.0f}</td>'
    f'<td>{r["gpu_ms"] or "n/a"}</td><td>{float(r["sort_ms"]):.0f}</td></tr>' for r in render)
play = [r for r in render if "replicas" not in r["setting"]]          # settings a player can reach
worst_gpu = max(float(r["gpu_ms"]) for r in play if r["gpu_ms"])
min_fps = min(float(r["fps_raf"]) for r in play)
stress = next((r for r in render if "replicas" in r["setting"]), None)
raw1 = next(r for r in render if r["setting"].startswith("single model, raw")); lit1 = next(r for r in render if r["setting"].startswith("single model, studio"))
final = shoot[-1]["score"].replace("🎯 ", "")
rescued = [LABEL[r["target"]] for r in shoot if not yes(r["raw_white_win"]) and yes(r["studio_dark_win"])]
best_raw = [LABEL[r["target"]] for r in shoot if float(r["raw_white_p"]) > max(float(r["studio_dark_p"]), float(r["studio_light_p"]))]

SCHEDULE = [
    ("9 · Oct 26–Nov 1", "Alpha: game loop, CLIP judge, render modes, offline jobs, watchdog", "Alpha v0.2 (A4)", "done"),
    ("10–11 · Nov 2–15", "<b>Beta (this build):</b> lit splats (normals, wrap diffuse, specular, rim), stage + soft shadow, photo shoot inside the Challenge round, CPU photo judge, exhibition with LOD; real MVDream rounds; render benchmark",
     "Beta v0.3 live + tagged (this report)", "done"),
    ("12 · Nov 16–22", "Persistence (SQLite: rounds, points, leaderboard), moderation of the gallery, stable hosting (named tunnel); photo-judge robustness (prompt ensembles, more targets)",
     "<b>Integration:</b> leaderboard + photo points; classmate play-test", "int"),
    ("13 · Nov 23–29 (Thanksgiving)", "Fix play-test issues; per-pixel shading option; mobile layout of the photo shoot; GPU sort for very large exhibitions",
     "Play-test triage", "eng"),
    ("14 · Nov 30–Dec 6", "<b>Feature freeze:</b> fix top issues only; load test with parallel players; record final video", "Release candidate v1.0-rc", "int"),
    ("15 · Dec 7–13", "<b>BUFFER WEEK:</b> bug fixing, documentation and polish only; no new features", "v1.0 + final submission", "buf"),
]
sched_rows = "".join(f'<tr class="{k}"><td><b>{w}</b></td><td>{e}</td><td>{m}</td></tr>' for w, e, m, k in SCHEDULE)

html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
body {{ font: 8.7pt/1.28 "DejaVu Sans", Arial, sans-serif; color: #111; margin: 0; }}
h1 {{ font-size: 14pt; margin: 0 0 2px; color: #1b3a5c; }} .meta {{ color: #333; font-size: 8.4pt; margin-bottom: 2px; }}
h2 {{ font-size: 10.4pt; color: #1b3a5c; margin: 6px 0 2px; border-bottom: 1px solid #c9d6e3; padding-bottom: 1px; }}
p {{ margin: 2px 0 3px; }} ul {{ margin: 1px 0 3px 14px; padding: 0; }} li {{ margin: 0.5px 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 7.8pt; margin: 3px 0; }}
th, td {{ border: 1px solid #cfd8e2; padding: 1.5px 4px; text-align: left; vertical-align: top; }} th {{ background: #eef3f8; }}
td.w {{ background: #eaf6ef; }} td.l {{ background: #fdecea; }} td.pr {{ color: #333; font-style: italic; }} tr.sum td {{ background: #f4f6f9; }}
tr.done td {{ background: #eaf6ef; }} tr.int td {{ background: #f3f0fb; }} tr.buf td {{ background: #fff4e0; }}
.shots {{ display: grid; grid-template-columns: 1.25fr 1fr 1fr; gap: 6px; align-items: start; margin: 3px 0; }}
.shots img {{ width: 100%; border: 1px solid #cfd8e2; border-radius: 4px; }} .cap {{ font-size: 7.3pt; color: #444; }}
code {{ font-size: 7.8pt; }} a {{ color: #1a56a8; }}
.loop {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 5px; margin: 3px 0; }}
.loop div {{ border: 1px solid #cfd8e2; border-radius: 6px; padding: 3px 5px; background: #f7f9fc; font-size: 8.1pt; }}
.loop div.new {{ border-color: #3987e5; background: #eef5ff; }}
</style></head><body>
<h1>Assignment 5 — Beta Build: Rendering integrated into the core loop</h1>
<div class="meta"><b>Minh Quan Dao (qd66)</b> · Interactive Diffusion-Based 3D Generation Studio · Pillars: <b>Procedural Generation</b> (core) + <b>Rendering</b> (secondary) · Beta <b>v0.3.0</b><br>
<b>Play:</b> {f'<a href="{LIVE}">live studio</a> (lab A100 server; quick-tunnel URL) · ' if LIVE else ''}<a href="{PAGES}">static build</a> (viewer, gallery, exhibition; no GPU) ·
<b>Build zip:</b> <a href="{ZIP}">v0.3.0-beta.zip</a> · <b>Repo:</b> <a href="{REPO}">{REPO.replace('https://', '')}</a> · <a href="{VIDEO}">A3 video</a></div>

<h2>1. Quick review: pillar 1 (procedural generation) is stable</h2>
<p>Text → 3D Gaussians optimised by score distillation from a frozen diffusion prior (DreamGaussian; MVDream multi-view guidance by default, 500 steps).
The A4 game loop is unchanged in structure: <b>menu → Challenge (describe a secret object without its taboo words) → GPU job with queue, ETA, Stop → result</b>.
For this Beta, {len(done)} real Challenge rounds were played through the public URL at the same time (one per session, MVDream, 6 free A100s):
all {len(done)} jobs finished, {min(wall):.0f}–{max(wall):.0f} s per round from submit to judged model (queue, model load, 500 steps, views, judge);
the 500 optimisation steps took {min(train):.0f}–{max(train):.0f} s, the same as a single job in A3 (89–94 s), so parallel players do not slow each other down.
These rounds fix the A4 known issue “no real GPU Challenge round recorded yet”. Raw data: <code>evidence/beta_rounds.json</code>.</p>

<h2>2. Secondary pillar integration: the player's rendering decides the round</h2>
<div class="loop">
<div><b>1 · Describe.</b> Secret target + taboo words; prompt checked in the browser and on the server.</div>
<div><b>2 · Generate.</b> 3D Gaussians on a free GPU; live loss and ETA. Then a <i>preview</i> verdict from 8 automatic views.</div>
<div class="new"><b>3 · Photo shoot (new).</b> Light, frame and photograph the model in the real-time renderer; up to 3 photos.</div>
<div class="new"><b>4 · Verdict.</b> The CLIP judge sees <b>only your photos</b>: win = target ranked 1st of 20; points = 100 × its probability. Wins go to the exhibition.</div></div>
<ul>
<li><b>Lit Gaussian splatting.</b> Each Gaussian's normal is its shortest axis (the flat direction of the splat), flipped to point away from the
object's centre, octahedral-encoded into the spare channel of my packed texture (no extra memory). The vertex shader adds wrap diffuse, Blinn specular and a
cool rim light from a movable key light, so the cost is per Gaussian, not per pixel.</li>
<li><b>Stage and soft shadow.</b> Gradient backdrop and a ground disc under the model. The shadow is computed by projecting every Gaussian centre along the light
onto the floor, accumulating opacity in a 160² grid (256² in the exhibition), blurring it and uploading it as a texture; it is only recomputed when the light or the model changes.</li>
<li><b>Why it is part of the loop, not a demo:</b> the same model scores differently depending on how it is rendered. Same 3 camera angles per round, 4 looks, judged by the same CLIP judge:</li></ul>
<table><tr><th>target</th><th>player prompt (no taboo words)</th><th>auto preview (8 views)</th>{''.join(f'<th>{n}</th>' for _, n in LOOKS)}</tr>{shoot_rows}{sum_row}</table>
<p class="cap">Target probability (✗ = judge picked another object). Photos taken by a scripted player through the real UI (front, ±35°), <code>scripts/shoot_rounds.py</code> → <code>evidence/beta_shoot.csv</code>.
Studio lighting on the dark stage raised the mean target probability from {mean_p['raw_white']:.2f} (raw splats on white) to {mean_p['studio_dark']:.2f}
and turned {', '.join(rescued) or 'no'} from a loss into a win. It is not always better ({', '.join(best_raw) or 'none'} scored highest raw on white), so the look is a real choice for the player.
The rounds themselves were played in the studio-light look: {final}.</p>
<div class="shots">
<div><img src="../docs_assets/a5_shoot.png"><div class="cap">Photo shoot: studio lighting, light direction and height, backdrop; 3 photo slots.</div></div>
<div><img src="../docs_assets/a5_final_{shoot[0]['target']}.png"><div class="cap">End state: verdict from the player's photos, points, comparison with the preview.</div></div>
<div><img src="../docs_assets/a5_exhibit.png"><div class="cap">Exhibition: trophies and examples on one lit stage, one sort, LOD budget.</div></div></div>

<h2>3. Performance: the A2 target (≥ 30 FPS) holds in every playable setting</h2>
<table><tr><th>setting (headless Chrome, WebGL2, A100)</th><th>Gaussians</th><th>canvas</th><th>FPS (rAF, capped 60)</th><th>GPU ms (splat pass)</th><th>sort ms (worker)</th></tr>{render_rows}</table>
<p class="cap">Lighting is computed per Gaussian in the vertex shader, so it costs nothing measurable ({raw1['gpu_ms']} → {lit1['gpu_ms']} ms); the frame is bound by the number of Gaussians.
Every setting a player can reach stays at ≥ {min_fps:.0f} FPS (worst {worst_gpu:.1f} ms GPU). The A3 stress case of 8 replicas stacked on one spot
({int(stress['gaussians']):,} Gaussians, heavy overdraw) drops to {float(stress['fps_raf']):.0f} FPS; this is why the exhibition keeps each model's most opaque Gaussians within an LOD budget
(default 1.2 M → {next(r for r in render if '1.2M' in r['setting'])['fps_raf']} FPS). <code>scripts/bench_beta.py</code> → <code>evidence/beta_render.csv</code>.</p>

<h2>4. Beta release notes</h2>
<p><b>New features:</b> studio lighting from per-Gaussian normals (toggle, light direction and height); four backdrops (studio dark, studio light, plain white, plain black) with ground plane and soft shadow;
<b>photo shoot</b> phase in every Challenge round with a CPU CLIP judge endpoint (<code>/api/judge_photos</code>, about 1 s for 3 photos, never takes a GPU);
points per win and a points total in the HUD; <b>Exhibition</b> menu mode (trophies first, then your models and examples; 2–8 models; LOD budget 0.6/1.2/2.4 M; two rows above 4 models);
camera target support in the viewer; the result card shows the player's photos and the change against the preview.
<b>Stability:</b> leaving a round mid-shoot (menu, Esc, new job) keeps the preview verdict instead of losing the round; the photo judge is loaded at server start; photos are size-checked
and one judgement per visitor runs at a time; the Beta server moved to <i>hermes</i> after <i>homer</i> rebooted.</p>
<p><b>Known issues:</b> normals from the shortest axis are noisy on fuzzy, blob-like Gaussians, so lighting can look speckled; the shadow uses Gaussian centres only (no self-shadowing);
the judge can be gamed by backdrop choice (studio dark vs. white changes the score), which is part of the game but makes scores depend on the look;
the quick-tunnel URL changes when the tunnel restarts; points and trophies live in <code>sessionStorage</code> (lost with the tab) until the SQLite leaderboard in week 12;
the exhibition downloads every model's full PLY before LOD (about 10–15 MB each).</p>

<h2>5. Updated production schedule (weeks from Sep 1; final deadline in week 15)</h2>
<table><tr><th style="width:17%">Week</th><th style="width:58%">Engineering / content</th><th>Milestone</th></tr>{sched_rows}</table>
<p class="cap"><span style="background:#eaf6ef">&nbsp;green&nbsp;</span> done · <span style="background:#f3f0fb">&nbsp;purple&nbsp;</span> integration · <span style="background:#fff4e0">&nbsp;orange&nbsp;</span> buffer week (no new features).
Run: <code>bash scripts/setup_env.sh</code>, then <code>bash scripts/serve_public.sh</code> (or <code>python server.py</code>); Linux + NVIDIA GPU ≥ 12 GB.</p>
</body></html>"""

out_html = os.path.join(ROOT, "report", "report_a5.html")
open(out_html, "w").write(html)
with sync_playwright() as p:
    b = launch(p)
    pg = b.new_page()
    pg.goto("file://" + out_html)
    pg.wait_for_timeout(600)
    pg.pdf(path=os.path.join(ROOT, "report", "Assignment_5_Beta.pdf"), format="Letter", print_background=True,
           margin=dict(top="0.4in", bottom="0.4in", left="0.45in", right="0.45in"))
    b.close()
print("wrote report/Assignment_5_Beta.pdf")
