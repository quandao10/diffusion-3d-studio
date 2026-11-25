"""Build report/Assignment_6_Feature_Complete.pdf (<= 2 pages): feature-complete release notes, plan check, user study
(simulated subjects, instructor-approved), action items for Assignment 7, schedule, links.
Numbers come from evidence/ (playtest/telemetry_summary.csv, playtest/S*_events.json, v1_render_single_browser.csv, judge_eval.csv).

usage: LIVE_URL=... python report/make_report_a6.py
"""
import csv, json, os, statistics, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from browser import launch
from playwright.sync_api import sync_playwright

REPO = "https://github.com/quandao10/diffusion-3d-studio"
PAGES = "https://quandao10.github.io/diffusion-3d-studio/"
ZIP = REPO + "/archive/refs/tags/v1.0.0-rc.zip"
LIVE = os.environ.get("LIVE_URL", "")
EV = os.path.join(ROOT, "evidence")
PT = os.path.join(EV, "playtest")

tel = {r["subject"]: r for r in csv.DictReader(open(os.path.join(PT, "telemetry_summary.csv")))}
sv = json.load(open(os.path.join(PT, "surveys.json")))
rounds = {s: [e["data"] for e in json.load(open(os.path.join(PT, f"{s}_events.json"))) if e["type"] == "round_end"] for s in tel}
je = list(csv.DictReader(open(os.path.join(EV, "judge_eval.csv"))))
acc = f"{sum(r['correct'] == 'True' for r in je)}/{len(je)}"
render = list(csv.DictReader(open(os.path.join(EV, "v1_render_single_browser.csv"))))
single = next(r for r in render if r["setting"].startswith("single model, studio"))
exh = [r for r in render if r["setting"].startswith("exhibition")]

PERSONA = {"S1": ("fast gamer, skims text", "1440×900"), "S2": ("non-gamer, careful reader", "1366×768"),
           "S3": ("graphics student on a phone", "390×844"), "S4": ("competitive, tests the rules", "1920×1080"),
           "S5": ("impatient; stops, reloads, leaves mid-step", "1280×800")}
all_r = [r for s in rounds for r in rounds[s]]
photo_r = [r for r in all_r if r["source"] == "photos" and r.get("auto_p") is not None]
better = sum(r["p"] >= r["auto_p"] for r in photo_r)
intu = [int(t["intuitive_1_10"]) for t in tel.values()]
fun = [int(t["fun_1_10"]) for t in tel.values()]
fps_med = [float(t["fps_median"]) for t in tel.values()]


def rnd(r):
    d = round(100 * (r["p"] - r["auto_p"])) if r["source"] == "photos" else None
    return (f'{r["target"]} ({r["tier"][0].upper()}) {"✓ " + str(r["points"]) if r["win"] else "✗"}'
            + (f' <span class="{"up" if d >= 0 else "dn"}">{d:+d}</span>' if d is not None else " <i>auto</i>"))


subj_rows = "".join(
    f'<tr><td><b>{s}</b></td><td>{PERSONA[s][0]}</td><td>{PERSONA[s][1]}</td><td>{"; ".join(rnd(r) for r in rounds[s])}</td>'
    f'<td>{t["jobs_done"]}{" + " + t["jobs_stopped"] + " stopped" if t["jobs_stopped"] != "0" else ""}</td><td>{t["photos"]}</td>'
    f'<td>{t["fps_median"]}</td><td>{t["intuitive_1_10"]}</td><td>{t["fun_1_10"]}</td><td>0</td></tr>' for s, t in tel.items())

THEMES = [  # (finding, subjects who reported it) - from the think-aloud logs and questionnaires in evidence/playtest/S*.md
    ("The photo-shoot panel sits on top of the 3D view and hides most of the model, so photos are framed blind (worst on the phone); "
     "the shoot opens with the camera too close and <i>reset view</i> returns to that close-up.", "S1 S2 S3 S4 S5"),
    (f"Photos can lower the score with no warning: in {len(photo_r) - better} of {len(photo_r)} photo rounds the judge was less sure than on the "
     "automatic preview (sailboat 41% → 3%; fox 51% → 29% turned a likely win into 0 pts). Losses are not explained.", "S1 S2 S4"),
    ("While the model trains, the main viewer keeps showing the previous model; progress is only in the snapshot strip below the fold. "
     "Broken snapshot image right after Generate/Retry; status shows “starting” and “#1 in queue” together.", "S1 S2 S4 S5"),
    ("Leaving the photo shoot (Esc → menu) scores the round on the preview without asking and drops the photos taken; after a reload during "
     "a job the Challenge card disappears until the model is ready.", "S5"),
    ("Small UI faults: difficulty badge overlaps the round counter; the playtest bar covers buttons; the prompt box silently cuts text at "
     "120 characters; the forbidden-word warning is below the fold (and “F O X” passes); Leaderboard “Close” returns to the menu; "
     "the survey confirmation disappears too fast; the phone page scrolls sideways.", "S1 S2 S3 S4 S5"),
    ("Technical jargon (CFG, SDS loss, Gaussians, CLIP, guidance methods, raw training log) confuses non-graphics players.", "S2"),
    ("Frame-rate dips when a job starts, a model opens or the photo shoot opens (3 FPS twice, 11 FPS once): PLY parse + first sort run on the main thread.", "S1 S2 S5"),
]
theme_rows = "".join(f'<tr><td>{t}</td><td class="c">{len(w.split())}/5<br><span class="dim">{w}</span></td></tr>' for t, w in THEMES)

PLAN = [
    ("Game loop, Challenge, CLIP judge, render modes (A4)", "done (A4)"),
    ("Rendering pillar in the loop: lighting, stage, photo shoot, exhibition + LOD (A5)", "done (A5)"),
    ("Persistence (SQLite) + leaderboard", "<b>done (this sprint)</b>"),
    ("More targets with easy / medium / hard tiers", f"<b>done</b>: 40 targets, ×1 / ×1.5 / ×2; judge top-1 {acc}"),
    ("Gallery moderation (owner hide)", "<b>done</b>"),
    ("Onboarding, help, empty / error states; final UI and audio assets", "<b>done</b>: How-to-play, Help, nickname, synthesised SFX + mute"),
    ("Playtest instrumentation", "<b>done</b>: ?playtest=ID telemetry + in-app interview"),
    ("Stage-2 mesh refinement + mesh/Gaussian toggle", "<b>cut</b>: doubles GPU time per round; gameplay and judge use Gaussians; .ply download kept"),
    ("Reference render per challenge target", "<b>cut</b>: replaced by the auto-camera preview verdict (A5)"),
    ("Stable named hosting", "<b>cut</b>: needs a registered domain; quick tunnel + GitHub Pages static build"),
    ("GPU sort, per-pixel shading, mobile layout polish", "moved to A7 (optimisation / polish, not features)"),
]
plan_rows = "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in PLAN)

ACTIONS = [
    ("<b>Photo-shoot layout.</b> Move the shoot controls into a bar under the canvas (bottom sheet on phones), open the shoot with the camera "
     "fitted to the model, make <i>reset view</i> fit too, and show a large preview of the exact photo crop.", "5/5"),
    ("<b>Fair photo scoring.</b> Score the round with the better of preview and photos (photos can only help), show the judge's guess per photo "
     "before sending, and explain a loss (what the judge saw, one tip).", "3/5"),
    ("<b>Round state.</b> Ask before leaving a photo shoot instead of auto-scoring; restore the Challenge card after a reload; show the model "
     "forming in the main viewer during training; fix the broken snapshot placeholder and stale view after Stop / Retry.", "4/5"),
    ("<b>Frame-time hitches.</b> Parse PLYs in the sort worker and transfer the buffers; stream the first sort; target no frame over 50 ms when a "
     "model opens. Re-measure with the same 5-session setup.", "3/5"),
    ("<b>UI fixes.</b> Badge overlap, playtest bar overlap, prompt counter (120), forbidden-word warning next to the button + letter-spacing "
     "normalisation, leaderboard returns to the game, visible survey confirmation, phone overflow, plain-language labels in Challenge mode.", "5/5"),
]
act_rows = "".join(f'<li>{a} <span class="dim">(reported by {n})</span></li>' for a, n in ACTIONS)

SCHEDULE = [
    ("9–12 · Oct 26–Nov 22", "Alpha v0.2 (A4), Beta v0.3 (A5): game loop, judge, rendering pillar in the loop", "done"),
    ("13 · Nov 23–29", "<b>Feature freeze (this build, v1.0-rc):</b> persistence, leaderboard, tiers, onboarding, audio, moderation, playtest mode; 5-subject playtest", "done"),
    ("14 · Nov 30–Dec 6", "<b>A7 optimisation sprint:</b> the 5 action items above; re-run the playtest protocol and the render benchmark", "int"),
    ("15 · Dec 7–13", "<b>BUFFER:</b> bug fixes, documentation, final video and v1.0 release only; no new features", "buf"),
]
sched_rows = "".join(f'<tr class="{k}"><td><b>{w}</b></td><td>{e}</td></tr>' for w, e, k in SCHEDULE)

html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
body {{ font: 8.5pt/1.27 "DejaVu Sans", Arial, sans-serif; color: #111; margin: 0; }}
h1 {{ font-size: 13.6pt; margin: 0 0 2px; color: #1b3a5c; }} .meta {{ color: #333; font-size: 8.2pt; margin-bottom: 2px; }}
h2 {{ font-size: 10.2pt; color: #1b3a5c; margin: 6px 0 2px; border-bottom: 1px solid #c9d6e3; padding-bottom: 1px; }}
p {{ margin: 2px 0 3px; }} ul, ol {{ margin: 1px 0 3px 15px; padding: 0; }} li {{ margin: 0.6px 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 7.7pt; margin: 3px 0; }}
th, td {{ border: 1px solid #cfd8e2; padding: 1.5px 4px; text-align: left; vertical-align: top; }} th {{ background: #eef3f8; }}
td.c {{ text-align: center; white-space: nowrap; }} .dim {{ color: #666; font-size: 7.2pt; }} .up {{ color: #137a3e; font-weight: 700; }} .dn {{ color: #b3261e; font-weight: 700; }}
tr.done td {{ background: #eaf6ef; }} tr.int td {{ background: #f3f0fb; }} tr.buf td {{ background: #fff4e0; }}
.two {{ display: grid; grid-template-columns: 1.05fr 1fr; gap: 8px; }}
.shots {{ display: grid; grid-template-columns: 1.15fr 1.15fr 0.42fr 1.15fr; gap: 5px; align-items: start; margin: 3px 0; }}
.shots img {{ width: 100%; border: 1px solid #cfd8e2; border-radius: 3px; }} .cap {{ font-size: 7.1pt; color: #444; }}
code {{ font-size: 7.6pt; }} a {{ color: #1a56a8; }} .note {{ background: #fff8e6; border: 1px solid #f0d9a0; padding: 2px 5px; border-radius: 4px; }}
</style></head><body>
<h1>Assignment 6 — Feature Complete (v1.0-rc) &amp; Playtest</h1>
<div class="meta"><b>Minh Quan Dao (qd66)</b> · Interactive Diffusion-Based 3D Generation Studio · Procedural Generation + Rendering ·
<b>Play:</b> {f'<a href="{LIVE}">live studio</a> (lab A100 server) · ' if LIVE else ''}<a href="{PAGES}">static build</a> ·
<b>Build zip:</b> <a href="{ZIP}">v1.0.0-rc.zip</a> · <b>Repo:</b> <a href="{REPO}">{REPO.replace('https://', '')}</a></div>

<h2>1. Feature-complete release notes (since Beta v0.3)</h2>
<div class="two"><div>
<p><b>Final feature integrations (this sprint):</b></p>
<ul>
<li><b>Server-side scoring + persistence</b> (SQLite): a round is scored from the job id only (target, judge, points never come from the browser), once per job, and survives server restarts.</li>
<li><b>Leaderboard</b> with a nickname chosen on the first visit; points, wins and trophies persist across visits.</li>
<li><b>40 targets in 3 tiers</b> (easy ×1, medium ×1.5, hard ×2) with a difficulty picker. Judge top-1 with 40 labels: {acc} on all existing results.</li>
<li><b>Onboarding:</b> How-to-play screen on the first visit (and behind <i>? Help</i>), pulsing shutter in the photo shoot, empty states.</li>
<li><b>Audio:</b> synthesised WebAudio effects (start, model ready, shutter, win, loss) with a mute toggle.</li>
<li><b>Gallery moderation</b> (owner hides community models) and <b>playtest mode</b> <code>?playtest=ID</code>: clicks, screens, photos, jobs,
rounds, errors and FPS every 5 s are logged; the 5 interview questions are asked in the app.</li></ul>
</div><div>
<table><tr><th>Production-plan item (A4 / A5 schedule)</th><th>Status at the freeze</th></tr>{plan_rows}</table>
</div></div>
<div class="shots">
<div><img src="../docs_assets/a6_onboard.png"><div class="cap">First visit: objective + controls, nickname.</div></div>
<div><img src="../docs_assets/a6_shoot_panel.png"><div class="cap">S2's photo shoot: the panel hides the model (top finding).</div></div>
<div><img src="../docs_assets/a6_phone.png"><div class="cap">S3, phone.</div></div>
<div><img src="../docs_assets/a6_leaderboard.png"><div class="cap">Leaderboard after the playtest.</div></div></div>

<h2>2. User study</h2>
<p class="note"><b>Methodology.</b> With the instructor's approval the 5 subjects were <b>simulated</b>: 5 independent AI agents, each with its own persona and
screen size, not classmates. Each played the <b>live public build</b> in its own browser and saw it only through screenshots and on-screen text
(no source code, no API calls). The protocol was the same for each session. Each subject got only the premise ("describe things in words, an AI turns them into 3D,
there is a challenge mode") and no explanation of the controls. They thought aloud (timestamped transcripts), and an observer took notes on failures, missed prompts, glitches,
FPS and facilitator hints. They then answered the post-play interview (5 questions) in the app. Telemetry was logged in <code>?playtest=S1…S5</code> mode.
Caveats: agents act slower than people, so session times are not human times. All 5 sessions ran at once on one machine, which lowered the
measured FPS (control below). Raw data: <code>evidence/playtest/</code> (S1–S5.md, event logs, surveys.json, telemetry_summary.csv).</p>
<table><tr><th>subj.</th><th>persona</th><th>screen</th><th>rounds: target (tier) result, photo vs preview (conf. pts)</th><th>jobs</th><th>photos</th><th>FPS med.</th><th>controls /10</th><th>fun /10</th><th>hints</th></tr>{subj_rows}</table>
<p><b>Summary:</b> all 5 finished at least one Challenge round with <b>no facilitator hints</b>. Four read the How-to-play screen; S5 skipped it with Esc
and learned the forbidden-word rule from the Challenge card. They played {len(all_r)} scored rounds and won {sum(r["win"] for r in all_r)}
(tiers played: {", ".join(sorted({r["tier"] for r in all_r}))}; nobody chose easy). Controls were rated {min(intu)}–{max(intu)}/10
(mean {statistics.mean(intu):.1f}) and fun {min(fun)}–{max(fun)}/10 (mean {statistics.mean(fun):.1f}). There were no JavaScript errors and no crashes.
Stop, reload and leaving mid-step all recovered without errors, though not always cleanly. <b>Performance:</b> the FPS median during the sessions was {min(fps_med):.0f}–{max(fps_med):.0f},
with 5 browsers sharing one GPU host. A single browser on the same build gives {single['fps_raf']} FPS for one model ({single['gpu_ms']} ms GPU) and
{min(float(r['fps_raf']) for r in exh):.0f}–{max(float(r['fps_raf']) for r in exh):.0f} FPS for exhibitions, so the ≥ 30 FPS target holds. The 1–3 FPS stalls when a model opens are real.</p>
<table><tr><th>Subject feedback and observations (grouped)</th><th style="width:11%">reported</th></tr>{theme_rows}</table>

<h2>3. Action items for Assignment 7 (from this playtest data)</h2>
<ol>{act_rows}</ol>

<h2>4. Schedule (weeks from Sep 1)</h2>
<table><tr><th style="width:18%">Week</th><th>Work</th></tr>{sched_rows}</table>
<p class="cap">Run: <code>bash scripts/setup_env.sh</code> then <code>bash scripts/serve_public.sh</code> (Linux + NVIDIA GPU ≥ 12 GB). Playtest protocol for real
sessions: <code>report/playtest_protocol.md</code>; analysis: <code>python scripts/playtest_summary.py</code>.</p>
</body></html>"""

out_html = os.path.join(ROOT, "report", "report_a6.html")
open(out_html, "w").write(html)
with sync_playwright() as p:
    b = launch(p)
    pg = b.new_page()
    pg.goto("file://" + out_html)
    pg.wait_for_timeout(600)
    pg.pdf(path=os.path.join(ROOT, "report", "Assignment_6_Feature_Complete.pdf"), format="Letter", print_background=True,
           margin=dict(top="0.38in", bottom="0.38in", left="0.42in", right="0.42in"))
    b.close()
print("wrote report/Assignment_6_Feature_Complete.pdf")
