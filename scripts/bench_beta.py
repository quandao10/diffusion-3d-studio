"""Beta renderer cost: raw splats vs studio lighting (+ground/shadow) vs exhibition scenes, in headless Chrome (WebGL2).
Median over 6 s per setting of rAF FPS (display-capped at 60), GPU ms of the splat pass (EXT_disjoint_timer_query)
and worker sort ms. Writes evidence/beta_render.csv.

usage: python scripts/bench_beta.py http://127.0.0.1:8549
"""
import csv, os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from browser import launch
from playwright.sync_api import sync_playwright

BASE = sys.argv[1].rstrip("/")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = [  # (label, setup js)
    ("single model, raw splats", "async s => { await s.selectRun(s.st.runs.find(r => r.name === 'a_cute_owl_plush_toy_mvdream_s0').name); s.setStudioLook(false); s.setBackdrop('white'); }"),
    ("single model, studio lighting + stage", "async s => { s.setStudioLook(true); s.setBackdrop('dark'); }"),
    ("single model x8 replicas, studio", "async s => { s.setMult(8); }"),
    ("exhibition 4 models, studio", "async s => { s.setMult(1); document.querySelector('#exhibitN').value = '4'; document.querySelector('#exhibitBudget').value = '1200000'; await s.showExhibit(); }"),
    ("exhibition 8 models, LOD 1.2M", "async s => { document.querySelector('#exhibitN').value = '8'; await s.showExhibit(); }"),
    ("exhibition 8 models, LOD 2.4M", "async s => { document.querySelector('#exhibitBudget').value = '2400000'; await s.showExhibit(); }"),
]
rows = []
with sync_playwright() as p:
    b = launch(p)
    pg = b.new_page(viewport={"width": 1600, "height": 1000})
    pg.goto(BASE + "/"); time.sleep(7)
    pg.evaluate("document.querySelector('#menu').hidden = true")
    for label, js in CASES:
        pg.evaluate(f"({js})(window.studio)")
        time.sleep(4)
        pg.evaluate("window.studio.viewer.autoRotate = 30")   # keep re-sorting, like a player orbiting
        fps, gpu, sort = [], [], []
        for _ in range(12):
            time.sleep(0.5)
            s = pg.evaluate("(() => { const v = window.studio.viewer; return { fps: v.frameTimes.length, gpu: v.gpuMs, sort: v.sortMs, n: v.count, w: v.canvas.width, h: v.canvas.height }; })()")
            fps.append(s["fps"]); sort.append(s["sort"])
            if s["gpu"] is not None: gpu.append(s["gpu"])
        row = dict(setting=label, gaussians=s["n"], canvas=f"{s['w']}x{s['h']}", fps_raf=statistics.median(fps),
                   gpu_ms=round(statistics.median(gpu), 2) if gpu else "", sort_ms=round(statistics.median(sort), 1))
        print(row); rows.append(row)
    b.close()
with open(os.path.join(ROOT, "evidence", "beta_render.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("wrote evidence/beta_render.csv")
