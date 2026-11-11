"""Photo shoot for the rounds in evidence/beta_rounds.json, through the real web UI (headless Chrome, A100 WebGL).

For every finished round:
  1. rendering ablation: the same 3 camera angles under 4 looks (raw splats on white, raw on dark, studio dark,
     studio light); each set of 3 photos goes to /api/judge_photos -> target probability per look;
  2. the actual round: result preview -> "Start the photo shoot" -> 3 photos in the studio-light look -> judge -> final card.
A scripted player picks the angles (front, front +-35 deg); no human choices are involved.
Writes evidence/beta_shoot.csv, evidence/beta_photos/*.jpg and docs_assets/a5_*.png.

usage: python scripts/shoot_rounds.py https://<tunnel>.trycloudflare.com
"""
import base64, csv, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from browser import launch
from playwright.sync_api import sync_playwright

BASE = sys.argv[1].rstrip("/")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PH = os.path.join(ROOT, "evidence", "beta_photos"); os.makedirs(PH, exist_ok=True)
rounds = [r for r in json.load(open(os.path.join(ROOT, "evidence", "beta_rounds.json"))) if r.get("status") == "done"]
LOOKS = [("raw_white", False, "white"), ("raw_dark", False, "black"), ("studio_dark", True, "dark"), ("studio_light", True, "light")]
ANGLES = [(0, 12), (35, 20), (-35, 6)]   # (azimuth offset from the model's front, elevation)

SHOOT_JS = """async ([angles, lit, backdrop, front]) => {
  const s = window.studio, v = s.viewer, out = [];
  s.setStudioLook(lit); s.setBackdrop(backdrop); v.autoRotate = 0;
  for (const [da, el] of angles) {
    Object.assign(v.cam, { az: front + da, el, r: 2.5 }); v.setLight(front + da + 40, 45);
    await new Promise((r) => setTimeout(r, 900));
    out.push(s.capturePhoto());
  }
  return out;
}"""

rows = []
with sync_playwright() as p:
    b = launch(p)
    pg = b.new_page(viewport={"width": 1500, "height": 950})
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(BASE + "/"); time.sleep(7)
    for i, r in enumerate(rounds):
        j = pg.evaluate(f"fetch('api/jobs/{r['id']}').then(x => x.json())")
        run = j["run"]; front = run.get("front_azimuth") or 0
        pg.evaluate("async (name) => { const s = window.studio; document.querySelector('#menu').hidden = true; await s.loadRuns(); await s.selectRun(name); }", run["name"])
        time.sleep(4)
        auto = run.get("judge") or {}
        row = dict(target=r["target"], prompt=r["prompt"], wall_s=r["wall_s"], auto_top=auto.get("top", [{}])[0].get("label"),
                   auto_p=auto.get("target_p"), auto_win=auto.get("win"))
        for name, lit, bd in LOOKS:   # rendering ablation, same angles
            shots = pg.evaluate(SHOOT_JS, [ANGLES, lit, bd, front])
            jd = pg.evaluate("([t, pr, im]) => fetch('api/judge_photos', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({target: t, prompt: pr, images: im})}).then(x => x.json())",
                             [r["target"], r["prompt"], shots])
            row[f"{name}_p"] = jd.get("target_p"); row[f"{name}_win"] = jd.get("win"); row[f"{name}_top"] = jd["top"][0]["label"]
            open(os.path.join(PH, f"{r['target']}_{name}.jpg"), "wb").write(base64.b64decode(shots[0].split(",", 1)[1]))
        # the round itself, through the UI: preview -> photo shoot -> final verdict
        pg.evaluate("(j) => { const s = window.studio; s.enterMode('challenge'); s.showResult(j); }", j)
        time.sleep(1.5)
        if i == 0: pg.screenshot(path=os.path.join(ROOT, "docs_assets", "a5_preview.png"))
        pg.click("#r-shoot"); time.sleep(1.5)
        pg.evaluate("(bd) => window.studio.setBackdrop(bd)", "light")
        for da, el in ANGLES:
            pg.evaluate(f"(() => {{ const v = window.studio.viewer; Object.assign(v.cam, {{ az: {front + da}, el: {el}, r: 2.5 }}); v.setLight({front + da + 40}, 45); }})()")
            time.sleep(1.0); pg.click("#snapBtn"); time.sleep(0.3)
        if i == 0: pg.screenshot(path=os.path.join(ROOT, "docs_assets", "a5_shoot.png"))
        pg.click("#submitShots")
        pg.wait_for_selector("#result:not([hidden])", timeout=60000); time.sleep(1)
        row["final_banner"] = pg.inner_text("#r-banner")
        row["score"] = pg.inner_text("#score")
        pg.screenshot(path=os.path.join(ROOT, "docs_assets", f"a5_final_{r['target']}.png"))
        print(json.dumps(row))
        rows.append(row)
        pg.click("#r-explore"); time.sleep(0.5)
    # exhibition (session trophies first) for the report
    pg.evaluate("window.studio.showMenu()"); time.sleep(0.5)
    pg.click('#menu .mode[data-go="exhibit"]'); time.sleep(10)
    pg.evaluate("window.studio.setBackdrop('dark')"); time.sleep(2)
    pg.locator(".canvaswrap").screenshot(path=os.path.join(ROOT, "docs_assets", "a5_exhibit.png"))
    print("exhibit HUD:", pg.inner_text("#hud").replace("\n", " | "))
    print("page errors:", errs)
    b.close()
with open(os.path.join(ROOT, "evidence", "beta_shoot.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("wrote evidence/beta_shoot.csv")
