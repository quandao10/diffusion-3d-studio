"""Record demo video v2 (final studio) from the running server, same capture/assembly as record_demo.py.
Only waits (model loading, the rest of the optimisation) are fast-forwarded 6x and labelled on screen.

  SCRIPT=demo_script2 VIDEO_DIR=video_v2 python scripts/tts.py   (tts env)
  python scripts/record_demo2.py                                    (dreamgaussian env) -> video_v2/demo.mp4
"""
import json, os, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import record_demo as R
from browser import launch
from demo_script2 import LIVE_PROMPT, STOP_PROMPT, SCENES
from playwright.sync_api import sync_playwright

ROOT = R.ROOT
R.VID = os.path.join(ROOT, "video_v2")
R.FRAMES = os.path.join(R.VID, "frames")
R.AUDIO = os.path.join(R.VID, "audio")
R.FF = 6.0
R.scenes = {s["id"]: s for s in SCENES}
R.dur = json.load(open(os.path.join(R.AUDIO, "durations.json")))
URL = R.URL


def job_state(pg):
    return pg.evaluate("() => [document.querySelector('#jobstatus').textContent, studio.st.jobId]")


def main():
    with sync_playwright() as p:
        browser = launch(p)
        ctx = browser.new_context(viewport={"width": 1600, "height": 900}, device_scale_factor=1.2)
        pg = ctx.new_page()
        pg.on("dialog", lambda d: d.accept())
        pg.goto(URL)
        pg.wait_for_function("() => window.studio && studio.st.runs.length && studio.viewer.sortedCount > 0 && studio.st.lastQ", timeout=90000)
        name = pg.evaluate("() => (studio.st.runs.find(r => r.prompt === 'a delicious hamburger') || studio.st.runs[0]).name")
        pg.evaluate(f"() => studio.selectRun('{name}')")
        pg.wait_for_function("() => studio.viewer.sortedCount > 0", timeout=60000)
        rec = R.Recorder(pg)
        rec.start()
        rec.wait(1.0)

        # 1 intro
        end = rec.scene("intro")
        rec.wait(8); pg.evaluate("() => studio.flash('#queue')")
        rec.until(end)

        # 2 controls: method picker + ETA
        end = rec.scene("controls")
        pg.select_option("#guidance", "sds"); rec.wait(1.2)
        pg.select_option("#guidance", "mvdream"); pg.evaluate("() => studio.flash('#guidance')"); rec.wait(1.5)
        pg.evaluate("() => studio.flash('#queuehint')")
        rec.until(end)

        # 3 submit
        end = rec.scene("submit")
        pg.click("#prompt", click_count=3); pg.keyboard.press("Control+A")
        pg.keyboard.type(LIVE_PROMPT, delay=60)
        rec.wait(0.4)
        pg.click("#go")
        rec.until(end)

        # 4 loading (fast-forwarded)
        t0 = rec.now()
        rec.scene("loading")
        while job_state(pg)[0] not in ("training", "failed"):
            rec.wait(0.4)
        rec.ff.append((t0, rec.now()))

        # 5-6 live training
        pg.evaluate("() => studio.flash('.strip')")
        rec.until(rec.scene("train1"))
        end = rec.scene("train2")
        pg.evaluate("() => { studio.flash('.monitor'); studio.flash('#jobbox'); }")
        rec.until(end)

        # 7 rest of the optimisation (fast-forwarded) until the result is loaded
        t0 = rec.now()
        rec.scene("ff_train")
        pg.wait_for_function("() => studio.st.sel && studio.st.sel.prompt === %s && studio.viewer.sortedCount > 0" % json.dumps(LIVE_PROMPT), timeout=600000)
        rec.ff.append((t0, rec.now()))
        rec.wait(0.6)

        # 8 viewer
        end = rec.scene("viewer")
        pg.evaluate("() => { document.querySelector('#rotate').click(); }")
        box = pg.locator("#gl").bounding_box()
        R.drag(pg, rec, box, 300, 0)
        R.drag(pg, rec, box, -150, 70)
        for _ in range(6):
            pg.mouse.wheel(0, -120); rec.wait(0.1)
        rec.wait(0.6)
        for _ in range(6):
            pg.mouse.wheel(0, 120); rec.wait(0.1)
        pg.evaluate("() => { document.querySelector('#rotate').click(); }")
        rec.until(end)

        # 9 one job per session + stop
        pg.select_option("#guidance", "sds")
        pg.click("#prompt", click_count=3); pg.keyboard.press("Control+A")
        pg.keyboard.type(STOP_PROMPT, delay=40)
        pg.click("#go")
        end = rec.scene("stop")
        pg.wait_for_function("() => ['loading diffusion model', 'training'].includes(document.querySelector('#jobstatus').textContent)", timeout=60000)
        rec.wait(2.5)
        pg.evaluate("() => studio.flash('#stop')")
        rec.wait(1.2)
        pg.click("#stop")
        pg.wait_for_function("() => document.querySelector('#jobstatus').textContent === 'cancelled'", timeout=60000)
        rec.until(end)

        # 10 gallery
        end = rec.scene("gallery")
        pg.evaluate("() => studio.flash('#runs')")
        for y in range(0, 700, 70):
            pg.evaluate(f"() => {{ document.querySelector('#runs').scrollTop = {y}; }}"); rec.wait(0.15)
        cactus = pg.evaluate("() => (studio.st.runs.find(r => r.prompt === 'a green cactus in a clay pot') || {}).name")
        if cactus:
            pg.evaluate(f"() => studio.selectRun('{cactus}')")
        rec.until(end)

        # 11-12 algorithm comparison
        R.smooth_scroll(pg, ".algo", rec)
        end = rec.scene("algo")
        rec.wait(6)
        pg.evaluate("() => window.scrollBy({top: 520, behavior: 'smooth'})")
        rec.until(end)
        R.smooth_scroll(pg, ".algo", rec)
        end = rec.scene("steps")
        pg.evaluate("() => studio.flash('#t-steps')")
        rec.until(end)

        # 13 stress test: measured CUDA table + live browser sweep
        R.smooth_scroll(pg, ".stress", rec)
        end = rec.scene("stress")
        pg.evaluate("() => studio.flash('#t-bench')")
        pg.evaluate("() => { studio.sweep(); }")
        rec.wait(1.0)
        pg.wait_for_function("() => document.querySelectorAll('#t-web tbody tr').length >= 4", timeout=120000)
        pg.evaluate("() => studio.flash('#t-web')")
        rec.until(end)
        pg.wait_for_function("() => document.querySelectorAll('#t-web tbody tr').length >= 5", timeout=120000)

        # 14 outro
        R.smooth_scroll(pg, ".how", rec, "center")
        rec.until(rec.scene("outro"))
        rec.wait(1.0)
        pg.evaluate("() => studio.caption('')")
        rec.wait(0.8)
        rec.stop()
        rec.wait(0.5)
        browser.close()
    R.assemble(rec)


if __name__ == "__main__":
    main()
