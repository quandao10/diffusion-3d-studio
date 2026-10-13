"""Record the demo video: drives the live studio (server.py) in headless Chrome rendering on the NVIDIA GPU and
captures the page with the Chrome DevTools screencast (timestamped JPEG frames). Everything on screen is the real
software running in real time; the only edits are two waits (mesh export, stress-test run) fast-forwarded 8x and labelled on screen.

  1. GPU=4 python server.py                      (dreamgaussian env)
  2. python scripts/tts.py                       (tts env)  -> video/audio/*.wav
  3. python scripts/record_demo.py               (dreamgaussian env) -> video/demo.mp4
"""
import base64, json, os, shutil, subprocess, sys, time, urllib.request

import numpy as np
import soundfile as sf
from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from browser import launch
from demo_script import LIVE_PROMPT, LIVE_SEED, fill_stress

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = os.environ.get("STUDIO_URL", "http://127.0.0.1:8549/")
VID = os.path.join(ROOT, "video")
FRAMES = os.path.join(VID, "frames")
AUDIO = os.path.join(VID, "audio")
FF = 8.0  # fast-forward factor for the export wait

scenes = {s["id"]: s for s in fill_stress(os.path.join(ROOT, "evidence", "performance.csv"))}
dur = json.load(open(os.path.join(AUDIO, "durations.json")))


def api(path):
    with urllib.request.urlopen(URL + path, timeout=10) as r:
        return json.load(r)


class Recorder:
    def __init__(self, page):
        self.pg = page
        self.frames = []  # (timestamp, path)
        self.marks = []  # (scene_id, t)
        self.ff = []  # (t0, t1) fast-forward ranges (recording clock)
        shutil.rmtree(FRAMES, ignore_errors=True)
        os.makedirs(FRAMES)
        self.cdp = page.context.new_cdp_session(page)
        self.cdp.on("Page.screencastFrame", self.on_frame)

    def on_frame(self, ev):
        t = ev["metadata"]["timestamp"]
        p = os.path.join(FRAMES, f"{len(self.frames):06d}.jpg")
        with open(p, "wb") as f:
            f.write(base64.b64decode(ev["data"]))
        self.frames.append((t, p))
        self.cdp.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]})

    def start(self):
        self.cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 90, "maxWidth": 1920, "maxHeight": 1080, "everyNthFrame": 1})
        self.wait(0.5)

    def stop(self):
        self.cdp.send("Page.stopScreencast")

    def now(self):
        return time.time()

    def wait(self, s):
        self.pg.wait_for_timeout(int(s * 1000))

    def scene(self, sid, min_s=None):
        """Show caption, mark narration start; returns the time by which narration finishes."""
        s = scenes[sid]
        self.pg.evaluate("([t, s]) => studio.caption(t, s)", [s["title"], s["sub"]])
        t = self.now()
        self.marks.append((sid, t))
        print(f"[rec] {sid} @ {t:.1f}", flush=True)
        return t + dur[sid] + 0.6

    def until(self, t_end):
        while self.now() < t_end:
            self.wait(0.1)


def smooth_scroll(pg, selector, rec, block="start"):
    pg.evaluate("([s, b]) => document.querySelector(s).scrollIntoView({behavior: 'smooth', block: b})", [selector, block])
    rec.wait(1.2)


def drag(pg, rec, box, dx, dy, steps=40):
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    pg.mouse.move(x, y)
    pg.mouse.down()
    for i in range(steps):
        pg.mouse.move(x + dx * (i + 1) / steps, y + dy * (i + 1) / steps)
        rec.wait(0.03)
    pg.mouse.up()


def main():
    assert not api("api/job").get("active"), "GPU busy - wait for the current job"
    with sync_playwright() as p:
        browser = launch(p)
        ctx = browser.new_context(viewport={"width": 1600, "height": 900}, device_scale_factor=1.2)
        pg = ctx.new_page()
        pg.goto(URL)
        pg.wait_for_function("() => window.studio && studio.st.runs.length && studio.viewer.sortedCount > 0", timeout=60000)
        pg.evaluate("() => studio.selectRun('mushroom_house')")
        pg.wait_for_function("() => studio.st.sel && studio.st.sel.name === 'mushroom_house' && studio.viewer.sortedCount > 0", timeout=60000)
        rec = Recorder(pg)
        rec.start()
        rec.wait(1.0)

        # 1-2: intro + live server
        end = rec.scene("intro")
        rec.wait(9)
        pg.evaluate("() => studio.flash('#gpu')")
        rec.until(end)

        # 3: type prompt and launch
        end = rec.scene("prompt")
        pg.click("#prompt", click_count=3)
        pg.keyboard.press("Control+A")
        pg.keyboard.type(LIVE_PROMPT, delay=70)
        pg.fill("#seed", str(LIVE_SEED))
        rec.wait(0.5)
        pg.click("#go")
        rec.until(end)

        # 4-7: training narration, timed against the real job
        def status():
            j = api("api/job")
            return (j.get("progress") or {}).get("status", "starting"), (j.get("progress") or {}).get("step", 0), j

        end = rec.scene("loading")
        rec.until(end)
        while status()[0] in ("starting", "loading_diffusion_model"):
            rec.wait(0.3)
        for sid in ("train1", "train2", "train3"):
            if sid == "train2":
                pg.evaluate("() => studio.flash('.monitor')")
            if sid == "train3":
                pg.evaluate("() => studio.flash('.strip')")
            end = rec.scene(sid)
            while rec.now() < end and status()[0] == "training":
                rec.wait(0.3)
            rec.until(end) if status()[0] == "training" else None
        while status()[0] == "training":
            rec.wait(0.3)

        # 8: export (fast-forwarded in post)
        t0 = rec.now()
        rec.scene("export")
        while True:
            st, _, j = status()
            if not j.get("active"):
                break
            rec.wait(0.5)
        # app.js reloads runs and selects the new one when the job ends
        pg.wait_for_function("() => studio.st.sel && studio.st.sel.prompt === %s && studio.viewer.sortedCount > 0" % json.dumps(LIVE_PROMPT), timeout=120000)
        rec.ff.append((t0, rec.now()))
        rec.wait(0.5)

        # 9: viewer interaction
        end = rec.scene("viewer")
        pg.evaluate("() => { document.querySelector('#rotate').click(); }")  # stop auto-rotate while dragging
        box = pg.locator("#gl").bounding_box()
        drag(pg, rec, box, 260, 0)
        drag(pg, rec, box, -120, 90)
        for _ in range(8):
            pg.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            pg.mouse.wheel(0, -120)
            rec.wait(0.12)
        rec.wait(0.8)
        for _ in range(8):
            pg.mouse.wheel(0, 120)
            rec.wait(0.12)
        pg.evaluate("() => { document.querySelector('#rotate').click(); }")
        rec.until(end)

        # 10: primitives + scrubber
        end = rec.scene("primitives")
        for v in np.linspace(1, 0.12, 20):
            pg.evaluate("v => { const e = document.querySelector('#scalemod'); e.value = v; e.dispatchEvent(new Event('input')); }", float(v))
            rec.wait(0.06)
        rec.wait(1.5)
        n = int(pg.evaluate("() => +document.querySelector('#scrub').max"))
        for i in range(0, n + 1):
            pg.evaluate("i => { const e = document.querySelector('#scrub'); e.value = i; e.dispatchEvent(new Event('input')); }", i)
            rec.wait(0.12)
        for v in np.linspace(0.12, 1, 15):
            pg.evaluate("v => { const e = document.querySelector('#scalemod'); e.value = v; e.dispatchEvent(new Event('input')); }", float(v))
            rec.wait(0.05)
        rec.until(end)

        # 11-12: CUDA stress test, run live on the GPU from the page
        smooth_scroll(pg, ".stress", rec)
        end = rec.scene("stress1")
        pg.click("#bench")
        rec.until(end)
        t0 = rec.now()
        pg.evaluate("() => studio.caption('Stress test running on the A100   ⏩ fast-forward 8×', 'rows appear as each setting finishes · same model, same orbit, same GPU')")
        while api("api/job").get("active"):
            rec.wait(0.5)
        rec.wait(1.5)
        if rec.now() - t0 > 12:
            rec.ff.append((t0, rec.now() - 1.0))
        name = pg.evaluate("() => studio.st.sel.name")
        scenes.update({s["id"]: s for s in fill_stress(os.path.join(ROOT, "results", name, "benchmark.csv"))})
        rec.until(rec.scene("stress2"))

        # 13: browser stress sweep
        end = rec.scene("stress3")
        pg.evaluate("() => { studio.sweep(); }")
        rec.wait(1.0)
        smooth_scroll(pg, ".viewer", rec)
        rec.wait(13)
        smooth_scroll(pg, "#t-web", rec, "center")
        pg.wait_for_function("() => document.querySelectorAll('#t-web tbody tr').length >= 5", timeout=90000)
        rec.wait(1.0)
        rec.until(end)

        # 14: outro
        smooth_scroll(pg, ".how", rec, "center")
        rec.until(rec.scene("outro"))
        rec.wait(1.0)
        pg.evaluate("() => studio.caption('')")
        rec.wait(0.8)
        rec.stop()
        rec.wait(0.5)
        browser.close()

    assemble(rec)


def assemble(rec):
    frames, marks, ff = rec.frames, rec.marks, rec.ff
    t_start = frames[0][0]

    def vt(t):  # recording clock -> video clock
        out, prev = 0.0, t_start
        for a, b in ff:
            if t <= a:
                break
            out += a - prev
            out += (min(t, b) - a) / FF
            prev = b
            if t <= b:
                return out
        return out + (t - prev)

    # frame list with video-time durations
    lst = os.path.join(VID, "frames.txt")
    times = [vt(t) for t, _ in frames]
    total = times[-1] + 0.5
    with open(lst, "w") as f:
        for i, (_, path) in enumerate(frames):
            d = (times[i + 1] if i + 1 < len(frames) else total) - times[i]
            if d <= 0:
                continue
            f.write(f"file '{path}'\nduration {d:.4f}\n")
        f.write(f"file '{frames[-1][1]}'\n")

    # voiceover track
    sr = 24000
    track = np.zeros(int((total + 2) * sr), dtype=np.float32)
    timeline = []
    for sid, t in marks:
        a, _ = sf.read(os.path.join(AUDIO, f"{sid}.wav"), dtype="float32")
        o = int(vt(t) * sr)
        track[o:o + len(a)] += a[: max(0, len(track) - o)]
        timeline.append((sid, round(vt(t), 2), round(len(a) / sr, 2)))
    track = np.clip(track, -1, 1)
    wav = os.path.join(VID, "voiceover.wav")
    sf.write(wav, track[: int(total * sr)], sr)
    json.dump(dict(timeline=timeline, fast_forward=[(round(vt(a), 2), round(b - a, 1), FF) for a, b in ff],
                   duration_s=round(total, 1), frames=len(frames)), open(os.path.join(VID, "timeline.json"), "w"), indent=1)

    out = os.path.join(VID, "demo.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-i", wav,
                    "-vf", "scale=1920:1080:flags=lanczos,format=yuv420p", "-r", "30", "-c:v", "libx264", "-preset", "slow",
                    "-crf", "18", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", out], check=True)
    print(f"[rec] wrote {out}  duration {total:.1f}s  frames {len(frames)}  timeline {timeline}")


if __name__ == "__main__":
    main()
