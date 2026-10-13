"""Synthesise the voiceover clips with Kokoro-82M (run in the `tts` env). Writes video/audio/<scene>.wav + durations.json"""
import json, os, sys
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.environ.get("SCRIPT", "demo_script")          # demo_script2 = final studio video
OUT = os.path.join(ROOT, os.environ.get("VIDEO_DIR", "video"), "audio")
os.makedirs(OUT, exist_ok=True)
mod = importlib.import_module(SCRIPT)
scenes = mod.fill_stress(os.path.join(ROOT, "evidence", "performance.csv")) if hasattr(mod, "fill_stress") else mod.SCENES

from kokoro import KPipeline
pipe = KPipeline(lang_code="a")
dur = {}
for s in scenes:
    audio = np.concatenate([np.asarray(a) for _, _, a in pipe(s["say"], voice=os.environ.get("VOICE", "af_heart"), speed=1.05)])
    sf.write(os.path.join(OUT, f"{s['id']}.wav"), audio, 24000)
    dur[s["id"]] = round(len(audio) / 24000, 2)
    print(s["id"], dur[s["id"]], "s")
json.dump(dur, open(os.path.join(OUT, "durations.json"), "w"), indent=1)
print("total narration", round(sum(dur.values()), 1), "s")
