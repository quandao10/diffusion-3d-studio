"""Synthesise the voiceover clips with Kokoro-82M (run in the `tts` env). Writes video/audio/<scene>.wav + durations.json"""
import json, os, sys
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo_script import fill_stress

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "video", "audio")
os.makedirs(OUT, exist_ok=True)
scenes = fill_stress(os.path.join(ROOT, "evidence", "performance.csv"))

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
