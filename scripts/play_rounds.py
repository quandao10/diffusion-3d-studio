"""Play real Beta Challenge rounds against a running studio (public URL): submit prompts that avoid the taboo words,
wait for the GPU jobs, and write the auto-camera judge results to evidence/beta_rounds.json.
The photo shoot for each round is done afterwards in the browser (scripts/shoot_rounds.py).

usage: python scripts/play_rounds.py https://<tunnel>.trycloudflare.com
"""
import json, os, secrets, sys, time, urllib.request

BASE = sys.argv[1].rstrip("/")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUNDS = [  # (target id, a prompt written without the target's taboo words)
    ("owl", "a round fluffy night bird with big yellow eyes and ear tufts"),
    ("pineapple", "a spiky tropical fruit with a crown of green leaves"),
    ("hamburger", "a grilled beef patty sandwich with lettuce, tomato and cheese in a sesame bun"),
    ("lighthouse", "a tall red and white striped coastal tower with a lamp room on top"),
    ("penguin", "a black and white flightless antarctic bird standing upright"),
    ("donut", "a pink frosted fried ring pastry with rainbow sprinkles"),
]


def call(path, body=None, sid=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", "User-Agent": "studio-beta-test", **({"X-Studio-Session": sid} if sid else {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


jobs = []
for tid, prompt in ROUNDS:
    sid = secrets.token_hex(16)
    j = call("/api/generate", {"prompt": prompt, "seed": 0, "iters": 500, "guidance": "mvdream", "target": tid}, sid)
    jobs.append(dict(target=tid, prompt=prompt, id=j["id"], sid=sid, submitted=time.time()))
    print("submitted", tid, j)
pending = list(jobs)
while pending:
    time.sleep(10)
    for jb in list(pending):
        j = call(f"/api/jobs/{jb['id']}")
        if j["status"] in ("done", "failed", "cancelled"):
            pending.remove(jb)
            jb.update(status=j["status"], wall_s=round(time.time() - jb["submitted"], 1), job=j)
            jd = (j.get("run") or {}).get("judge") or {}
            print(f"{jb['target']:<11} {j['status']:<6} {jb['wall_s']:>6}s  auto: top={jd.get('top', [{}])[0].get('label')} target_p={jd.get('target_p')} win={jd.get('win')}")
json.dump(jobs, open(os.path.join(ROOT, "evidence", "beta_rounds.json"), "w"), indent=1)
print("wrote evidence/beta_rounds.json")
