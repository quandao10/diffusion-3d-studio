"""Summarise the playtest from the studio database (run/studio.db, written by the live server in ?playtest=ID mode).

Per subject: session length, clicks, clicks on disabled controls (a confusion signal), time to the first generation,
time from model-ready to the end of the round, photos taken, FPS during play, errors, the round result and the
post-play questionnaire. Writes evidence/playtest/telemetry_summary.csv and evidence/playtest/surveys.json.

usage: python scripts/playtest_summary.py [S1 S2 ...]      (default: subjects S1-S5)
"""
import csv, json, os, shutil, sqlite3, statistics, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "evidence", "playtest")
os.makedirs(OUT, exist_ok=True)
subjects = sys.argv[1:] or [f"S{i}" for i in range(1, 6)]

tmp = os.path.join(tempfile.mkdtemp(), "studio.db")       # copy: the server may be writing
shutil.copy(os.path.join(ROOT, "run", "studio.db"), tmp)
db = sqlite3.connect(tmp)
db.row_factory = sqlite3.Row

rows, surveys = [], {}
for s in subjects:
    ev = [dict(r) for r in db.execute("SELECT session, t, type, data FROM events WHERE subject=? ORDER BY id", (s,))]
    if not ev:
        continue
    for e in ev:
        e["data"] = json.loads(e["data"]) if e["data"] else {}
    first = lambda typ, **kw: next((e["t"] for e in ev if e["type"] == typ and all(e["data"].get(k) == v for k, v in kw.items())), None)
    fps = [e["data"]["fps"] for e in ev if e["type"] == "fps" and e["data"].get("fps")]
    rounds = [e["data"] for e in ev if e["type"] == "round_end"]
    jobs = [e["data"] for e in ev if e["type"] == "job_end"]
    t_ready = first("job_end", status="done")
    t_round = first("round_end")
    sv = [json.loads(r["answers"]) for r in db.execute("SELECT answers FROM surveys WHERE subject=? ORDER BY id", (s,))]
    if sv:
        surveys[s] = sv[-1]
    rows.append(dict(
        subject=s,
        sessions=len({e["session"] for e in ev}),               # >1 = the page was reloaded
        minutes=round(max(e["t"] for e in ev) / 60, 1),          # longest session timeline (t restarts per session)
        clicks=sum(e["type"] == "click" for e in ev),
        disabled_clicks=sum(e["type"] == "click" and e["data"].get("disabled") for e in ev),
        help_opened=sum(e["type"] == "click" and e["data"].get("id") == "helpBtn" for e in ev),
        first_submit_s=first("job_submit"),
        jobs_done=sum(j.get("status") == "done" for j in jobs),
        jobs_stopped=sum(j.get("status") == "cancelled" for j in jobs),
        ready_to_verdict_s=round(t_round - t_ready, 1) if t_ready is not None and t_round is not None and t_round > t_ready else None,
        photos=sum(e["type"] == "photo" for e in ev),
        rounds=len(rounds),
        wins=sum(bool(r.get("win")) for r in rounds),
        points=sum(r.get("points") or 0 for r in rounds),
        sources="/".join(r.get("source", "?") for r in rounds),
        fps_median=statistics.median(fps) if fps else None,
        fps_min=min(fps) if fps else None,
        js_errors=sum(e["type"] == "js_error" for e in ev),
        ui_errors=sum(e["type"] == "ui_error" for e in ev),
        intuitive_1_10=(surveys.get(s) or {}).get("intuitive_1_10"),
        fun_1_10=(surveys.get(s) or {}).get("fun_1_10"),
    ))
    with open(os.path.join(OUT, f"{s}_events.json"), "w") as f:
        json.dump(ev, f, indent=0)

if rows:
    with open(os.path.join(OUT, "telemetry_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
json.dump(surveys, open(os.path.join(OUT, "surveys.json"), "w"), indent=1)
for r in rows:
    print(r)
print(f"{len(rows)} subjects with telemetry, {len(surveys)} questionnaires -> evidence/playtest/")
