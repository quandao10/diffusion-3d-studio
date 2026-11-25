"""Persistent game data (SQLite, run/studio.db): Challenge jobs, scored rounds, leaderboard, playtest telemetry and surveys.

Rounds are scored on the server only: the client sends a job id (and photos); the target, the judge result and the points
come from the server, and each job can be scored once.
"""
import json
import os
import sqlite3
import threading
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
DB = os.environ.get("STUDIO_DB", os.path.join(ROOT, "run", "studio.db"))
_lock = threading.Lock()
_con = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS challenge_jobs (job_id TEXT PRIMARY KEY, target TEXT, prompt TEXT, out TEXT, rel TEXT, name TEXT,
  player TEXT, created REAL);
CREATE TABLE IF NOT EXISTS rounds (id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT UNIQUE, player TEXT, nick TEXT, target TEXT,
  tier TEXT, prompt TEXT, source TEXT, auto_p REAL, p REAL, win INTEGER, points INTEGER, run_rel TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, subject TEXT, session TEXT, t REAL, type TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS surveys (id INTEGER PRIMARY KEY AUTOINCREMENT, subject TEXT, session TEXT, ts REAL, answers TEXT);
CREATE TABLE IF NOT EXISTS hidden_runs (name TEXT PRIMARY KEY, ts REAL);
"""


def con():
    global _con
    if _con is None:
        os.makedirs(os.path.dirname(DB), exist_ok=True)
        _con = sqlite3.connect(DB, check_same_thread=False)
        _con.row_factory = sqlite3.Row
        _con.executescript(SCHEMA)
    return _con


def q(sql, args=(), one=False):
    with _lock:
        cur = con().execute(sql, args)
        rows = [dict(r) for r in cur.fetchall()]
        con().commit()
    return (rows[0] if rows else None) if one else rows


def add_challenge_job(job, player):
    q("INSERT OR REPLACE INTO challenge_jobs VALUES (?,?,?,?,?,?,?,?)",
      (job["id"], job["target"], job["prompt"], job["out"], job["rel"], job["name"], player, time.time()))


def challenge_job(job_id):
    return q("SELECT * FROM challenge_jobs WHERE job_id=?", (job_id,), one=True)


def round_of(job_id):
    return q("SELECT * FROM rounds WHERE job_id=?", (job_id,), one=True)


def add_round(**r):
    keys = ["job_id", "player", "nick", "target", "tier", "prompt", "source", "auto_p", "p", "win", "points", "run_rel"]
    q(f"INSERT INTO rounds ({','.join(keys)}, ts) VALUES ({','.join('?' * len(keys))}, ?)", tuple(r.get(k) for k in keys) + (time.time(),))
    return round_of(r["job_id"])


def leaderboard(limit=10):
    return q("""SELECT player, (SELECT nick FROM rounds r2 WHERE r2.player = r.player ORDER BY ts DESC LIMIT 1) AS nick,
                SUM(points) AS points, SUM(win) AS wins, COUNT(*) AS rounds, MAX(points) AS best
                FROM rounds r WHERE player IS NOT NULL GROUP BY player ORDER BY points DESC, wins DESC LIMIT ?""", (limit,))


def player_stats(player):
    return q("SELECT COALESCE(SUM(points),0) AS points, COALESCE(SUM(win),0) AS wins, COUNT(*) AS rounds FROM rounds WHERE player=?", (player,), one=True)


def add_events(subject, session, events):
    with _lock:
        con().executemany("INSERT INTO events (subject, session, t, type, data) VALUES (?,?,?,?,?)",
                          [(subject, session, float(e.get("t", 0)), str(e.get("type", ""))[:40], json.dumps(e.get("data"))[:2000]) for e in events])
        con().commit()


def add_survey(subject, session, answers):
    q("INSERT INTO surveys (subject, session, ts, answers) VALUES (?,?,?,?)", (subject, session, time.time(), json.dumps(answers)[:8000]))


def hide_run(name, hidden=True):
    if hidden:
        q("INSERT OR REPLACE INTO hidden_runs VALUES (?,?)", (name, time.time()))
    else:
        q("DELETE FROM hidden_runs WHERE name=?", (name,))


def hidden_runs():
    return {r["name"] for r in q("SELECT name FROM hidden_runs")}
