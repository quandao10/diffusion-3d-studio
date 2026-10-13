"""Interactive Diffusion-Based 3D Generation Studio - GPU server.

Serves the web studio (web/) and runs real DreamGaussian jobs. A scheduler dispatches each queued job to an idle
NVIDIA GPU on this machine (little memory in use, low utilisation, not already running one of our jobs), so several
jobs run in parallel; when every GPU is busy, jobs wait in a FIFO queue and the visitor sees their place in line.

  POST /api/generate        enqueue text -> 3D (scripts/train_logged.py)           -> {id, position}
  GET  /api/jobs/{id}        status / queue position / GPU / live progress / log tail / finished run
  GET  /api/jobs/{id}/metrics?since=n   per-step metrics of a job
  GET  /api/queue            anonymous queue + GPU-pool summary (no prompts)
  POST /api/benchmark        CUDA rasterizer stress test (admin only)
  GET  /api/runs             curated gallery (results/<name>/summary.json)
  GET  /api/system           GPU pool snapshot (cached nvidia-smi)
  GET  /api/config           limits + whether the caller is admin

Admin = a request from localhost that did not come through the tunnel (no CF-Connecting-IP header), e.g. the owner
via `ssh -L`. Public visitors (PUBLIC=1): PER_IP_TOTAL generations (default 1) per PER_IP_WINDOW_S (default 24 h),
keyed by a salted hash of the IP; one active job at a time; prompt filter; capped settings; no mesh export. Their
outputs go to results/public/<id>/ (only shown to them, pruned to the newest KEEP_PUBLIC after PRUNE_MIN_AGE_S).

Usage:  PUBLIC=1 python server.py --port 8549      (binds 127.0.0.1; scripts/serve_public.sh adds the tunnel)
        GPUS=4,5,6,7 restricts the pool; MAX_PARALLEL=n caps simultaneous jobs (0 = every idle GPU)
"""
import argparse
import collections
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
PUBLIC_DIR = os.path.join(RESULTS, "public")
EVIDENCE = os.path.join(ROOT, "evidence")
RUN = os.path.join(ROOT, "run")
PUBLIC = os.environ.get("PUBLIC", "0") == "1"

GPUS = [int(g) for g in os.environ["GPUS"].split(",")] if os.environ.get("GPUS") else None  # None = all GPUs
MAX_PARALLEL = int(os.environ.get("MAX_PARALLEL", 0))  # simultaneous jobs (0 = one per process-free GPU, live)
GPU_FREE_MB = int(os.environ.get("GPU_FREE_MB", 1000))    # a GPU counts as idle below this memory use ...
GPU_FREE_UTIL = int(os.environ.get("GPU_FREE_UTIL", 10))  # ... and this utilisation (%)

PROMPT_MAX = 120
PUBLIC_ITERS = (100, 500)
PUBLIC_CFG = (10.0, 150.0)
QUEUE_MAX = int(os.environ.get("QUEUE_MAX", 20))
PER_IP_TOTAL = int(os.environ.get("PER_IP_TOTAL", 0))  # generations per visitor per window (0 = unlimited)
PER_IP_WINDOW_S = int(os.environ.get("PER_IP_WINDOW_S", 86400))
KEEP_PUBLIC = int(os.environ.get("KEEP_PUBLIC", 150))  # community gallery size (~25 MB per model)
PRUNE_MIN_AGE_S = int(os.environ.get("PRUNE_MIN_AGE_S", 3600))  # never delete a visitor's run younger than this
JOB_TIMEOUT_S = 900
_bl = os.path.join(ROOT, "configs", "blocklist.txt")
BLOCKLIST = [w.strip().lower() for w in open(_bl) if w.strip() and not w.startswith("#")] if os.path.exists(_bl) else []

app = FastAPI(title="Diffusion 3D Studio")
cond = threading.Condition()
jobs = {}                      # id -> job dict
queue = collections.deque()    # ids waiting for a GPU
running = {}                   # job id -> GPU index

os.makedirs(RUN, exist_ok=True)
VISITORS = os.path.join(RUN, "visitors.json")  # salted IP hash -> submit timestamps (survives restarts)
SALT_FILE = os.path.join(RUN, "salt")
if not os.path.exists(SALT_FILE):
    with open(SALT_FILE, "w") as f:
        f.write(secrets.token_hex(16))
SALT = open(SALT_FILE).read().strip()
try:
    history = collections.defaultdict(list, json.load(open(VISITORS)))
except (FileNotFoundError, json.JSONDecodeError):
    history = collections.defaultdict(list)


# ------------------------------------------------------------------ helpers
def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40] or "run"


def caller(request: Request):
    """(ip, is_admin). Through cloudflared every request carries CF-Connecting-IP."""
    host = request.client.host if request.client else ""
    cf = request.headers.get("cf-connecting-ip")
    if host in ("127.0.0.1", "::1") and not cf:
        return "local", True
    return cf or request.headers.get("x-forwarded-for", host).split(",")[0].strip(), False


def visitor_key(ip):
    return hashlib.sha256((SALT + ip).encode()).hexdigest()[:16]


def tail(path, n=40):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 64_000))
            lines = f.read().decode("utf-8", "replace").replace("\r", "\n").splitlines()
        return [l for l in lines if l.strip()][-n:]
    except FileNotFoundError:
        return []


_HIDE = sorted({p for p in (ROOT, sys.prefix, os.path.expanduser("~")) if len(p) > 1}, key=len, reverse=True)


def scrub(lines):
    """Hide server filesystem paths from public visitors' log view."""
    out = []
    for l in lines:
        for p in _HIDE:
            l = l.replace(p, "…")
        out.append(l)
    return out


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def position(jid):
    try:
        return list(queue).index(jid) + 1
    except ValueError:
        return 0


# ------------------------------------------------------------------ GPU pool
_gpu_cache = {"t": 0.0, "rows": []}
_gpu_lock = threading.Lock()


def gpu_pool():
    """nvidia-smi snapshot of the pool incl. every process on each GPU (compute + graphics), cached for 2 s."""
    import xml.etree.ElementTree as ET
    num = lambda s: int(float(s.split()[0])) if s and s.split()[0].replace(".", "").isdigit() else 0
    with _gpu_lock:
        if time.time() - _gpu_cache["t"] > 2:
            rows = []
            try:
                xml = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True, text=True, timeout=15).stdout
                for i, g in enumerate(ET.fromstring(xml).findall("gpu")):  # XML order == nvidia-smi index order
                    if GPUS is not None and i not in GPUS:
                        continue
                    procs = g.findall("processes/process_info")
                    rows.append(dict(index=i, name=g.findtext("product_name", "GPU"),
                                     mem_used_mb=num(g.findtext("fb_memory_usage/used", "0")),
                                     mem_total_mb=num(g.findtext("fb_memory_usage/total", "0")),
                                     util=num(g.findtext("utilization/gpu_util", "0")),
                                     procs=len(procs), proc_types=sorted({p.findtext("type", "?") for p in procs})))
            except Exception:
                pass
            _gpu_cache.update(t=time.time(), rows=rows)
        return _gpu_cache["rows"]


def idle_gpus():
    """GPUs with no process of anyone on them (compute or graphics), nearly empty memory and no load."""
    ours = set(running.values())
    return [g["index"] for g in gpu_pool()
            if g["index"] not in ours and g.get("procs", 0) == 0
            and g["mem_used_mb"] < GPU_FREE_MB and g["util"] < GPU_FREE_UTIL]


# ------------------------------------------------------------------ queue + scheduler
def enqueue(job, ip, admin):
    with cond:
        if not admin:
            key, now = visitor_key(ip), time.time()
            history[key] = [t for t in history[key] if now - t < PER_IP_WINDOW_S]
            sid = job.get("session")
            if not sid:
                raise HTTPException(400, "Missing session - please reload the page.")
            if any(j.get("session") == sid and j["status"] in ("queued", "running") for j in jobs.values()):
                raise HTTPException(429, "You already have a job in progress - wait until it finishes, then you can generate another.")
            if PER_IP_TOTAL and len(history[key]) >= PER_IP_TOTAL:
                n = f"{PER_IP_TOTAL} model" + ("s" if PER_IP_TOTAL > 1 else "")
                raise HTTPException(429, f"Each visitor can generate {n} per {PER_IP_WINDOW_S // 3600} h - "
                                         "thanks for trying the studio! You can still explore the gallery.")
            if len(queue) >= QUEUE_MAX:
                raise HTTPException(503, "The queue is full right now - please try again in a few minutes.")
            history[key].append(now)
            with open(VISITORS, "w") as f:
                json.dump(history, f)
            job["visitor"] = key
        job.update(admin=admin, status="queued", created=time.time(), started=None, ended=None, returncode=None, gpu=None)
        jobs[job["id"]] = job
        queue.append(job["id"])
        cond.notify_all()
        return {"id": job["id"], "position": position(job["id"])}


def scheduler():
    """Dispatch queued jobs FIFO onto idle GPUs; while jobs wait for a GPU, re-check every 2 s."""
    while True:
        with cond:
            while not queue:
                cond.wait()
            room = (MAX_PARALLEL - len(running)) if MAX_PARALLEL else len(queue)
        free = idle_gpus() if room > 0 else []
        with cond:
            while queue and free and (not MAX_PARALLEL or len(running) < MAX_PARALLEL):
                job = jobs[queue.popleft()]
                gpu = free.pop(0)
                running[job["id"]] = gpu
                job.update(status="running", started=time.time(), gpu=gpu)
                threading.Thread(target=run_job, args=(job, gpu), daemon=True).start()
            if queue:
                cond.wait(timeout=2)  # woken early when a job finishes


def run_job(job, gpu):
    rc = -1
    try:
        os.makedirs(job["out"], exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
        with open(job["log"], "w") as logf:
            logf.write("$ CUDA_VISIBLE_DEVICES=%s %s\n" % (gpu, " ".join(job["cmd"])))
            logf.flush()
            proc = subprocess.Popen(job["cmd"], cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)
            try:
                rc = proc.wait(timeout=JOB_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = proc.wait()
                logf.write(f"\n[server] killed after {JOB_TIMEOUT_S}s timeout\n")
    finally:
        _gpu_cache["t"] = 0  # next dispatch sees a fresh GPU snapshot
        with cond:
            running.pop(job["id"], None)
            job.update(status="done" if rc == 0 else "failed", returncode=rc, ended=time.time())
            cond.notify_all()
        if job["kind"] == "generate" and not job["admin"]:
            prune_public()


def prune_public():
    if not os.path.isdir(PUBLIC_DIR):
        return
    active = {j["name"] for j in jobs.values() if j["status"] in ("queued", "running")}
    mtime = lambda d: os.path.getmtime(os.path.join(PUBLIC_DIR, d))
    dirs = sorted(os.listdir(PUBLIC_DIR), key=mtime)
    excess = len(dirs) - KEEP_PUBLIC
    for d in dirs:
        if excess <= 0:
            break
        if d not in active and time.time() - mtime(d) >= PRUNE_MIN_AGE_S:
            shutil.rmtree(os.path.join(PUBLIC_DIR, d), ignore_errors=True)
            excess -= 1


def run_summary(out_dir, rel, name):
    s = read_json(os.path.join(out_dir, "summary.json"))
    if s is None:
        return None
    s.pop("config_overrides", None)
    s.update(name=name, rel=rel, has_benchmark=os.path.exists(os.path.join(out_dir, "benchmark.csv")))
    return s


# ------------------------------------------------------------------ API
class GenerateReq(BaseModel):
    prompt: str
    seed: int = 0
    iters: int = 500
    guidance_scale: float = 100.0


@app.post("/api/generate")
def generate(r: GenerateReq, request: Request):
    ip, admin = caller(request)
    prompt = " ".join(r.prompt.split())
    if not prompt:
        raise HTTPException(400, "Empty prompt.")
    if not admin or PUBLIC:
        if len(prompt) > PROMPT_MAX or not all(c.isprintable() for c in prompt):
            raise HTTPException(400, f"Prompt must be at most {PROMPT_MAX} printable characters.")
        low = prompt.lower()
        if any(re.search(r"\b" + re.escape(w) + r"\b", low) for w in BLOCKLIST):
            raise HTTPException(400, "That prompt is not allowed on this public demo.")
    seed = max(0, min(int(r.seed), 2**31 - 1))
    iters, cfg = int(r.iters), float(r.guidance_scale)
    if admin:
        iters, cfg = max(50, min(iters, 2000)), max(1.0, min(cfg, 200.0))
        name = slug(f"{prompt}_s{seed}")
        out, rel = os.path.join(RESULTS, name), f"data/results/{name}"
    else:
        iters, cfg = max(PUBLIC_ITERS[0], min(iters, PUBLIC_ITERS[1])), max(PUBLIC_CFG[0], min(cfg, PUBLIC_CFG[1]))
        name = time.strftime("p%Y%m%d_%H%M%S_") + secrets.token_hex(3)
        out, rel = os.path.join(PUBLIC_DIR, name), f"data/results/public/{name}"
    cmd = [sys.executable, "scripts/train_logged.py", "--prompt", prompt, "--name", name, "--out", out,
           "--seed", str(seed), "--iters", str(iters), "--guidance_scale", str(cfg)]
    if not admin:
        cmd.append("--no_mesh")  # public jobs: Gaussians + views only (mesh export would double the GPU time)
    sid = request.headers.get("x-studio-session", "")
    job = dict(id=secrets.token_urlsafe(9), kind="generate", name=name, prompt=prompt, seed=seed, iters=iters,
               guidance_scale=cfg, out=out, rel=rel, log=os.path.join(out, "training.log"), cmd=cmd,
               session=sid if re.fullmatch(r"[A-Za-z0-9_-]{16,64}", sid) else None)
    return enqueue(job, ip, admin)  # run_job creates `out` when the job starts (rejected requests leave nothing)


class BenchReq(BaseModel):
    name: str
    quick: bool = True


@app.post("/api/benchmark")
def benchmark(r: BenchReq, request: Request):
    ip, admin = caller(request)
    if not admin:
        raise HTTPException(403, "The stress test is run by the owner; measured results are shown below.")
    name = slug(r.name)
    ply = os.path.join(RESULTS, name, f"{name}_model.ply")
    if not os.path.exists(ply):
        raise HTTPException(404, "run has no Gaussian PLY")
    out = os.path.join(RESULTS, name)
    cmd = [sys.executable, "scripts/benchmark_renderer.py", "--ply", ply, "--out", os.path.join(out, "benchmark.csv")] + (["--quick"] if r.quick else [])
    job = dict(id=secrets.token_urlsafe(9), kind="benchmark", name=name, prompt=None, out=out, rel=f"data/results/{name}",
               log=os.path.join(out, "benchmark.log"), cmd=cmd)
    return enqueue(job, ip, admin)


def job_view(job):
    now = time.time()
    d = dict(id=job["id"], kind=job["kind"], name=job["name"], status=job["status"], position=position(job["id"]),
             queued=len(queue), rel=job["rel"], returncode=job["returncode"], gpu=job["gpu"],
             waiting_for_gpu=job["status"] == "queued" and (not idle_gpus() or bool(MAX_PARALLEL and len(running) >= MAX_PARALLEL)),
             elapsed_s=round(now - job["started"], 1) if job["started"] else 0.0,
             wait_s=round((job["started"] or now) - job["created"], 1),
             prompt=job["prompt"], log=tail(job["log"]) if job["status"] != "queued" else [])
    if not job["admin"]:
        d["log"] = scrub(d["log"])
    if job["kind"] == "generate":
        d["progress"] = read_json(os.path.join(job["out"], "progress.json")) if job["status"] != "queued" else None
        if job["status"] == "done":
            d["run"] = run_summary(job["out"], job["rel"], job["name"])
    else:
        d["rows"] = [json.loads(l.split("[bench] ", 1)[1]) for l in tail(job["log"], 200) if l.startswith("[bench] {")]
    return d


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    job = jobs.get(jid)
    if not job:
        raise HTTPException(404, "unknown job (the server may have restarted)")
    return job_view(job)


@app.get("/api/jobs/{jid}/metrics")
def job_metrics(jid: str, since: int = 0):
    job = jobs.get(jid)
    if not job:
        raise HTTPException(404, "unknown job")
    out = []
    try:
        with open(os.path.join(job["out"], "metrics.jsonl")) as f:
            for i, l in enumerate(f):
                if i >= since:
                    out.append(json.loads(l))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return out


@app.get("/api/queue")
def queue_state():
    pool = gpu_pool()
    idle = idle_gpus()
    with cond:
        cap = len(running) + len(idle)  # jobs the machine can run right now (our running jobs + process-free GPUs)
        if MAX_PARALLEL:
            cap = min(cap, MAX_PARALLEL)
        return {"running": len(running), "queued": len(queue), "gpus_total": len(pool), "gpus_idle": len(idle),
                "capacity": cap, "max_parallel": MAX_PARALLEL}


@app.get("/api/job")
def legacy_job(request: Request):
    """Latest job in the old single-user shape (used by scripts/record_demo.py). Admin only."""
    if not caller(request)[1]:
        raise HTTPException(403, "admin only")
    if not jobs:
        return {"active": False}
    job = max(jobs.values(), key=lambda j: j["created"])
    v = job_view(job)
    v.update(active=job["status"] in ("queued", "running"), cmd=" ".join(job["cmd"]))
    return v


@app.get("/api/config")
def config(request: Request):
    _, admin = caller(request)
    return {"public": PUBLIC, "admin": admin, "prompt_max": PROMPT_MAX, "iters": PUBLIC_ITERS, "cfg": PUBLIC_CFG,
            "queue_max": QUEUE_MAX, "per_visitor": PER_IP_TOTAL, "window_h": PER_IP_WINDOW_S // 3600,
            "max_parallel": MAX_PARALLEL}


def list_runs():
    """Curated example runs + every finished visitor run (community gallery), newest first."""
    runs = []
    for base, rel, community in ((RESULTS, "data/results", False), (PUBLIC_DIR, "data/results/public", True)):
        for name in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            if name == "public":
                continue
            s = run_summary(os.path.join(base, name), f"{rel}/{name}", name)
            if s:
                s.update(mtime=os.path.getmtime(os.path.join(base, name, "summary.json")), community=community)
                runs.append(s)
    return sorted(runs, key=lambda d: -d["mtime"])


@app.get("/api/runs")
def runs():
    return list_runs()


@app.get("/api/system")
def system():
    pool = gpu_pool()
    if not pool:
        return JSONResponse({"error": "nvidia-smi unavailable"}, status_code=500)
    return {"host": os.uname().nodename, "name": pool[0]["name"], "gpus_total": len(pool), "gpus_idle": len(idle_gpus()),
            "ours": len(running)}


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "web", "index.html"))


os.makedirs(PUBLIC_DIR, exist_ok=True)
app.mount("/data/results", StaticFiles(directory=RESULTS), name="results")
app.mount("/data/evidence", StaticFiles(directory=EVIDENCE), name="evidence")
app.mount("/", StaticFiles(directory=os.path.join(ROOT, "web")), name="web")
threading.Thread(target=scheduler, daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8549)
    a = p.parse_args()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning", proxy_headers=False)
