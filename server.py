"""Interactive Diffusion-Based 3D Generation Studio - local GPU server.

Serves the web studio (web/) and runs real DreamGaussian jobs on the local NVIDIA GPU:
  POST /api/generate   -> launches scripts/train_logged.py (text -> SDS-optimised 3D Gaussians)
  GET  /api/job        -> live progress.json + training-log tail of the current/last job
  POST /api/benchmark  -> launches scripts/benchmark_renderer.py (CUDA rasterizer stress test)
  GET  /api/runs       -> finished runs (summary.json of every results/<name>/)
  GET  /api/system     -> nvidia-smi snapshot of the GPU used for jobs
Static: /            -> web/        /data/results -> results/      /data/evidence -> evidence/

Usage:  GPU=4 python server.py --port 8549
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
EVIDENCE = os.path.join(ROOT, "evidence")
GPU = os.environ.get("GPU", "4")

app = FastAPI(title="Diffusion 3D Studio")
lock = threading.Lock()
job = {"kind": None, "name": None, "proc": None, "log": None, "started": None, "cmd": None}


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40] or "run"


def launch(kind, name, cmd, log_path):
    with lock:
        if job["proc"] is not None and job["proc"].poll() is None:
            raise HTTPException(409, f"GPU busy with {job['kind']} job '{job['name']}'")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, PYTHONUNBUFFERED="1")
        logf = open(log_path, "w")
        logf.write("$ CUDA_VISIBLE_DEVICES=%s %s\n" % (GPU, " ".join(cmd)))
        logf.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)
        job.update(kind=kind, name=name, proc=proc, log=log_path, started=time.time(), cmd=cmd)
    return {"ok": True, "kind": kind, "name": name, "pid": proc.pid}


def tail(path, n=40):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 64_000))
            lines = f.read().decode("utf-8", "replace").replace("\r", "\n").splitlines()
        return [l for l in lines if l.strip()][-n:]
    except FileNotFoundError:
        return []


class GenerateReq(BaseModel):
    prompt: str
    seed: int = 0
    iters: int = 500
    guidance_scale: float = 100.0
    name: str | None = None


@app.post("/api/generate")
def generate(r: GenerateReq):
    if not r.prompt.strip():
        raise HTTPException(400, "empty prompt")
    name = slug(r.name or f"{r.prompt}_s{r.seed}")
    out = os.path.join(RESULTS, name)
    cmd = [sys.executable, "scripts/train_logged.py", "--prompt", r.prompt, "--name", name,
           "--seed", str(r.seed), "--iters", str(max(50, min(r.iters, 2000))),
           "--guidance_scale", str(r.guidance_scale)]
    return launch("generate", name, cmd, os.path.join(out, "training.log"))


class BenchReq(BaseModel):
    name: str
    quick: bool = True


@app.post("/api/benchmark")
def benchmark(r: BenchReq):
    ply = os.path.join(RESULTS, r.name, f"{r.name}_model.ply")
    if not os.path.exists(ply):
        raise HTTPException(404, "run has no Gaussian PLY")
    out = os.path.join(RESULTS, r.name, "benchmark.csv")
    cmd = [sys.executable, "scripts/benchmark_renderer.py", "--ply", ply, "--out", out] + (["--quick"] if r.quick else [])
    return launch("benchmark", r.name, cmd, os.path.join(RESULTS, r.name, "benchmark.log"))


@app.get("/api/job")
def get_job():
    if job["proc"] is None:
        return {"active": False}
    rc = job["proc"].poll()
    d = {"active": rc is None, "returncode": rc, "kind": job["kind"], "name": job["name"],
         "elapsed_s": round(time.time() - job["started"], 1), "cmd": " ".join(job["cmd"]),
         "log": tail(job["log"])}
    if job["kind"] == "generate":
        try:
            with open(os.path.join(RESULTS, job["name"], "progress.json")) as f:
                d["progress"] = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            d["progress"] = None
    else:
        d["rows"] = [json.loads(l.split("[bench] ", 1)[1]) for l in tail(job["log"], 200) if l.startswith("[bench] {")]
    return d


@app.get("/api/metrics/{name}")
def metrics(name: str, since: int = 0):
    path = os.path.join(RESULTS, slug(name), "metrics.jsonl")
    out = []
    try:
        with open(path) as f:
            for i, l in enumerate(f):
                if i >= since:
                    out.append(json.loads(l))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return out


def list_runs():
    runs = []
    for name in sorted(os.listdir(RESULTS)) if os.path.isdir(RESULTS) else []:
        s = os.path.join(RESULTS, name, "summary.json")
        if os.path.exists(s):
            with open(s) as f:
                d = json.load(f)
            d["name"] = name
            d["mtime"] = os.path.getmtime(s)
            d["has_benchmark"] = os.path.exists(os.path.join(RESULTS, name, "benchmark.csv"))
            runs.append(d)
    return sorted(runs, key=lambda d: -d["mtime"])


@app.get("/api/runs")
def runs():
    return list_runs()


@app.get("/api/benchmark/{name}")
def bench_rows(name: str):
    path = os.path.join(RESULTS, slug(name), "benchmark.csv")
    if not os.path.exists(path):
        path = os.path.join(EVIDENCE, "performance.csv")
    with open(path) as f:
        return list(csv.DictReader(f))


@app.get("/api/system")
def system():
    q = "name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,driver_version"
    try:
        out = subprocess.run(["nvidia-smi", "-i", GPU, f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        name, used, total, util, temp, power, driver = [x.strip() for x in out.split(",")]
        return {"gpu_index": GPU, "name": name, "mem_used_mb": int(used), "mem_total_mb": int(total),
                "util": int(util), "temp_c": int(temp), "power_w": float(power), "driver": driver,
                "host": os.uname().nodename}
    except Exception as e:
        return JSONResponse({"error": repr(e)}, status_code=500)


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "web", "index.html"))


os.makedirs(RESULTS, exist_ok=True)
app.mount("/data/results", StaticFiles(directory=RESULTS), name="results")
app.mount("/data/evidence", StaticFiles(directory=EVIDENCE), name="evidence")
app.mount("/", StaticFiles(directory=os.path.join(ROOT, "web")), name="web")

if __name__ == "__main__":
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8549)
    a = p.parse_args()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
