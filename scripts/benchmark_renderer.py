"""Stress test for the Gaussian rasterizer used by DreamGaussian (diff-gaussian-rasterization, CUDA).

Loads an optimised Gaussian PLY and renders a fixed 360-degree orbit (forward pass only, the
viewer workload) while sweeping one axis at a time:
  * resolution: 256 .. 4096 px square, native Gaussian count
  * gaussians: each Gaussian cloned k times with jitter of ~1 std-dev (more overdraw/sorting work),
               at 512 px, until the GPU runs out of memory or the count cap is reached
Method: 10 warm-up frames (excluded), then `--frames` frames timed with torch.cuda.synchronize();
peak VRAM = torch.cuda.max_memory_allocated() after reset per setting. Same camera path and GPU for
every row. Writes CSV with columns setting,resolution,gaussians,frames,seconds,fps,peak_vram_mb,notes.

Usage:
  CUDA_VISIBLE_DEVICES=4 python scripts/benchmark_renderer.py --ply results/x/x_model.ply --out evidence/performance.csv
"""
import argparse
import csv
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "third_party", "dreamgaussian"))

import numpy as np
import torch

from cam_utils import orbit_camera
from gs_renderer import Renderer, MiniCam

FOVY = math.radians(49.1)
RADIUS = 2.5


def cams(res, n):
    for k in range(n):
        pose = orbit_camera(-15, 360.0 * k / n, RADIUS)
        yield MiniCam(pose, res, res, FOVY, FOVY, 0.01, 100)


def replicate(renderer, base, k, seed=0):
    """Replace the model's tensors with k jittered copies of the base Gaussians."""
    g = renderer.gaussians
    gen = torch.Generator(device="cuda").manual_seed(seed)
    xyz = base["xyz"].repeat(k, 1)
    if k > 1:
        std = torch.exp(base["scaling"]).repeat(k, 1)
        xyz = xyz + torch.randn(xyz.shape, device="cuda", generator=gen) * std
    g._xyz = xyz
    for name in ("features_dc", "features_rest", "scaling", "rotation", "opacity"):
        setattr(g, "_" + name, base[name].repeat(k, *([1] * (base[name].dim() - 1))))


@torch.no_grad()
def measure(renderer, res, frames, warmup=10):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    bg = torch.ones(3, device="cuda")
    for cam in cams(res, warmup):
        renderer.render(cam, bg_color=bg)
    cam_list = list(cams(res, frames))
    torch.cuda.synchronize()
    t = time.perf_counter()
    for cam in cam_list:
        renderer.render(cam, bg_color=bg)
    torch.cuda.synchronize()
    sec = time.perf_counter() - t
    return sec, torch.cuda.max_memory_allocated() / 2**20


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ply", required=True)
    p.add_argument("--out", default=os.path.join(ROOT, "evidence", "performance.csv"))
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--resolutions", default="256,512,1024,2048,4096")
    p.add_argument("--multipliers", default="1,4,16,64,128,256")
    p.add_argument("--max_gaussians", type=int, default=120_000_000)
    p.add_argument("--quick", action="store_true", help="fewer frames/settings (used by the web studio)")
    a = p.parse_args()
    if a.quick:
        a.frames, a.resolutions, a.multipliers = 60, "512,1024,2048", "1,16,64"

    gpu = torch.cuda.get_device_name(0)
    renderer = Renderer(sh_degree=0)
    renderer.initialize(a.ply)
    g = renderer.gaussians
    base = {n: getattr(g, "_" + n).detach().clone() for n in
            ("xyz", "features_dc", "features_rest", "scaling", "rotation", "opacity")}
    n0 = base["xyz"].shape[0]
    print(f"[bench] gpu={gpu} ply={a.ply} base_gaussians={n0}", flush=True)

    # global warm-up (CUDA context, rasterizer kernels, allocator) so the first measured row is not penalised
    measure(renderer, 512, 60)
    rows = []

    def emit(setting, res, n, frames, sec, vram, notes=""):
        fps = frames / sec if sec else 0.0
        row = dict(setting=setting, resolution=res, gaussians=n, frames=frames, seconds=round(sec, 4),
                   fps=round(fps, 1), peak_vram_mb=round(vram, 1), notes=notes)
        rows.append(row)
        print("[bench] " + json.dumps(row), flush=True)

    for res in map(int, a.resolutions.split(",")):
        try:
            sec, vram = measure(renderer, res, a.frames)
            emit("resolution", res, n0, a.frames, sec, vram, f"{gpu}; forward-only; 360deg orbit")
        except torch.OutOfMemoryError:
            emit("resolution", res, n0, 0, 0, torch.cuda.max_memory_allocated() / 2**20, "CUDA OOM")
            break

    for k in map(int, a.multipliers.split(",")):
        n = n0 * k
        if n > a.max_gaussians:
            emit("gaussians", 512, n, 0, 0, 0, f"skipped: above cap {a.max_gaussians}")
            break
        try:
            replicate(renderer, base, k)
            sec, vram = measure(renderer, 512, a.frames)
            emit("gaussians", 512, n, a.frames, sec, vram, f"{gpu}; base x{k} jittered clones")
        except torch.OutOfMemoryError:
            emit("gaussians", 512, n, 0, 0, torch.cuda.max_memory_allocated() / 2**20, "CUDA OOM (failure point)")
            break
        finally:
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[bench] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
