"""Build the static web demo (GitHub Pages, served from docs/): the same studio page in read-only mode, with the
real outputs of finished GPU runs (Gaussian PLY, snapshots, metrics, benchmark CSVs). Generation itself needs the
local GPU server; the static page says so.

usage: python scripts/export_static.py [run names...] [--with-community]
       (default: every results/<name>/summary.json; --with-community also adds visitors' models from results/public/)
"""
import csv, json, os, shutil, subprocess, sys

import numpy as np
from plyfile import PlyData, PlyElement

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "docs")
KEEP = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def compact_ply(src, dst):
    v = PlyData.read(src)["vertex"]
    arr = np.empty(v.count, dtype=[(k, "f4") for k in KEEP])
    for k in KEEP:
        arr[k] = v[k]
    PlyData([PlyElement.describe(arr, "vertex")], byte_order="<").write(dst)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    with_community = "--with-community" in sys.argv  # also bake visitors' models (results/public/) into the static demo
    names = args or sorted(n for n in os.listdir(RES) if os.path.exists(os.path.join(RES, n, "summary.json")))
    items = [(n, n, False) for n in names]
    pub = os.path.join(RES, "public")
    if with_community and os.path.isdir(pub):
        items += [(f"public/{n}", n, True) for n in sorted(os.listdir(pub)) if os.path.exists(os.path.join(pub, n, "summary.json"))]
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    shutil.copytree(os.path.join(ROOT, "web"), OUT)
    runs = []
    for relname, n, community in items:
        src, dst = os.path.join(RES, relname), os.path.join(OUT, "data", "results", relname)
        os.makedirs(os.path.join(dst, "progress"))
        s = json.load(open(os.path.join(src, "summary.json")))
        s.pop("config_overrides", None)
        compact_ply(os.path.join(src, s["ply"]), os.path.join(dst, s["ply"]))
        for f in ["metrics.jsonl", "summary.json", "turntable.mp4", "benchmark.csv", "config_resolved.yaml"]:
            if os.path.exists(os.path.join(src, f)):
                shutil.copy(os.path.join(src, f), dst)
        shutil.copytree(os.path.join(src, "views"), os.path.join(dst, "views"))
        for f in os.listdir(os.path.join(src, "progress")):
            if f != "latest.png":
                shutil.copy(os.path.join(src, "progress", f), os.path.join(dst, "progress"))
        s.update(name=n, rel=f"data/results/{relname}", community=community)
        s["has_benchmark"] = os.path.exists(os.path.join(src, "benchmark.csv"))
        s["mtime"] = os.path.getmtime(os.path.join(src, "summary.json"))
        runs.append(s)
    runs.sort(key=lambda r: -r["mtime"])
    json.dump(runs, open(os.path.join(OUT, "data", "runs.json"), "w"), indent=1)
    os.makedirs(os.path.join(OUT, "data", "evidence"))
    shutil.copy(os.path.join(ROOT, "evidence", "performance.csv"), os.path.join(OUT, "data", "evidence"))
    gpu = subprocess.run(["nvidia-smi", "-i", os.environ.get("GPU", "4"), "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip() or "NVIDIA GPU"
    json.dump({"name": gpu}, open(os.path.join(OUT, "data", "system.json"), "w"))
    open(os.path.join(OUT, ".nojekyll"), "w").close()
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(OUT) for f in fs) / 2**20
    print(f"static site -> {OUT}  ({len(runs)} runs, {size:.1f} MB)")


if __name__ == "__main__":
    main()
