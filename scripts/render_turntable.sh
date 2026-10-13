#!/usr/bin/env bash
# Stress test / novel-view rendering of a saved Gaussian PLY with the CUDA rasterizer
# usage: GPU=4 bash scripts/render_turntable.sh results/mushroom_house/mushroom_house_model.ply
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLY="${1:?path to *_model.ply}"
cd "$ROOT"
CUDA_VISIBLE_DEVICES="${GPU:-0}" python scripts/benchmark_renderer.py --ply "$PLY" --out "$ROOT/evidence/performance.csv" 2>&1 | tee "$ROOT/evidence/benchmark.log"
