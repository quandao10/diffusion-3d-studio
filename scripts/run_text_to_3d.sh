#!/usr/bin/env bash
# Reproducible text-to-3D run (DreamGaussian stage 1: SDS-optimised 3D Gaussians + mesh export, novel views, turntable)
# usage: GPU=4 bash scripts/run_text_to_3d.sh "a small red mushroom house" mushroom_house [seed]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROMPT="${1:-a small red mushroom house}"; NAME="${2:-mushroom_house}"; SEED="${3:-0}"
mkdir -p "$ROOT/results/$NAME"
cd "$ROOT"
CUDA_VISIBLE_DEVICES="${GPU:-0}" /usr/bin/time -v python scripts/train_logged.py \
  --prompt "$PROMPT" --name "$NAME" --seed "$SEED" --iters 500 --guidance_scale 100 \
  2>&1 | tee "$ROOT/results/$NAME/training.log"
