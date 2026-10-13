# Setup notes (homer.cs.rutgers.edu, 2026-10-13)

Machine: 8× A100-PCIE-40GB, driver 560.28.03, system CUDA toolkit 12.5 (`/usr/local/cuda-12.5`), gcc 13.3, Ubuntu 24.04.
GPUs 0–3 were fully used by another user's job; **all DreamGaussian jobs and benchmarks ran on GPU 4** (`CUDA_VISIBLE_DEVICES=4`).
No system-level packages were changed; everything lives in the conda env `dreamgaussian` (and `tts` for the voiceover).

| # | Problem (log) | Fix |
|---|---|---|
| 1 | `simple_knn.cu(90): error: identifier "FLT_MAX" is undefined` (`setup_install.log`) | add `#include <cfloat>` (idempotent `sed` in `scripts/setup_env.sh`); only change inside upstream |
| 2 | nvdiffrast: `fatal error: cuda_runtime.h: No such file or directory` (`setup_install_2.log`) | this machine's `cuda-12.5/include` only holds cuDNN headers; runtime headers are in `targets/x86_64-linux/include` → `export CPATH=...` in `setup_env.sh` |
| 3 | `stabilityai/stable-diffusion-2-1-base` returns HTTP 401 (repo no longer downloadable) | pass `hf_key="Manojb/stable-diffusion-2-1-base"` (mirror of the same SD 2.1-base diffusers weights, already in the local HF cache) to upstream `StableDiffusion`; override with `DG_SD_KEY` |
| 4 | First run reported 37 GB peak VRAM (`debug/aborted_run1_cudnn_benchmark_37GB.log`) | my wrapper had called upstream's GUI `seed_everything()`, which also sets `cudnn.benchmark=True` (autotuning workspaces). Upstream's headless path never does that; now only RNGs are seeded → 4.4 GB peak |
| 5 | Headless Chrome: WebGL context lost with ANGLE/Vulkan (`CONTEXT_LOST_WEBGL`), blank viewer | use ANGLE GL-EGL (`--use-angle=gl-egl`) → WebGL2 on the A100 with timer queries (`scripts/browser.py`) |
| 6 | Chrome's EGL picks GPU 3 (busy with another user's job); `CUDA_VISIBLE_DEVICES`, `DRI_PRIME` did not move it | **unresolved** — browser-viewer FPS is reported as indicative only; the CUDA rasterizer stress test runs on dedicated GPU 4 |
| 7 | Kokoro TTS: latest `transformers` rejects torch 2.4 ("PyTorch was not found") | pin `transformers==4.46.3` in the separate `tts` env |
| 8 | Benchmark v1: 256² slower than 512² (`debug/performance_v1_no_global_warmup.csv`) | added a global warm-up pass; the ordering persisted (314 vs 368 FPS) so it is reported as measured, not as an artefact |
| 9 | Live studio charts stayed empty during training (first video take, `video/take1_too_long.mp4`) | `metrics.jsonl` was block-buffered; now flushed every step |

Versions that work: Python 3.10.22, torch 2.4.1+cu124 (cuDNN 9.1), diffusers 0.27.2, transformers 4.40.2,
diff-gaussian-rasterization @ 8829d14 (ashawkey fork, depth+alpha), nvdiffrast 0.4.0, upstream DreamGaussian @ 69a3c70.
See `environment.txt` for the full capture.
