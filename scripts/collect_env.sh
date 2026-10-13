#!/usr/bin/env bash
# Records the hardware/software environment to evidence/environment.txt
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPU="${GPU:-4}"
{
  echo "# collected $(date -Is) on $(hostname)"
  echo "## OS";      uname -a; grep PRETTY /etc/os-release
  echo "## GPU (CUDA_VISIBLE_DEVICES=$GPU used for all jobs)"
  nvidia-smi -i "$GPU" --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv
  echo "## nvcc";    "${CUDA_HOME:-/usr/local/cuda-12.5}/bin/nvcc" --version | tail -2
  echo "## gcc";     gcc --version | head -1
  echo "## python";  python --version; which python
  echo "## upstream DreamGaussian"; git -C "$ROOT/third_party/dreamgaussian" rev-parse HEAD; git -C "$ROOT/third_party/dreamgaussian" status --short
  echo "## diff-gaussian-rasterization"; git -C "$ROOT/third_party/dreamgaussian/diff-gaussian-rasterization" rev-parse HEAD
  echo "## torch / libs"
  CUDA_VISIBLE_DEVICES=$GPU python - <<'PY'
import torch, diffusers, transformers, importlib
print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
print("cuda available", torch.cuda.is_available(), "device", torch.cuda.get_device_name(0))
print("diffusers", diffusers.__version__, "transformers", transformers.__version__)
for m in ["diff_gaussian_rasterization", "simple_knn", "nvdiffrast", "kiui"]:
    mod = importlib.import_module(m); print(m, "OK", getattr(mod, "__file__", ""))
PY
  echo "## disk"; df -h "$ROOT" | tail -1
} > "$ROOT/evidence/environment.txt" 2>&1
cat "$ROOT/evidence/environment.txt"
