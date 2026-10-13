#!/usr/bin/env bash
# Creates an isolated conda env for upstream DreamGaussian on an Ampere GPU (A100, sm_80).
# Tested: Ubuntu 24.04, driver 560.28, system CUDA toolkit 12.5 (nvcc), gcc 13.3.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${ENV_NAME:-dreamgaussian}"
DG="$ROOT/third_party/dreamgaussian"
DG_COMMIT=69a3c7065d580da124dc0bf2e9a21cd3b7973b35

if [ ! -d "$DG" ]; then
  git clone https://github.com/dreamgaussian/dreamgaussian.git "$DG"
fi
git -C "$DG" checkout "$DG_COMMIT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -q "^$ENV_NAME " || conda create -y -n "$ENV_NAME" python=3.10
conda activate "$ENV_NAME"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.5}"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
# this machine's CUDA 12.5 keeps runtime headers only under targets/ (include/ holds cuDNN); host-compiled .cpp files need them
export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"

pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r "$ROOT/configs/requirements_pinned.txt"

# Custom CUDA extensions (per upstream README)
[ -d "$DG/diff-gaussian-rasterization" ] || git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization "$DG/diff-gaussian-rasterization"
pip install --no-build-isolation "$DG/diff-gaussian-rasterization"
# simple-knn uses FLT_MAX without including <cfloat>; required with gcc 13 / CUDA 12.5 (patch is idempotent)
grep -q "#include <cfloat>" "$DG/simple-knn/simple_knn.cu" || sed -i '1i #include <cfloat>' "$DG/simple-knn/simple_knn.cu"
pip install --no-build-isolation "$DG/simple-knn"
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast/
pip install git+https://github.com/ashawkey/kiuikit
python -c "import torch, diff_gaussian_rasterization, simple_knn; print('ok', torch.__version__, torch.cuda.is_available())"
