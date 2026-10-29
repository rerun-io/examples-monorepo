#!/usr/bin/env bash
# Build one upstream PyTorch CUDA extension (directory $EXT_DIR) for CUDA 13.
# nvcc lives in the build prefix; the cuda-nvcc activation (CONDA_BUILD=1) adds the
# host prefix's CUDA headers and libraries to CFLAGS/LDFLAGS and NVCC_APPEND_FLAGS.
set -euxo pipefail
export CUDA_HOME="$BUILD_PREFIX"
# Ampere (the paper's A5000 and the 3060), Blackwell RTX 5090 and GB10 (Spark), plus PTX.
export TORCH_CUDA_ARCH_LIST="8.6;12.0;12.1+PTX"
export MAX_JOBS="${MAX_JOBS:-8}"
cd "$EXT_DIR"
python -m pip install . --no-deps --no-build-isolation -vv
