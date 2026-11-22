#!/usr/bin/env bash
# Build fused-ssim's CUDA extension for CUDA 13. nvcc lives in the build prefix; the cuda-nvcc activation
# (CONDA_BUILD=1) adds the host prefix's CUDA headers and libraries to CFLAGS/LDFLAGS and NVCC_APPEND_FLAGS.
set -euxo pipefail
rm -rf "$SP_DIR"/fused_ssim "$SP_DIR"/fused_ssim-*.dist-info "$SP_DIR"/fused_ssim-*.egg-info "$SP_DIR"/fused_ssim_cuda*.so
export CUDA_HOME="$BUILD_PREFIX"
# No GPU probe at build time: setup.py then compiles the CUDA backend for the architectures listed here
# (RTX 5090 sm_120, DGX Spark GB10 sm_121) instead of the device it would detect.
export CUDA_VISIBLE_DEVICES=""
export CUDA_ARCHITECTURES="120;121"
export TORCH_CUDA_ARCH_LIST="12.0;12.1"
export MAX_JOBS="${MAX_JOBS:-4}"
python -m pip install . --no-deps --no-build-isolation -vv
ls "$SP_DIR"/fused_ssim_cuda*.so
