#!/usr/bin/env bash
# Build SageAttention's CUDA extensions for CUDA 13. nvcc lives in the build prefix; the cuda-nvcc activation
# (CONDA_BUILD=1) adds the host prefix's CUDA headers and libraries to CFLAGS/LDFLAGS and NVCC_APPEND_FLAGS.
set -euxo pipefail
rm -rf "$SP_DIR"/sageattention "$SP_DIR"/sageattention-*.dist-info "$SP_DIR"/_qattn_* "$SP_DIR"/_fused*
export CUDA_HOME="$BUILD_PREFIX"
# RTX 5090 (sm_120) and DGX Spark GB10 (sm_121); setup.py reads this instead of probing a GPU.
export TORCH_CUDA_ARCH_LIST="12.0;12.1"
export MAX_JOBS="${MAX_JOBS:-8}"
python -m pip install . --no-deps --no-build-isolation -vv
python -c "import importlib.util as u; assert u.find_spec('sageattention')"
