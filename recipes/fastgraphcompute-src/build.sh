#!/usr/bin/env bash
# Build FastGraphCompute's C++/CUDA extensions for CUDA 13. nvcc lives in the build prefix; the cuda-nvcc activation
# (CONDA_BUILD=1) adds the host prefix's CUDA headers and libraries to CFLAGS/LDFLAGS and NVCC_APPEND_FLAGS.
set -euxo pipefail
rm -rf "$SP_DIR"/fastgraphcompute "$SP_DIR"/fastgraphcompute-*.dist-info "$SP_DIR"/fastgraphcompute-*.egg-info
export CUDA_HOME="$BUILD_PREFIX"
# No GPU probe at build time: setup.py then builds the CUDA extensions only when FORCE_CUDA is set.
export CUDA_VISIBLE_DEVICES=""
export FORCE_CUDA=1
# RTX 5090 (sm_120) and DGX Spark GB10 (sm_121); setup.py reads this instead of probing a GPU.
export TORCH_CUDA_ARCH_LIST="12.0;12.1"
export MAX_JOBS="${MAX_JOBS:-4}"
export NUM_PARALLEL_JOBS="$MAX_JOBS"
python -m pip install . --no-deps --no-build-isolation -vv
test -f "$SP_DIR"/fastgraphcompute/extensions/binned_knn_ops.so
