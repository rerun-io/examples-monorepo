#!/usr/bin/env bash
# Build NVIDIA ViPE for MetricTrail (CUDA 13, conda-forge torch 2.12):
#   1. vipe + vipe_ext (ViPE's own extension, without lietorch_ext, patch 0001),
#   2. MetricTrail's DPVO kernels cuda_corr + cuda_ba into vipe/metrictrail/dpvo_ext (no lietorch_backends, patch 0001).
# DINOv3 comes from dinov3-src (patch 0002) and MoGe-2 from monopriors (patch 0003).
# nvcc lives in the build prefix; the cuda-nvcc activation adds the host prefix's CUDA headers and libraries.
set -euxo pipefail

# pixi-build reuses the host prefix between rebuilds: clear the target first.
rm -rf "$SP_DIR/vipe" "$SP_DIR"/vipe_ext*.so "$SP_DIR"/nvidia_vipe-*.dist-info

# Vendored lietorch: Python groups + C++ kernels of all three copies. The patched __init__.py files re-export
# the external lietorch-mojo package instead (patch 0001).
rm -rf csrc/lietorch_ext vipe/metrictrail/csrc/dpvo/lietorch vipe/metrictrail/csrc/dpvo_stereo/lietorch
for copy in vipe/ext/lietorch vipe/metrictrail/dpvo/lietorch vipe/metrictrail/stereo/dpvo_stereo/lietorch; do
    rm -f "$copy/groups.py" "$copy/group_ops.py" "$copy/broadcasting.py" "$copy/groups.pyi"
done
# ViPE's vendored MoGe-2: patch 0003 loads monopriors' MoGe v2 instead; nothing else imports this tree.
rm -rf vipe/priors/depth/moge2
# Upstream ships Depth Anything 3's DINOv2 without __init__.py, so find_packages() drops it and pipeline=dav3 fails.
touch vipe/priors/depth/dav3/model/dinov2/__init__.py
export CUDA_HOME="$BUILD_PREFIX"
# Ampere (the 3060) and Blackwell RTX 5090, plus compute_120 PTX for newer GPUs (the GB10 JITs it).
export TORCH_CUDA_ARCH_LIST="8.6;12.0+PTX"
export MAX_JOBS="${MAX_JOBS:-16}"
# USE_SYSTEM_EIGEN=1 (conda-forge eigen ==3.4.0, no network): vipe/ext/specs.py skips its Eigen download, and
# slam_ext's <eigen3/Eigen/...> resolves through torch's -I$PREFIX/include; vipe/metrictrail/build.py takes
# $CONDA_PREFIX/include/eigen3 (CONDA_PREFIX=$PREFIX on its command line below) and checks it is 3.4.0.
export USE_SYSTEM_EIGEN=1
# ViPE's setup.py uses $CONDA_PREFIX/bin/nvcc when it exists; the host prefix's nvcc has no cicc. Keep the
# build prefix's nvcc (CUDA_HOME above).
unset CONDA_PREFIX

t0=$(date +%s)
python -m pip install . --no-deps --no-build-isolation -vv
t1=$(date +%s)
# DPVO's recipe: Eigen 3.4.0 headers, -O3; build.py refuses any other Eigen version.
CONDA_PREFIX="$PREFIX" python vipe/metrictrail/build.py --build-lib "$SP_DIR" --verbose
t2=$(date +%s)
echo "VIPE_SRC_TIMING vipe_ext_pip_install_s=$((t1 - t0)) dpvo_kernels_s=$((t2 - t1))"

test -f "$SP_DIR"/vipe/metrictrail/dpvo_ext/cuda_corr*.so
test -f "$SP_DIR"/vipe/metrictrail/dpvo_ext/cuda_ba*.so
test ! -e "$SP_DIR"/vipe/metrictrail/dpvo_ext/lietorch_backends*.so
ls "$SP_DIR"/vipe_ext*.so
test ! -e "$SP_DIR"/vipe/priors/depth/moge2
test -f "$SP_DIR"/vipe/priors/depth/dav3/model/dinov2/__init__.py
