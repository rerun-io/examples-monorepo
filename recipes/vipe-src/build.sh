#!/usr/bin/env bash
# Build NVIDIA ViPE for MetricTrail (CUDA 13, conda-forge torch 2.12): vipe + vipe_ext (ViPE's own extension,
# without lietorch_ext, patch 0001). MetricTrail's DPVO is packages/dpvo's dpvo.metric (supplied by the consuming
# feature, kernels included): the vendored DPVO, its kernels and the stereo tree are deleted and vipe/metrictrail/dpvo
# becomes the re-export shims in $RECIPE_DIR/dpvo_shim. DINOv3 comes from dinov3-src (patch 0002) and MoGe-2 from
# monopriors (patch 0003).
# nvcc lives in the build prefix; the cuda-nvcc activation adds the host prefix's CUDA headers and libraries.
set -euxo pipefail

# pixi-build reuses the host prefix between rebuilds: clear the target first.
rm -rf "$SP_DIR/vipe" "$SP_DIR"/vipe_ext*.so "$SP_DIR"/nvidia_vipe-*.dist-info

# ViPE's vendored lietorch: Python groups + C++ kernels. The patched __init__.py re-exports the external
# lietorch-mojo package instead (patch 0001).
rm -rf csrc/lietorch_ext
rm -f vipe/ext/lietorch/groups.py vipe/ext/lietorch/group_ops.py vipe/ext/lietorch/broadcasting.py vipe/ext/lietorch/groups.pyi
# MetricTrail's vendored DPVO (Python, CUDA sources, extension package and loader, torch_scatter stand-in, build
# script) and the optional stereo runner with its own DPVO copy. Nothing outside these trees imports them except
# slam.py and dino_fusion.py, which import vipe.metrictrail.dpvo.{dpvo,config,net,altcorr,projective_ops,lietorch}:
# the shims below keep those paths.
rm -rf vipe/metrictrail/dpvo vipe/metrictrail/csrc vipe/metrictrail/dpvo_ext vipe/metrictrail/dpvo_stereo_ext \
    vipe/metrictrail/stereo
rm -f vipe/metrictrail/dpvo_ext_loader.py vipe/metrictrail/dpvo_scatter.py vipe/metrictrail/build.py
cp -r "$RECIPE_DIR/dpvo_shim" vipe/metrictrail/dpvo
find vipe/metrictrail/dpvo -name __pycache__ -prune -exec rm -rf {} +
# ViPE's vendored MoGe-2: patch 0003 loads monopriors' MoGe v2 instead; nothing else imports this tree.
rm -rf vipe/priors/depth/moge2
# Upstream ships Depth Anything 3's DINOv2 without __init__.py, so find_packages() drops it and pipeline=dav3 fails.
touch vipe/priors/depth/dav3/model/dinov2/__init__.py

if [[ "$(uname)" == Darwin ]]; then
    # macOS: no CUDA. setup.py builds no vipe_ext (patch 0007); vipe_mojo carries the kernels.
    USE_SYSTEM_EIGEN=1 python -m pip install . --no-deps --no-build-isolation -vv
    test ! -e "$SP_DIR"/vipe_ext*.so
    test ! -e "$SP_DIR"/vipe/priors/depth/moge2
    exit 0
fi

export CUDA_HOME="$BUILD_PREFIX"
# Ampere (the 3060) and Blackwell RTX 5090, plus compute_120 PTX for newer GPUs (the GB10 JITs it).
export TORCH_CUDA_ARCH_LIST="8.6;12.0+PTX"
export MAX_JOBS="${MAX_JOBS:-16}"
# USE_SYSTEM_EIGEN=1 (conda-forge eigen ==3.4.0, no network): vipe/ext/specs.py skips its Eigen download, and
# slam_ext's <eigen3/Eigen/...> resolves through torch's -I$PREFIX/include.
export USE_SYSTEM_EIGEN=1
# ViPE's setup.py uses $CONDA_PREFIX/bin/nvcc when it exists; the host prefix's nvcc has no cicc. Keep the
# build prefix's nvcc (CUDA_HOME above).
unset CONDA_PREFIX

t0=$(date +%s)
python -m pip install . --no-deps --no-build-isolation -vv
t1=$(date +%s)
echo "VIPE_SRC_TIMING vipe_ext_pip_install_s=$((t1 - t0))"

ls "$SP_DIR"/vipe_ext*.so
test ! -e "$SP_DIR"/vipe/priors/depth/moge2
test -f "$SP_DIR"/vipe/priors/depth/dav3/model/dinov2/__init__.py
# The DPVO shims are installed and nothing of the vendored DPVO, its kernels or the stereo tree is.
for shim in __init__.py dpvo.py config.py net.py projective_ops.py altcorr/__init__.py lietorch/__init__.py; do
    test -f "$SP_DIR/vipe/metrictrail/dpvo/$shim"
done
test "$(cd "$SP_DIR/vipe/metrictrail/dpvo" && find . -name '*.py' | LC_ALL=C sort | tr '\n' ' ')" = \
    "./__init__.py ./altcorr/__init__.py ./config.py ./dpvo.py ./lietorch/__init__.py ./net.py ./projective_ops.py "
for gone in csrc dpvo_ext dpvo_stereo_ext stereo dpvo_ext_loader.py dpvo_scatter.py build.py; do
    test ! -e "$SP_DIR/vipe/metrictrail/$gone"
done
