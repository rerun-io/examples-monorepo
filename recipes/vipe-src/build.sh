#!/usr/bin/env bash
# Build NVIDIA ViPE for MetricTrail, pure Python. ViPE's CUDA extension vipe_ext is not built: `vipe.ext` imports
# packages/vipe-mojo's `vipe_mojo` (patch 0007) and its sources (csrc, vipe/ext/specs.py) are deleted. MetricTrail's
# DPVO is packages/dpvo's dpvo.metric (supplied by the consuming feature, kernels included): the vendored DPVO, its
# kernels and the stereo tree are deleted and vipe/metrictrail/dpvo becomes the re-export shims in $RECIPE_DIR/dpvo_shim.
# DINOv3 comes from dinov3-src (patch 0002) and MoGe-2 from monopriors (patch 0003).
set -euxo pipefail

# pixi-build reuses the host prefix between rebuilds: clear the target first.
rm -rf "$SP_DIR/vipe" "$SP_DIR"/vipe_ext*.so "$SP_DIR"/nvidia_vipe-*.dist-info

# vipe_ext's sources (lietorch_ext included) and build helper: patch 0007 imports vipe_mojo instead.
rm -rf csrc
rm -f vipe/ext/specs.py
# ViPE's vendored lietorch groups: the patched __init__.py re-exports the external lietorch-mojo package (patch 0001).
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

t0=$(date +%s)
python -m pip install . --no-deps --no-build-isolation -vv
t1=$(date +%s)
echo "VIPE_SRC_TIMING pip_install_s=$((t1 - t0))"

test -z "$(ls "$SP_DIR"/vipe_ext* 2>/dev/null)"
test ! -e "$SP_DIR"/vipe/ext/specs.py
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
