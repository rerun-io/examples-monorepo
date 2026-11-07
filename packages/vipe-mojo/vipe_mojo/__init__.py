"""ViPE's DROID/SLAM CUDA kernels (`vipe_ext`) in Mojo, for CPU, CUDA and Apple GPUs (MPS).

A drop-in for the parts of `vipe_ext` that MetricTrail calls: `import vipe_mojo as _C` exposes the six
submodules `vipe.ext` reads. Ported, with vipe_ext's signatures:

- `droid_net_ext.corr_index_forward`, `droid_net_ext.altcorr_index_forward`
- `slam_ext.ba_extended`, `slam_ext.frame_distance`

Every other function of those namespaces, and all of `grounding_dino_ext`, `utils_ext`, `scatter_ext`
and `corr_ext`, raises NotImplementedError when called.
"""

import os

if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()

from vipe_mojo import droid_net_ext, slam_ext  # noqa: E402
from vipe_mojo._unported import unported_namespace  # noqa: E402

grounding_dino_ext = unported_namespace("grounding_dino_ext")
utils_ext = unported_namespace("utils_ext")
scatter_ext = unported_namespace("scatter_ext")
corr_ext = unported_namespace("corr_ext")

__all__ = ["corr_ext", "droid_net_ext", "grounding_dino_ext", "scatter_ext", "slam_ext", "utils_ext"]
