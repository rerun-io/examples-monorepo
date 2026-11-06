"""lietorch_mojo: lietorch's Lie group API on a Mojo backend.

`import lietorch_mojo as lietorch` is a drop-in for upstream lietorch
(VSLAM-LAB/lietorch @ 3e2695dd). SE3 is backed by Mojo kernels (CPU and CUDA,
float32 and float64, forward and backward); SO3, RxSO3 and Sim3 are API shells
whose ops raise NotImplementedError until their Mojo groups land.
"""

import os

if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()

from lietorch_mojo import gradcheck  # noqa: E402,F401  (lietorch.gradcheck is a public submodule upstream)
from lietorch_mojo.groups import SE3, SO3, LieGroupParameter, RxSO3, Sim3, cat, stack  # noqa: E402

__all__ = ["groups", "LieGroupParameter", "SO3", "RxSO3", "SE3", "Sim3", "cat", "stack"]
