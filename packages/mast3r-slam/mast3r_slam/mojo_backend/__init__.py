"""MASt3R-SLAM's kernels in Mojo: matching (`iter_proj`, `refine_matches`) and the Gauss-Newton pose optimisation
(`gauss_newton_rays`, `gauss_newton_calib`), with the signatures of the former CUDA extension `mast3r_slam._backends`.

The kernels run on CUDA only (MASt3R itself runs on the GPU), on PyTorch's current stream; the Gauss-Newton pose
solve runs on the CPU, as the extension's Eigen step did. The extension `_mast3r_mojo` is built by the
`_mast3r-slam-build-mojo-kernels` task from `mojo/*.mojo`, against lietorch-mojo's runtimes
(`-I <lietorch-mojo>/lietorch_mojo/mojo`). Parity with the former CUDA extension on real MASt3R-SLAM inputs:
`tests/test_mojo_backend.py` against `tests/fixtures/kernel_references.rrd`.
"""

from mast3r_slam.mojo_backend.gn import gauss_newton_calib, gauss_newton_rays
from mast3r_slam.mojo_backend.matching import iter_proj, refine_matches

__all__ = ["gauss_newton_calib", "gauss_newton_rays", "iter_proj", "refine_matches"]
