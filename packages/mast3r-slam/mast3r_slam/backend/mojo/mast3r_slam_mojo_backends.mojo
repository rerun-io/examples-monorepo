"""CPython extension module `mast3r_slam_mojo_backends`: the Mojo GPU kernels of MASt3R-SLAM.

`import mast3r_slam_mojo_backends` calls `PyInit_mast3r_slam_mojo_backends`, which registers the functions below and
creates the GPU DeviceContext they share. Matching entry points share names and signatures with `mast3r_slam._backends`.

Frontend (matching): iter_proj, refine_matches.
Backend (Gauss-Newton): gauss_newton_rays_step (one linearisation), pose_retr (retraction). The iteration loop and the
dense solve are Python (mast3r_slam/ops/optim.py); the points and calib costs use the CUDA extension.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from gn import gauss_newton_rays_step_py, pose_retr_py
from matching import iter_proj_py, refine_matches_py
from python_interop import install_cached_context


@export
def PyInit_mast3r_slam_mojo_backends() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("mast3r_slam_mojo_backends")
        m.def_function[iter_proj_py]("iter_proj")
        m.def_function[refine_matches_py]("refine_matches")
        m.def_function[gauss_newton_rays_step_py]("gauss_newton_rays_step")
        m.def_function[pose_retr_py]("pose_retr")
        var module = m.finalize()
        install_cached_context(module)
        return module
    except e:
        abort(String("Failed to create mast3r_slam_mojo_backends module: ", e))
