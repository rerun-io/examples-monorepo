"""CPython extension module entry point for the Mojo MASt3R-SLAM backend.

When Python does `import mast3r_slam_mojo_backends`, CPython calls the
`PyInit_mast3r_slam_mojo_backends()` function exported from the compiled
shared library. This file builds the module object and registers every
Python-callable function.

The registered API intentionally **mirrors the original CUDA extension**
(`mast3r_slam._backends`) so the Python SLAM pipeline can swap backends by
changing a single import — no call-site changes needed.

Registered functions
--------------------
Frontend (matching):
  iter_proj          — iterative projective search (Mojo GPU kernel)
  refine_matches     — descriptor-based match refinement (Mojo GPU kernel)

Backend (Gauss-Newton optimisation):
  pose_retr                    — Sim(3) pose retraction (Mojo GPU kernel)
  gauss_newton_rays_step       — one linearisation pass, ray cost (Mojo GPU kernel)
  gauss_newton_points[_impl]   — full GN loop, point cost (delegates to CUDA)
  gauss_newton_rays[_impl]     — full GN loop, ray cost (Mojo)
  gauss_newton_calib[_impl]    — full GN loop, calibration cost (delegates to CUDA)

The `_impl` suffixes are aliases — both names point to the same function so
either import style works from the Python side.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from gn import (
    gauss_newton_calib_impl,
    gauss_newton_points_impl,
    gauss_newton_rays_impl,
    gauss_newton_rays_step_py,
    pose_retr_py,
)
from matching import iter_proj_py, refine_matches_py
from python_interop import install_cached_context


@export
def PyInit_mast3r_slam_mojo_backends() -> PythonObject:
    """CPython module initialiser — called once at `import` time.

    Builds the module, registers all Python-callable functions, and creates
    the process-lifetime GPU DeviceContext that all Mojo kernels share.
    """
    try:
        var m = PythonModuleBuilder("mast3r_slam_mojo_backends")

        # ── Frontend (matching) ──
        m.def_function[iter_proj_py]("iter_proj")
        m.def_function[refine_matches_py]("refine_matches")

        # ── Backend (GN optimisation) ──
        m.def_function[pose_retr_py]("pose_retr")
        m.def_function[gauss_newton_rays_step_py]("gauss_newton_rays_step")
        m.def_function[gauss_newton_points_impl]("gauss_newton_points")
        m.def_function[gauss_newton_points_impl]("gauss_newton_points_impl")
        m.def_function[gauss_newton_rays_impl]("gauss_newton_rays")
        m.def_function[gauss_newton_rays_impl]("gauss_newton_rays_impl")
        m.def_function[gauss_newton_calib_impl]("gauss_newton_calib")
        m.def_function[gauss_newton_calib_impl]("gauss_newton_calib_impl")

        var module = m.finalize()
        # Allocate the shared GPU context and attach it to the module as
        # module._ctx_addr (an integer storing the heap pointer address).
        install_cached_context(module)
        return module
    except e:
        abort(String("Failed to create mast3r_slam_mojo_backends module: ", e))
