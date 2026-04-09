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
    try:
        var m = PythonModuleBuilder("mast3r_slam_mojo_backends")
        m.def_function[iter_proj_py]("iter_proj")
        m.def_function[refine_matches_py]("refine_matches")
        m.def_function[pose_retr_py]("pose_retr")
        m.def_function[gauss_newton_rays_step_py]("gauss_newton_rays_step")
        m.def_function[gauss_newton_points_impl]("gauss_newton_points_impl")
        m.def_function[gauss_newton_rays_impl]("gauss_newton_rays_impl")
        m.def_function[gauss_newton_calib_impl]("gauss_newton_calib_impl")
        var module = m.finalize()
        install_cached_context(module)
        return module
    except e:
        abort(String("Failed to create mast3r_slam_mojo_backends module: ", e))
