from __future__ import annotations

from typing import Any

try:
    import mast3r_slam_mojo_backends as _mojo_backends  # pyrefly: ignore
except ImportError:
    _mojo_backends = None

from mast3r_slam import _backends as _cuda_backends  # pyrefly: ignore

try:
    from mast3r_slam import max_ops as _max_ops
except Exception:
    _max_ops = None


def _use_mojo(name: str) -> bool:
    return _mojo_backends is not None and hasattr(_mojo_backends, name)


def gauss_newton_points(*args: Any) -> tuple[Any, ...]:
    if _use_mojo("gauss_newton_points_impl"):
        return _mojo_backends.gauss_newton_points_impl(args)
    return _cuda_backends.gauss_newton_points(*args)


def gauss_newton_rays(*args: Any) -> tuple[Any, ...]:
    if _use_mojo("gauss_newton_rays_impl"):
        return _mojo_backends.gauss_newton_rays_impl(args)
    return _cuda_backends.gauss_newton_rays(*args)


def gauss_newton_calib(*args: Any) -> tuple[Any, ...]:
    if _use_mojo("gauss_newton_calib_impl"):
        return _mojo_backends.gauss_newton_calib_impl(args)
    return _cuda_backends.gauss_newton_calib(*args)


gauss_newton_points_step = _cuda_backends.gauss_newton_points_step
gauss_newton_rays_step = _cuda_backends.gauss_newton_rays_step
gauss_newton_calib_step = _cuda_backends.gauss_newton_calib_step


def pose_retr(*args: Any) -> Any:
    if _max_ops is not None and _max_ops.has_pose_retr():
        return _max_ops.pose_retr(*args)
    if _use_mojo("pose_retr"):
        return _mojo_backends.pose_retr(*args)
    raise AttributeError("pose_retr is only available when mast3r_slam_mojo_backends is built")
