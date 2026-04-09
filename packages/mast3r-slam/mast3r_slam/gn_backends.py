from __future__ import annotations

import os
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
    if os.environ.get("MAST3R_SLAM_FORCE_CUDA_BACKENDS", "") not in ("", "0", "false", "False"):
        return False
    return _mojo_backends is not None and hasattr(_mojo_backends, name)


def _mojo_rays_variant() -> str:
    variant = os.environ.get("MAST3R_SLAM_FORCE_MOJO_RAYS", "").strip().lower()
    if variant in {"current", "idiomatic"}:
        return variant
    return ""


def _require_mojo_backends() -> Any:
    assert _mojo_backends is not None
    return _mojo_backends


def gauss_newton_points(*args: Any) -> tuple[Any, ...]:
    if _use_mojo("gauss_newton_points_impl"):
        return _require_mojo_backends().gauss_newton_points_impl(args)
    return tuple(_cuda_backends.gauss_newton_points(*args))


def gauss_newton_rays(*args: Any) -> tuple[Any, ...]:
    variant = _mojo_rays_variant()
    if variant == "idiomatic" and _use_mojo("gauss_newton_rays_impl_idiomatic"):
        return _require_mojo_backends().gauss_newton_rays_impl_idiomatic(args)
    if variant == "current" and _use_mojo("gauss_newton_rays_impl"):
        return _require_mojo_backends().gauss_newton_rays_impl(args)
    if _use_mojo("gauss_newton_rays_impl"):
        return _require_mojo_backends().gauss_newton_rays_impl(args)
    return tuple(_cuda_backends.gauss_newton_rays(*args))


def gauss_newton_calib(*args: Any) -> tuple[Any, ...]:
    if _use_mojo("gauss_newton_calib_impl"):
        return _require_mojo_backends().gauss_newton_calib_impl(args)
    return tuple(_cuda_backends.gauss_newton_calib(*args))


def gauss_newton_points_step(*args: Any) -> tuple[Any, ...]:
    return tuple(_cuda_backends.gauss_newton_points_step(*args))


def gauss_newton_rays_step(*args: Any) -> tuple[Any, ...]:
    variant = _mojo_rays_variant()
    if variant == "idiomatic" and _use_mojo("gauss_newton_rays_step_idiomatic"):
        return tuple(_require_mojo_backends().gauss_newton_rays_step_idiomatic(args))
    if variant == "current" and _use_mojo("gauss_newton_rays_step"):
        return tuple(_require_mojo_backends().gauss_newton_rays_step(args))
    if _use_mojo("gauss_newton_rays_step"):
        return tuple(_require_mojo_backends().gauss_newton_rays_step(args))
    return tuple(_cuda_backends.gauss_newton_rays_step(*args))


def gauss_newton_calib_step(*args: Any) -> tuple[Any, ...]:
    return tuple(_cuda_backends.gauss_newton_calib_step(*args))


def pose_retr(*args: Any) -> Any:
    if _max_ops is not None and _max_ops.has_pose_retr():
        return _max_ops.pose_retr(*args)
    if _use_mojo("pose_retr"):
        return _require_mojo_backends().pose_retr(*args)
    raise AttributeError("pose_retr is only available when mast3r_slam_mojo_backends is built")
