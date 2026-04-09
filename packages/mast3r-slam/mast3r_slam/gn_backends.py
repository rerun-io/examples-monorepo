import os
from typing import Any


def _load_cuda_backend() -> Any:
    from mast3r_slam import _backends  # pyrefly: ignore

    return _backends


def _load_mojo_backend() -> Any:
    from mast3r_slam.max_ops import gauss_newton_rays as gauss_newton_rays_custom_op
    from mast3r_slam.max_ops import preferred_mojo_interface

    shared_backend: Any | None = None

    def _shared() -> Any:
        nonlocal shared_backend
        if shared_backend is None:
            import mast3r_slam_mojo_backends as mojo_backends  # pyrefly: ignore

            shared_backend = mojo_backends
        return shared_backend

    class _MojoGNBackend:
        def gauss_newton_points(self, *args: Any) -> Any:
            return _shared().gauss_newton_points_impl_idiomatic(tuple(args))

        def gauss_newton_rays(self, *args: Any) -> Any:
            if preferred_mojo_interface() == "custom_op":
                return gauss_newton_rays_custom_op(*args)
            return _shared().gauss_newton_rays_impl_idiomatic(tuple(args))

        def gauss_newton_calib(self, *args: Any) -> Any:
            return _shared().gauss_newton_calib_impl_idiomatic(tuple(args))

    return _MojoGNBackend()


def load_gn_backend(name: str | None = None) -> Any:
    backend_name = (name or os.environ.get("MAST3R_SLAM_GN_BACKEND", "cuda")).strip().lower()
    if backend_name == "cuda":
        return _load_cuda_backend()
    if backend_name == "mojo":
        return _load_mojo_backend()
    raise ValueError(f"Unknown GN backend: {backend_name}")
