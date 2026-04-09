import os
from typing import Any


def _load_cuda_backend() -> Any:
    from mast3r_slam import _backends  # pyrefly: ignore

    return _backends


def _load_mojo_backend() -> Any:
    import mast3r_slam_mojo_backends as mojo_backends  # pyrefly: ignore

    class _MojoGNBackend:
        def gauss_newton_points(self, *args: Any) -> Any:
            return mojo_backends.gauss_newton_points_impl_idiomatic(tuple(args))

        def gauss_newton_rays(self, *args: Any) -> Any:
            return mojo_backends.gauss_newton_rays_impl_idiomatic(tuple(args))

        def gauss_newton_calib(self, *args: Any) -> Any:
            return mojo_backends.gauss_newton_calib_impl_idiomatic(tuple(args))

    return _MojoGNBackend()


def load_gn_backend(name: str | None = None) -> Any:
    backend_name = (name or os.environ.get("MAST3R_SLAM_GN_BACKEND", "cuda")).strip().lower()
    if backend_name == "cuda":
        return _load_cuda_backend()
    if backend_name == "mojo":
        return _load_mojo_backend()
    raise ValueError(f"Unknown GN backend: {backend_name}")

