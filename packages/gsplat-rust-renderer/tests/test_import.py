"""Smoke test: verify the gsplat_rust_renderer package can be imported."""


def test_import_gsplat_rust_renderer() -> None:
    import gsplat_rust_renderer  # noqa: F401


def test_import_gaussians3d() -> None:
    from gsplat_rust_renderer.gaussians3d import Gaussians3D  # noqa: F401
