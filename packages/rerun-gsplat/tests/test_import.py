"""Stage-0 gate: the env carries the 0.36 prerelease SDK, its dataloader, and gsplat."""

import rerun as rr


def test_rerun_is_036_prerelease() -> None:
    assert rr.__version__.startswith("0.36.0"), rr.__version__


def test_gaussian_splats_archetype_exists() -> None:
    assert hasattr(rr, "GaussianSplats3D")


def test_dataloader_api_present() -> None:
    from rerun.experimental.dataloader import (
        BlockShuffle,  # 0.36-only: keyframe-aware shuffling
        DataSource,
        Field,
        RerunMapDataset,
    )

    assert all([DataSource, Field, RerunMapDataset, BlockShuffle])


def test_gsplat_imports() -> None:
    from gsplat import MCMCStrategy, rasterization

    assert all([MCMCStrategy, rasterization])


def test_package_imports() -> None:
    import rerun_gsplat

    assert rerun_gsplat is not None
