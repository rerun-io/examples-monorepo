"""TRiGS's ``build_canonical_from_npz`` accepts the RoMa init cloud's ``to_trigs_mapping`` (synthetic arrays and the real layer)."""

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("trigs", reason="TRiGS comes from recipes/open4dgs-trigs-src (open4dgs envs)")

pytestmark = pytest.mark.integration

REAL_INIT: Path = Path("/mnt/nas/ssd-8tb/open4dgs/init/mamma-markerless-dance__140725_Breakdancemarkers_Improv_2_03686_1_1_1__roma__f99-60.rrd")
"""The MAMMA 1-person init_roma layer (``open4dgs-init-roma`` defaults)."""


def _upstream_like_arrays() -> dict[str, np.ndarray]:
    """Upstream's array layout: 3 keyframes (source 100, 103, 106) of a moving blob, a sub-keyframe at 101 appended last."""
    rng = np.random.default_rng(0)
    source = np.repeat(np.array([100, 103, 106, 101], dtype=np.int32), [300, 300, 300, 100]).reshape(-1, 1)
    n = source.shape[0]
    velocity = np.array([0.01, 0.0, 0.0], np.float32)
    positions = (rng.normal(scale=0.2, size=(n, 3)) + (source - 100) * velocity).astype(np.float32)
    return {
        "positions": positions,
        "colors": rng.random((n, 3), dtype=np.float32),
        "velocities": np.tile(velocity, (n, 1)),
        "times": ((source - 100) / 6.0).astype(np.float32),
        "durations": np.full((n, 1), 0.5, np.float32),
        "has_velocity": np.ones(n, bool),
        "is_static": np.zeros(n, bool),
        "dynamic_protect": np.zeros(n, bool),
        "dynamic_allowed": np.ones(n, bool),
        "stage_keep": np.ones(n, bool),
        "frame_start": np.array(0, np.int32),
        "frame_end": np.array(6, np.int32),
        "keyframe_step": np.array(3, np.int32),
        "source_frame_start": np.array(100, np.int32),
        "source_frames": source,
        "fps": np.array(30.0, np.float32),
        "velocity_units": np.array("world_per_video_frame"),
    }


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_build_canonical_accepts_converted_upstream_arrays() -> None:
    from trigs.init.from_colmap import build_canonical_from_npz

    from open4dgs.roma_init import init_cloud_from_upstream

    cloud = init_cloud_from_upstream(_upstream_like_arrays(), dataset="d", segment_id="s")
    params, sh_degree, scene_extent = build_canonical_from_npz(
        cloud.to_trigs_mapping(), fps=cloud.fps, keyframe_stride=cloud.keyframe_step, device=_device()
    )
    assert params["means"].shape[0] == cloud.num_rows
    assert sh_degree == 3 and np.isfinite(scene_extent) and scene_extent > 0
    assert all(torch.isfinite(value).all() for value in params.values())


def test_build_canonical_accepts_the_mamma_init_roma_layer() -> None:
    if not REAL_INIT.exists():
        pytest.skip(f"the init_roma layer {REAL_INIT} is absent (run pixi run -e open4dgs open4dgs-init-roma)")
    from trigs.init.from_colmap import build_canonical_from_npz

    from open4dgs.init_cloud import read_init_cloud

    cloud = read_init_cloud(REAL_INIT)
    params, _sh_degree, scene_extent = build_canonical_from_npz(
        cloud.to_trigs_mapping(), fps=cloud.fps, keyframe_stride=cloud.keyframe_step, device=_device()
    )
    assert params["means"].shape[0] == cloud.num_rows and scene_extent > 0
