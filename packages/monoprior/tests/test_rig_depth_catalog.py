"""Fast tests for X-Lens catalog sampling and output geometry."""

from typing import Any

import numpy as np
import pytest
import torch
from jaxtyping import Float64, UInt8
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig
from torch import Tensor

from monopriors.apis.benchmark_xlens_trt import nearest_time_index
from monopriors.apis.rig_depth_catalog import RigBatch, RigCollate, RigDepthCatalogConfig

CAMS: tuple[str, ...] = ("cam_00", "cam_01")


class _TimedFrame:
    """One camera's decoded frame beside its grid timestamp, as ``TimedNvdecDecoder`` returns it."""

    def __init__(self, t_ns: int, rgb: UInt8[Tensor, "3 h w"]) -> None:
        self.t_ns = t_ns
        self.rgb = rgb


def _row(t_ns: int, *, pose: bool = True) -> dict[str, Any]:
    """One dataloader row: a frame per camera whose pixels name the camera, plus a rig pose.

    The pose tensors are float32, the width Rerun stores ``Vec3D`` and ``Quaternion`` at, and empty
    when the grid slot precedes the first pose row — what ``NumericDecoder`` returns for that slot.
    """
    row: dict[str, Any] = {cam: _TimedFrame(t_ns, torch.full((3, 2, 4), index, dtype=torch.uint8)) for index, cam in enumerate(CAMS)}
    row["pose_t"] = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32) if pose else torch.empty(0, dtype=torch.float32)
    row["pose_q"] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32) if pose else torch.empty(0, dtype=torch.float32)
    return row


@pytest.fixture(scope="module")
def rr_config() -> RerunTyroConfig:
    """Provide a headless recording configuration without a viewer process."""
    return RerunTyroConfig(headless=True)


def test_catalog_config_accepts_model_patch_grid(rr_config: RerunTyroConfig) -> None:
    """The four outward cameras use the larger patch-aligned 16:9 grid."""
    config = RigDepthCatalogConfig(rr_config=rr_config)
    assert config.cams == ("cam_00", "cam_01", "cam_04", "cam_05")
    assert (config.width, config.height) == (1120, 630)
    assert config.batch_size == 1
    assert config.fuse is True
    assert config.log_points is False
    assert config.fusion_voxel_m == 0.04
    assert config.fusion_max_depth_m == 6.0
    assert config.mesh_every == 300


@pytest.mark.parametrize(("width", "height"), [(895, 504), (896, 503), (896, 490)])
def test_catalog_config_rejects_invalid_output_geometry(rr_config: RerunTyroConfig, width: int, height: int) -> None:
    """Both dimensions must be patch-aligned and preserve the source aspect ratio."""
    with pytest.raises(ValueError, match="multiples of 14|16:9"):
        RigDepthCatalogConfig(rr_config=rr_config, width=width, height=height)


def test_nearest_time_index_chooses_closest_sample() -> None:
    """The benchmark's single-frameset sampler picks nearest timestamps, including either endpoint."""
    times = np.array([0, 10, 20], dtype="timedelta64[ns]")
    assert nearest_time_index(times, 1) == 0
    assert nearest_time_index(times, 16) == 2
    assert nearest_time_index(times, -5) == 0
    assert nearest_time_index(times, 50) == 2


def test_collate_stacks_framesets_in_camera_order() -> None:
    """Every row becomes one frameset whose views follow the configured camera order."""
    batch: RigBatch = RigCollate(CAMS)([_row(0), _row(200)])
    assert batch.t_ns == [0, 200]
    assert batch.images.shape == (2, 2, 3, 2, 4)
    assert [int(batch.images[0, view, 0, 0, 0]) for view in range(len(CAMS))] == [0, 1]


def test_collate_builds_the_rig_pose_from_translation_and_xyzw_quaternion() -> None:
    """``pose_t`` and ``pose_q`` become one ``world_T_rig`` per frameset."""
    row: dict[str, Any] = _row(100)
    row["pose_q"] = torch.tensor([0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)], dtype=torch.float32)  # +90 degrees about z
    batch: RigBatch = RigCollate(CAMS)([row])
    expected: Float64[ndarray, "4 4"] = np.array(
        [[0.0, -1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64
    )
    assert batch.world_T_rig.dtype == np.float64
    assert np.allclose(batch.world_T_rig[0], expected, atol=1e-6)


def test_collate_drops_rows_without_every_camera_or_a_pose() -> None:
    """Grid slots before a camera's first packet (None) or before the first pose (empty) carry no frameset."""
    missing_frame: dict[str, Any] = _row(0)
    missing_frame[CAMS[1]] = None
    batch: RigBatch = RigCollate(CAMS)([missing_frame, _row(200, pose=False), _row(400)])
    assert batch.t_ns == [400]
    assert batch.images.shape == (1, 2, 3, 2, 4)
    assert batch.world_T_rig.shape == (1, 4, 4)


def test_collate_returns_an_empty_batch_the_run_loop_can_skip() -> None:
    """A batch whose rows are all incomplete stays shaped but empty."""
    batch: RigBatch = RigCollate(CAMS)([_row(0, pose=False)])
    assert batch.t_ns == []
    assert batch.images.shape[0] == 0
    assert batch.world_T_rig.shape == (0, 4, 4)
