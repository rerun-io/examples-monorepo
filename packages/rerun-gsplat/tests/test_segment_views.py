"""Unit tests for the sample→view adapter, plus a live-catalog integration gate."""

import numpy as np
import pytest
import torch
from conftest import PROBE_CATALOG_URL, catalog_reachable
from jaxtyping import Float32
from numpy import ndarray
from scipy.spatial.transform import Rotation
from torch import Tensor

from rerun_gsplat.apis.segment_views import SegmentViewsConfig, SplatView, load_segment_views, view_from_sample


def synthetic_sample(height: int = 64, width: int = 96) -> dict[str, Tensor]:
    quat_xyzw: Float32[ndarray, "4"] = Rotation.from_euler("y", 0.3).as_quat().astype(np.float32)
    depth_height: int = height // 4
    depth_width: int = width // 4
    depth_mm: Tensor = torch.full((1, depth_height, depth_width), 1500, dtype=torch.uint16)
    depth_mm[0, 0, 0] = 0  # invalid: no return
    conf: Tensor = torch.full((depth_height, depth_width), 2, dtype=torch.uint8)
    conf[0, 1] = 0  # invalid: low confidence
    return {
        "video": torch.arange(3 * height * width, dtype=torch.uint8).reshape(3, height, width) % 251,
        "pose_t": torch.tensor([1.0, 2.0, 3.0]),
        "pose_q": torch.from_numpy(quat_xyzw),
        # Column-major flattening of [[100, 0, 48], [0, 100, 32], [0, 0, 1]],
        # matching how Rerun stores Pinhole image_from_camera.
        "k": torch.tensor([100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 48.0, 32.0, 1.0]),
        "k_lo": torch.tensor([25.0, 0.0, 0.0, 0.0, 25.0, 0.0, 12.0, 8.0, 1.0]),
        "depth": depth_mm,
        "conf": conf,
    }


def test_view_from_sample_shapes_and_intrinsics() -> None:
    view: SplatView = view_from_sample(synthetic_sample(), downscale=2)
    assert view.rgb_hwc.shape == (32, 48, 3)
    assert view.rgb_hwc.dtype == torch.uint8
    assert torch.allclose(view.k_33[0, 0], torch.tensor(50.0))
    assert torch.allclose(view.k_33[:2, 2], torch.tensor([24.0, 16.0]))
    assert view.k_33[2, 2] == 1.0
    # Depth stays native-resolution, millimeters -> meters, masked by zero/low-confidence.
    assert view.depth_m_hw.shape == (16, 24)
    assert torch.isclose(view.depth_m_hw[1, 1], torch.tensor(1.5))
    assert not view.depth_valid_hw[0, 0]
    assert not view.depth_valid_hw[0, 1]
    assert bool(view.depth_valid_hw[1:, 1:].all())
    # The lowres K is NOT downscaled (it belongs to the depth resolution).
    assert torch.allclose(view.k_lowres_33[0, 0], torch.tensor(25.0))


def test_view_from_sample_pose_is_world_to_cam() -> None:
    sample: dict[str, Tensor] = synthetic_sample()
    view: SplatView = view_from_sample(sample, downscale=1)
    world_t_cam: Float32[ndarray, "4 4"] = np.eye(4, dtype=np.float32)
    world_t_cam[:3, :3] = Rotation.from_quat(sample["pose_q"].numpy()).as_matrix()
    world_t_cam[:3, 3] = sample["pose_t"].numpy()
    roundtrip: Float32[ndarray, "4 4"] = view.cam_t_world_44.numpy() @ world_t_cam
    assert np.allclose(roundtrip, np.eye(4), atol=1e-5)


@pytest.mark.skipif(not catalog_reachable(), reason="ephemeral catalog server on :51299 not running")
def test_load_segment_views_live() -> None:
    config: SegmentViewsConfig = SegmentViewsConfig(catalog_url=PROBE_CATALOG_URL, target_view_count=24)
    views: list[SplatView] = load_segment_views(config)
    assert len(views) >= 20
    for view in views:
        assert view.rgb_hwc.shape == (720, 960, 3)
        rotation: Float32[ndarray, "3 3"] = view.cam_t_world_44[:3, :3].numpy()
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4)
        assert np.isclose(float(np.linalg.det(rotation)), 1.0, atol=1e-4)
        assert 100.0 < float(view.k_33[0, 0]) < 2000.0
        assert view.depth_m_hw.shape == (192, 256)
        assert float(view.depth_valid_hw.float().mean()) > 0.2
        valid_depths = view.depth_m_hw[view.depth_valid_hw]
        assert 0.1 < float(valid_depths.median()) < 10.0
        assert abs(float(view.k_33[0, 2]) - 480.0) < 100.0
