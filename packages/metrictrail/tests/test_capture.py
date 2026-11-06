"""Pure parts of the live capture: DPVO pose conversion, frustums, DINOv3 PCA, the kept pass."""

from pathlib import Path

import numpy as np
import pytest
from jaxtyping import Float32, Float64
from numpy import ndarray
from scipy.spatial.transform import Rotation

from metrictrail.capture import CapturedPass, CaptureSession, DinoPca, dpvo_world_T_cam, frustum_strips, normals_rgb, rgb_normals

torch = pytest.importorskip("torch", reason="torch comes from the metrictrail env's cuda feature")


def test_dpvo_world_T_cam_inverts_dpvo_world_to_camera_poses() -> None:
    rng: np.random.Generator = np.random.default_rng(0)
    rotations: Float64[ndarray, "n 3 3"] = Rotation.random(5, random_state=1).as_matrix()
    world_T_cam: Float64[ndarray, "n 4 4"] = np.tile(np.eye(4), (5, 1, 1))
    world_T_cam[:, :3, :3] = rotations
    world_T_cam[:, :3, 3] = rng.normal(size=(5, 3))
    cam_T_world: Float64[ndarray, "n 4 4"] = np.linalg.inv(world_T_cam)
    dpvo: Float32[ndarray, "n 7"] = np.concatenate([cam_T_world[:, :3, 3], Rotation.from_matrix(cam_T_world[:, :3, :3]).as_quat()], axis=1).astype(
        np.float32
    )
    np.testing.assert_allclose(dpvo_world_T_cam(dpvo), world_T_cam, atol=1e-5)


def test_frustum_strips_one_rectangle_and_four_rays_per_camera() -> None:
    poses: Float32[ndarray, "2 4 4"] = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    poses[1, :3, 3] = [1.0, 0.0, 0.0]
    strips = frustum_strips(poses, np.array([100.0, 100.0, 50.0, 40.0], dtype=np.float32), (80, 100), 0.5)
    assert len(strips) == 10
    assert strips[0].shape == (5, 3)
    np.testing.assert_allclose(strips[0][0], [-0.25, -0.2, 0.5], atol=1e-6)  # top-left corner at depth 0.5
    np.testing.assert_allclose(strips[5][0], [0.75, -0.2, 0.5], atol=1e-6)


def test_dino_pca_is_fixed_after_the_fit() -> None:
    generator = torch.Generator().manual_seed(0)
    first = torch.randn(8, 6, 5, generator=generator)
    pca: DinoPca = DinoPca.fit(first)
    rgb = pca.rgb(first)
    assert rgb.shape == (6, 5, 3) and rgb.dtype == np.uint8
    assert np.array_equal(pca.rgb(first), rgb)
    assert rgb.max() == 255 and rgb.min() == 0


def test_kept_pass_is_the_last_of_the_longest(tmp_path: Path) -> None:
    session = CaptureSession(out_dir=tmp_path, application_id="a", recording_id="r", depth_range_m=(0.3, 3.0))
    assert session.kept_pass() is None
    session.passes = [
        CapturedPass(tmp_path / "pass_0.rrd", 32),
        CapturedPass(tmp_path / "pass_1.rrd", 596),
        CapturedPass(tmp_path / "pass_2.rrd", 596),
    ]
    kept = session.kept_pass()
    assert kept is not None and kept.path is not None and kept.path.name == "pass_2.rrd"


def test_normals_survive_the_8bit_rgb_round_trip() -> None:
    generator = torch.Generator().manual_seed(0)
    normals = torch.nn.functional.normalize(torch.randn(6, 5, 3, generator=generator), dim=-1)
    rgb = normals_rgb(normals)
    assert rgb.dtype == torch.uint8
    back: Float32[ndarray, "6 5 3"] = rgb_normals(rgb.numpy())
    np.testing.assert_allclose(np.linalg.norm(back, axis=-1), 1.0, atol=1e-5)
    np.testing.assert_allclose(back, normals.numpy(), atol=0.01)


def test_kept_normals_follow_the_kept_pass_and_superseded_passes_are_emptied(tmp_path: Path) -> None:
    session = CaptureSession(out_dir=tmp_path, application_id="a", recording_id="r", depth_range_m=(0.3, 3.0))
    assert session.kept_normals() == {}
    image: np.ndarray = np.zeros((2, 2, 3), dtype=np.uint8)
    for marker, frames in enumerate((32, 596, 596)):
        session.pass_frames = frames
        session.open_pass(len(session.passes))
        session.pass_normals[-1][0] = image + marker
    session.uninstall()
    assert [len(normals) for normals in session.pass_normals] == [0, 0, 1]
    assert session.kept_normals()[0][0, 0, 0] == 2
