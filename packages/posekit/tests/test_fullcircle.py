"""FullCircle helpers: nearest-frame fill, the view up hint, disk dilation, mask scales, and the lens calibration reader."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch


def test_fill_nearest_prefers_the_earlier_frame_at_equal_distance() -> None:
    from posekit.models.fullcircle import fill_nearest

    a: np.ndarray = np.array([1.0, 0.0, 0.0])
    b: np.ndarray = np.array([0.0, 1.0, 0.0])
    filled: list[np.ndarray] = fill_nearest([None, a, None, b, None, None])
    assert [int(np.argmax(d)) for d in filled] == [0, 0, 0, 1, 1, 1]


def test_fill_nearest_refuses_a_sequence_without_detections() -> None:
    from posekit.models.fullcircle import fill_nearest

    with pytest.raises(ValueError, match="no person"):
        fill_nearest([None, None])


def test_view_up_keeps_fullcircle_up_unless_the_operator_is_near_it() -> None:
    from posekit.models.fullcircle import view_up

    side: list[np.ndarray] = [np.array([1.0, 0.2, 0.0]) for _ in range(5)]
    np.testing.assert_array_equal(view_up(side), [0.0, 1.0, 0.0])
    below: list[np.ndarray] = [np.array([0.1, 1.0, 0.2]) for _ in range(5)]
    np.testing.assert_array_equal(view_up(below), [0.0, 0.0, 1.0])


def test_view_up_avoids_look_at_flips_near_the_nadir() -> None:
    from posekit.models.fullcircle import view_up
    from posekit.ops.sphere import look_at

    directions: list[np.ndarray] = [np.array([0.12 * math.cos(a), 1.0, 0.12 * math.sin(a)]) for a in np.linspace(0, 2 * math.pi, 16)]
    up: np.ndarray = view_up(directions)
    x_axes: np.ndarray = np.stack([look_at(d, up)[:, 0] for d in directions])
    # The view's x axis stays within a few degrees across the ring of directions (no roll flips).
    assert float(np.min(x_axes @ x_axes[0])) > math.cos(math.radians(20))


def test_disk_dilate_radius() -> None:
    from posekit.models.fullcircle import disk_dilate

    mask: torch.Tensor = torch.zeros(11, 11, dtype=torch.bool)
    mask[5, 5] = True
    grown: torch.Tensor = disk_dilate(mask, 2)
    ys, xs = torch.nonzero(grown, as_tuple=True)
    distance: torch.Tensor = ((ys - 5) ** 2 + (xs - 5) ** 2).float().sqrt()
    assert float(distance.max()) <= 2.0 + 1e-6
    assert int(grown.sum()) == 13
    assert torch.equal(disk_dilate(mask, 0), mask)


@pytest.mark.integration
@pytest.mark.parametrize("unified", [False, True])
def test_read_lens_sources_puts_lens_1_in_lens_0_frame(tmp_path, unified: bool) -> None:
    """Either Mei name gives the same lens: ``mei_v6`` with gamma as the Pinhole focal, or ``mei_v6_unified`` with gamma
    in the coefficients and the paraxial focal in the Pinhole (exoego:v2 section 4)."""
    import rerun as rr
    from scipy.spatial.transform import Rotation
    from simplecv.rerun_custom_types import CameraDistortion, PinholeWithDistortion

    from posekit.apis.fullcircle_masks import read_lens_sources
    from posekit.ops.sphere import MeiCamera

    coefficients: list[float] = [2.3, 1.745, 0.94, -1.68, 2.19, 3.86, 0.0, -0.005, 0.0016, -0.0033, 0.004, 0.0083, -0.0013, 0.024, -0.0014]
    cam_R_rigs: list[np.ndarray] = [Rotation.from_euler("xyz", angles).as_matrix() for angles in ([0.1, -1.5, 0.2], [-0.05, 1.6, 0.1])]
    path = tmp_path / "base.rrd"
    recording: rr.RecordingStream = rr.RecordingStream("dataforge", recording_id="test")
    recording.save(path)
    for cam, cam_R_rig in enumerate(cam_R_rigs):
        node: str = f"/world/rig_00/cam_0{cam}"
        recording.log(
            node, rr.Transform3D(mat3x3=cam_R_rig, translation=[0.0, 0.0, 0.01], relation=rr.TransformRelation.ChildFromParent), static=True
        )
        fx, fy = (1835.0 / 3.3, 1834.0 / 3.3) if unified else (1835.0, 1834.0)
        pinhole: rr.Pinhole = rr.Pinhole(image_from_camera=[[fx, 0.0, 958.0], [0.0, fy, 966.0], [0.0, 0.0, 1.0]], width=1920, height=1920)
        logged: list[float] = [1835.0, 1834.0, *coefficients] if unified else coefficients
        name: str = "mei_v6_unified" if unified else "mei_v6"
        recording.log(f"{node}/pinhole", PinholeWithDistortion(pinhole, CameraDistortion(name, np.asarray(logged))), static=True)
    recording.flush()
    recording.disconnect()  # closes the file sink, which writes the footer the lazy store reads
    with rr.server.Server(datasets={"base": [path]}) as server:
        sources, cam0_R_rig = read_lens_sources(server.client().get_dataset("base"), "test", ("/world/rig_00/cam_00", "/world/rig_00/cam_01"), 4)
    np.testing.assert_allclose(cam0_R_rig, cam_R_rigs[0], atol=1e-6)
    np.testing.assert_allclose(sources[0].rig_R_cam, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(sources[1].rig_R_cam, cam_R_rigs[0] @ cam_R_rigs[1].T, atol=1e-6)
    assert sources[1].image_hw == (1920, 1920) and sources[1].mask_hw == (480, 480)
    lens = sources[0].camera
    assert isinstance(lens, MeiCamera)
    assert (lens.model, lens.cx, lens.cy) == ("mei_v6", 958.0, 966.0)
    np.testing.assert_allclose([lens.fx, lens.fy], [1835.0, 1834.0], rtol=1e-6)
    np.testing.assert_allclose([lens.xi, lens.half_field_rad, *lens.distortion], coefficients, rtol=1e-6)


def test_level_rotation_puts_gravity_on_y() -> None:
    from posekit.models.fullcircle import level_rotation

    down: np.ndarray = np.array([-0.56, 0.01, -0.83])
    rig_R_level: np.ndarray = level_rotation(down)
    np.testing.assert_allclose(rig_R_level @ rig_R_level.T, np.eye(3), atol=1e-12)
    assert abs(np.linalg.det(rig_R_level) - 1.0) < 1e-12
    np.testing.assert_allclose(rig_R_level.T @ (down / np.linalg.norm(down)), [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(level_rotation(np.array([0.0, 2.0, 0.0])), np.eye(3), atol=1e-12)


def test_mask_cells_sample_where_the_written_scale_places_them() -> None:
    from posekit.models.fullcircle import SphereSource
    from posekit.ops.sphere import EquirectCamera, pixel_grid

    # 11 x 8 image, 2 x 2 mask: 5.5 image pixels per cell across, 4 down.
    source: SphereSource = SphereSource(EquirectCamera(11, 8), np.eye(3), (8, 11), (2, 2))
    assert source.mask_scale == (5.5, 4.0)
    centres: np.ndarray = pixel_grid(*source.mask_hw, "cpu", scale=source.mask_scale).numpy()
    # Transform3D(scale=[sx, sy, 1]) puts mask cell (j, i) over image pixels [i sx, (i + 1) sx) x [j sy, (j + 1) sy).
    expected: list[list[float]] = [[(i + 0.5) * 5.5 - 0.5, (j + 0.5) * 4.0 - 0.5] for j in range(2) for i in range(2)]
    np.testing.assert_allclose(centres, expected)
