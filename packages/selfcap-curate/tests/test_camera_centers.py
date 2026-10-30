"""Camera-centre overlays use calibrated lenses and discard invisible centres."""

import numpy as np
from scipy.spatial.transform import Rotation

from selfcap_curate.camera_centers import camera_center_chunks, project_centers
from selfcap_curate.keypoints import Camera, camera_transforms


def test_centers_use_lens_and_clip_behind_and_outside_image() -> None:
    camera = Camera(np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]), np.eye(4), (0.1, 0.0), "/world/rig_00/cam_00", (100, 80))
    points = np.array([[[0.2, 0.1, 2.0], [0.0, 0.0, -1.0], [2.0, 0.0, 1.0], [0.0, -2.0, 1.0], [np.nan, 0.0, 1.0]]])
    pixels = project_centers(camera, points)
    np.testing.assert_allclose(pixels[0, 0], [60.0125, 45.00625], atol=1e-5)
    assert np.isnan(pixels[0, 1:]).all()


def test_camera_center_interpolates_rotation_before_applying_lever_arm() -> None:
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[:, 2, 3] = 3.0
    poses[1:, :3, :3] = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    offset = np.eye(4)
    offset[0, 3] = 1.0
    actual = camera_transforms(np.array([0.05, 0.2, -1.0]), np.array([0.0, 0.1, 0.4]), poses, offset)
    np.testing.assert_allclose(actual[0, :3, 3], [np.sqrt(0.5), 0.0, 3.0 - np.sqrt(0.5)])
    assert np.isnan(actual[1:]).all()


def test_center_overlay_uses_each_video_clock_and_omits_its_own_camera() -> None:
    intrinsics = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    other_pose = np.eye(4)
    other_pose[:3, 3] = [0.2, 0.1, 2.0]
    third_pose = other_pose.copy()
    third_pose[0, 3] = -0.2
    cameras = {
        "a": Camera(intrinsics, np.eye(4), (0.0, 0.0), "/world/rig_00/cam_00", (100, 80)),
        "b": Camera(intrinsics, other_pose, (0.0, 0.0), "/world/rig_01/cam_00", (100, 80)),
        "c": Camera(intrinsics, third_pose, (0.0, 0.0), "/world/rig_02/cam_00", (100, 80)),
    }
    chunks = list(camera_center_chunks(cameras, (), [], {"a": np.array([10, 20]), "b": np.array([12, 24, 36]), "c": np.array([13, 26])}))
    assert chunks[0].to_record_batch().column("Points2D:labels").to_pylist() == [["rig_01", "rig_02"]]
    assert chunks[2].to_record_batch().column("Points2D:labels").to_pylist() == [["rig_00", "rig_02"]]
    first, second = chunks[1].to_record_batch(), chunks[3].to_record_batch()
    assert np.asarray(first.column("video_time")).view(np.int64).tolist() == [10, 20]
    assert np.asarray(second.column("video_time")).view(np.int64).tolist() == [12, 24, 36]
    np.testing.assert_allclose(first.column("Points2D:positions").to_pylist(), [[[60.0, 45.0], [40.0, 45.0]], [[60.0, 45.0], [40.0, 45.0]]])
    assert np.isnan(np.asarray(second.column("Points2D:positions").to_pylist())).all()
