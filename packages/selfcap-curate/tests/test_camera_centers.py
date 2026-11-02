"""Camera-centre overlays use calibrated lenses and discard invisible centres."""

from pathlib import Path

import numpy as np
import pytest
import rerun as rr
from dataforge.triangulation import Camera, camera_transforms
from jaxtyping import Float64
from numpy import ndarray
from scipy.spatial.transform import Rotation

from selfcap_curate.camera_centers import CenterTrack, camera_center_chunks, project_centers, read_quest_centers

SEGMENT: str = "selfcap__2025-12-30__65d00748__ep001"
QUEST: str = "/world/rig_05"


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


def quest_layers(calib: Path, base: Path, videos: tuple[int, ...], calib_right_m: float | None, base_right_m: float | None) -> None:
    """A base with the head track, its Quest videos and any factory right eye; a calib layer with any resolved borrowed camera."""
    with rr.RecordingStream("test", recording_id=SEGMENT, send_properties=False) as recording:
        recording.save(calib)
        recording.log("/world/rig_00/cam_00", rr.Transform3D(translation=[1.0, 0.0, 0.0], mat3x3=np.eye(3)), static=True)  # an exo camera: always present
        if calib_right_m is not None:
            recording.log(f"{QUEST}/cam_01", rr.Transform3D(translation=[calib_right_m, 0.0, 0.0], mat3x3=np.eye(3)), static=True)
    with rr.RecordingStream("test", recording_id=SEGMENT, send_properties=False) as recording:
        recording.save(base)
        for cam in videos:
            recording.log(f"{QUEST}/cam_{cam:02d}/pinhole/video", rr.VideoStream.from_fields(codec=rr.VideoCodec.AV1), static=True)
        if base_right_m is not None:
            recording.log(f"{QUEST}/cam_01", rr.Transform3D(translation=[base_right_m, 0.0, 0.0], mat3x3=np.eye(3)), static=True)
        recording.send_columns(
            QUEST,
            indexes=[rr.TimeColumn("video_time", duration=np.array([0.0, 0.1]))],
            columns=rr.Transform3D.columns(translation=np.zeros((2, 3)), quaternion=np.tile([0.0, 0.0, 0.0, 1.0], (2, 1))),
        )


@pytest.mark.parametrize(
    ("videos", "calib_right_m", "base_right_m", "expected"),
    [
        pytest.param((0, 1), None, 0.06, {"Quest L": 0.0, "Quest R": 0.06}, id="healthy-factory-right-eye-from-the-base"),
        pytest.param((1,), 0.03, None, {"Quest 50": 0.03}, id="borrowed-camera-50-from-the-calib-layer"),
        pytest.param((1,), None, None, {}, id="unresolved-camera-50-invents-no-baseline"),
        pytest.param((0, 1), 0.07, 0.06, {"Quest L": 0.0, "Quest R": 0.07}, id="calib-over-base"),
    ],
)
def test_quest_centers_read_each_camera_calib_over_base(
    tmp_path: Path, private_server, videos: tuple[int, ...], calib_right_m: float | None, base_right_m: float | None, expected: dict[str, float]
) -> None:
    calib: Path = tmp_path / "calib.rrd"
    base: Path = tmp_path / "base.rrd"
    quest_layers(calib, base, videos, calib_right_m, base_right_m)
    with private_server({"calib": [calib], "base": [base]}) as server:
        client = server.client()
        tracks: list[CenterTrack] = read_quest_centers(client.get_dataset("calib"), client.get_dataset("base"), SEGMENT, 5)
    assert {track.label: float(track.rig_T_cam[0, 3]) for track in tracks} == pytest.approx(expected)
    for track in tracks:
        identity: Float64[ndarray, "2 4 4"] = np.tile(np.eye(4), (2, 1, 1))
        np.testing.assert_allclose(track.world_T_rig, identity)
        np.testing.assert_allclose(track.times_s, [0.0, 0.1])
