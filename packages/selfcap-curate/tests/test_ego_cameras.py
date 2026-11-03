"""The OAK's three cameras read from a calibration layer that holds only the mount and a base that holds the factory facts."""

from pathlib import Path

import numpy as np
import pytest
import rerun as rr
from jaxtyping import Float32, Float64
from numpy import ndarray
from rerun.chunk import RrdReader
from simplecv.rerun_custom_types import DistortionCoefficientsBatch

from selfcap_curate.keypoints import EgoCamera, Keypoints3D, read_ego_cameras, write_keypoints

SEGMENT: str = "selfcap__2025-12-30__65d00748__ep001"
LENSES: dict[int, tuple[float, float, float, list[float]]] = {
    # focal px, principal point, OpenCV coefficients (k1 first)
    0: (700.0, 640.0, 360.0, [0.0] * 14),
    1: (500.0, 640.0, 360.0, [0.0] * 14),
    2: (600.0, 650.0, 350.0, [-0.1, *[0.0] * 13]),
}


def calib_layer(path: Path, oak_rig: int) -> None:
    """What calibrate writes for the OAK: the fitted mount on the rig and nothing below it."""
    with rr.RecordingStream("test", recording_id=SEGMENT, send_properties=False) as recording:
        recording.save(path)
        recording.log(f"/world/rig_{oak_rig:02d}", rr.Transform3D(translation=[0.0, 0.0, 0.0], mat3x3=np.eye(3)), rr.AnyValues(oak_mount_source="fitted"), static=True)


def base_layer(path: Path, oak_rig: int, quest_rig: int) -> None:
    """The factory facts: left and right under the rgb rig frame, each camera's own lens, and the Quest track."""
    rotated: Float64[ndarray, "3 3"] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # cam x along rig y
    with rr.RecordingStream("test", recording_id=SEGMENT, send_properties=False) as recording:
        recording.save(path)
        recording.log(f"/world/rig_{oak_rig:02d}/cam_01", rr.Transform3D(translation=[-0.04, 0.0, 0.0], mat3x3=rotated), static=True)
        recording.log(f"/world/rig_{oak_rig:02d}/cam_02", rr.Transform3D(translation=[0.04, 0.0, 0.0], mat3x3=np.eye(3)), static=True)
        for cam, (focal, cx, cy, coefficients) in LENSES.items():
            pinhole: str = f"/world/rig_{oak_rig:02d}/cam_{cam:02d}/pinhole"
            recording.log(pinhole, rr.Pinhole(focal_length=focal, principal_point=[cx, cy], width=1280, height=720), static=True)
            recording.log(pinhole, [DistortionCoefficientsBatch(np.asarray(coefficients, dtype=np.float32))], static=True)
        recording.send_columns(
            f"/world/rig_{quest_rig:02d}",
            indexes=[rr.TimeColumn("video_time", duration=np.array([0.46, 0.54]))],
            columns=rr.Transform3D.columns(translation=np.zeros((2, 3)), quaternion=np.tile([0.0, 0.0, 0.0, 1.0], (2, 1))),
        )


@pytest.fixture(params=[(4, 5), (3, 4)], ids=["four-phones", "three-phones"])
def rigs(request: pytest.FixtureRequest) -> tuple[int, int]:
    return request.param


def read_cameras_from_layers(tmp_path: Path, private_server, rigs: tuple[int, int]) -> tuple[Path, list[EgoCamera]]:
    calib: Path = tmp_path / "calib.rrd"
    base: Path = tmp_path / "base.rrd"
    calib_layer(calib, rigs[0])
    base_layer(base, *rigs)
    with private_server({"calib": [calib], "base": [base]}) as server:
        client = server.client()
        cameras: list[EgoCamera] = read_ego_cameras(client.get_dataset("calib"), client.get_dataset("base"), SEGMENT, *rigs)
    return calib, cameras


def test_each_oak_camera_comes_back_with_its_own_lens_and_factory_extrinsic(tmp_path: Path, private_server, rigs: tuple[int, int]) -> None:
    _, cameras = read_cameras_from_layers(tmp_path, private_server, rigs)
    assert [(camera.rig, camera.cam) for camera in cameras] == [(rigs[0], 0), (rigs[0], 1), (rigs[0], 2)]
    assert [camera.K[0, 0] for camera in cameras] == [700.0, 500.0, 600.0]
    assert [float(camera.distortion[0]) for camera in cameras] == pytest.approx([0.0, 0.0, -0.1])
    assert [float(camera.rig_T_cam[0, 3]) for camera in cameras] == pytest.approx([0.0, -0.04, 0.04])


def test_three_oak_cameras_give_three_skeletons_each_through_its_own_model(tmp_path: Path, private_server, rigs: tuple[int, int]) -> None:
    calib, cameras = read_cameras_from_layers(tmp_path, private_server, rigs)
    # Every joint at (0.1, -0.05, 1.0) in world and rig; the mount and the head are identity.
    times: Float64[ndarray, "1"] = np.array([0.5])
    xyz: Float32[ndarray, "1 133 3"] = np.tile(np.array([0.1, -0.05, 1.0], dtype=np.float32), (1, 133, 1))
    result: Keypoints3D = Keypoints3D(times, xyz, np.full((1, 133), 4, np.uint8), np.ones((1, 133), np.float32), {}, {}, 25.0)
    output: Path = tmp_path / "layer.rrd"
    write_keypoints(calib, output, result, ego=tuple(cameras))
    expected: dict[int, tuple[float, float]] = {
        0: (710.0, 325.0),  # 700 x 0.1 + 640, 700 x -0.05 + 360
        1: (615.0, 290.0),  # rig (0.14, -0.05, 1) minus the lever arm; turned 90 degrees: cam (-0.05, -0.14, 1); f 500
        2: (685.978, 320.018),  # cam (0.06, -0.05, 1); r^2 0.0061, k1 -0.1 scales by 0.99939; f 600, centre (650, 350)
    }
    for cam, pixel in expected.items():
        entity: str = f"/world/rig_{rigs[0]:02d}/cam_{cam:02d}/pinhole/skeleton"
        rows = [c.to_record_batch() for c in RrdReader(output).stream() if str(c.entity_path) == entity and not c.is_static]
        assert len(rows) == 1, entity
        points: Float64[ndarray, "133 2"] = np.asarray(rows[0].column("Points2D:positions").to_pylist()[0], dtype=np.float64)
        np.testing.assert_allclose(points[5], pixel, atol=1e-3, err_msg=entity)
