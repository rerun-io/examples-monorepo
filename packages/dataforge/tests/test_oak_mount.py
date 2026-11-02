from __future__ import annotations

from pathlib import Path

import numpy as np
import rerun as rr
from jaxtyping import Float64
from numpy import ndarray
from rerun.server import Server
from simplecv.rerun_custom_types import DistortionCoefficientsBatch

from dataforge.oak_mount import StaticCamera, compose_track, quaternion_matrices, read_static_cameras

SEGMENT: str = "selfcap__2025-12-30__65d00748__ep001"


def test_compose_track_hangs_the_oak_on_the_quest() -> None:
    # Quest at (1, 2, 3) yawed 90 deg about +y; the mount puts the OAK 5 cm along the Quest's +x.
    half = np.sqrt(0.5)
    mount = np.eye(4)
    mount[:3, 3] = [0.05, 0.0, 0.0]
    world_T_oak = compose_track(np.array([[1.0, 2.0, 3.0]]), np.array([[0.0, half, 0.0, half]]), mount)
    np.testing.assert_allclose(world_T_oak[0, :3, 3], [1.0, 2.0, 3.0 - 0.05], atol=1e-12)


def test_quaternion_matrices_match_identity_and_normalise() -> None:
    np.testing.assert_allclose(quaternion_matrices(np.array([[0.0, 0.0, 0.0, 2.0]]))[0], np.eye(3))


def layer(path: Path, entries: dict[str, list[object]]) -> None:
    with rr.RecordingStream("test", recording_id=SEGMENT, send_properties=False) as recording:
        recording.save(path)
        for entity, archetypes in entries.items():
            recording.log(entity, *archetypes, static=True)


def test_static_cameras_take_each_component_from_calibration_over_base(tmp_path: Path) -> None:
    """The fitted lens wins; the transform and distortion only the base holds still come through; an absent camera is all None."""
    camera: str = "/world/rig_05/cam_01"
    turned: Float64[ndarray, "3 3"] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    calibration: Path = tmp_path / "calibration.rrd"
    base: Path = tmp_path / "base.rrd"
    layer(calibration, {f"{camera}/pinhole": [rr.Pinhole(focal_length=700.0, principal_point=[640.0, 360.0], width=1280, height=720)]})
    layer(
        base,
        {
            camera: [rr.Transform3D(translation=[0.06, 0.0, 0.0], mat3x3=turned)],
            f"{camera}/pinhole": [
                rr.Pinhole(focal_length=500.0, principal_point=[320.0, 240.0], width=640, height=480),
                [DistortionCoefficientsBatch(np.array([-0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))],
            ],
        },
    )
    with Server(host="127.0.0.1", port=0, datasets={"calibration": [calibration], "base": [base]}) as server:
        client = server.client()
        cameras: dict[str, StaticCamera] = read_static_cameras(
            client.get_dataset("calibration").filter_segments(SEGMENT), client.get_dataset("base").filter_segments(SEGMENT), [camera, "/world/rig_05/cam_00"]
        )
    found: StaticCamera = cameras[camera]
    assert found.rig_T_cam is not None and found.K is not None and found.distortion is not None
    np.testing.assert_allclose(found.rig_T_cam[:3, :3], turned)
    np.testing.assert_allclose(found.rig_T_cam[:3, 3], [0.06, 0.0, 0.0])
    np.testing.assert_allclose(found.K, [[700.0, 0.0, 640.0], [0.0, 700.0, 360.0], [0.0, 0.0, 1.0]])
    assert found.resolution == (1280, 720)
    np.testing.assert_allclose(found.distortion[0], -0.1)
    assert cameras["/world/rig_05/cam_00"] == StaticCamera(None, None, None, None)
