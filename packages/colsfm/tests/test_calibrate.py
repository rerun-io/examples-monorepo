"""`colsfm.calibrate`: the rig-window pairs, the seed cameras, and the rig solve recovering a known fisheye rig."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pycolmap
import pytest
from scipy.spatial.transform import Rotation

from colsfm.calibrate import cameras_from_rig, rig_window_pairs, seed_camera, solve_rig

WIDTH: int = 640
HEIGHT: int = 480
NAMES: tuple[str, ...] = ("front", "left", "right")
TRUE_PARAMS: tuple[list[float], ...] = (
    [210.0, 209.0, 322.0, 238.0, 0.02, 0.005, -0.002, 0.0005],
    [206.0, 206.5, 318.0, 243.0, 0.03, -0.004, 0.001, 0.0],
    [213.0, 212.0, 325.0, 236.0, 0.01, 0.008, -0.003, 0.0008],
)
"""fx fy cx cy k1..k4 per camera; the seed (640 / pi = 203.7 px, centred, no distortion) is a few percent off, as on RoboCap."""


def _rig_T_cam(yaw_deg: float, translation: list[float]) -> np.ndarray:
    pose: np.ndarray = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("y", yaw_deg, degrees=True).as_matrix()
    pose[:3, 3] = translation
    return pose


TRUE_RIG_T_CAM: tuple[np.ndarray, ...] = (
    _rig_T_cam(0.0, [0.0, 0.0, 0.0]),
    _rig_T_cam(-70.0, [-0.08, 0.01, -0.05]),
    _rig_T_cam(70.0, [0.09, -0.01, -0.06]),
)
"""Front, and two side cameras looking 70 degrees out (camera frame: x right, y down, z forward)."""


def test_rig_window_pairs_join_every_camera_of_nearby_rig_frames() -> None:
    image_ids: list[list[int | None]] = [[1, 2], [3, None], [5, 6]]
    assert rig_window_pairs(image_ids, overlap=1) == [(1, 2), (1, 3), (2, 3), (3, 5), (3, 6), (5, 6)]
    assert (1, 5) in rig_window_pairs(image_ids, overlap=2)


def test_a_fisheye_seed_is_a_prior_and_a_pinhole_seed_is_a_guess() -> None:
    fisheye: pycolmap.Camera = seed_camera(1, "OPENCV_FISHEYE", 1920, 1080)
    pinhole: pycolmap.Camera = seed_camera(2, "OPENCV", 1920, 1080)
    assert fisheye.has_prior_focal_length and fisheye.focal_length_x == pytest.approx(1920 / math.pi)
    assert not pinhole.has_prior_focal_length and pinhole.focal_length_x == pytest.approx(1.2 * 1920)
    assert (fisheye.principal_point_x, fisheye.principal_point_y) == (960.0, 540.0)


def _synthetic_database(path: Path, num_rig_frames: int = 24, overlap: int = 3, num_points: int = 1500) -> None:
    """A room of points seen by the true rig along a turning walk; the database holds the seeds, not the truth."""
    rng: np.random.Generator = np.random.default_rng(7)
    walls: np.ndarray = rng.uniform([-4.0, -1.5, -4.0], [4.0, 1.5, 4.0], size=(num_points, 3))
    side: np.ndarray = rng.integers(0, 4, size=len(walls))
    walls[side == 0, 0], walls[side == 1, 0], walls[side == 2, 2], walls[side == 3, 2] = -4.0, 4.0, -4.0, 4.0
    true_cameras: list[pycolmap.Camera] = [
        pycolmap.Camera(model="OPENCV_FISHEYE", width=WIDTH, height=HEIGHT, params=params) for params in TRUE_PARAMS
    ]
    with pycolmap.Database.open(path) as database:
        for camera_id in range(1, len(NAMES) + 1):
            database.write_camera(seed_camera(camera_id, "OPENCV_FISHEYE", WIDTH, HEIGHT), True)
        image_ids: list[list[int | None]] = []
        observed: dict[int, dict[int, int]] = {}  # image id -> point index -> keypoint index
        for frame in range(num_rig_frames):
            world_T_rig: np.ndarray = np.eye(4)
            world_T_rig[:3, :3] = Rotation.from_euler("yx", [3.0 * frame, 8.0 * math.sin(frame / 5.0)], degrees=True).as_matrix()
            world_T_rig[:3, 3] = [0.04 * frame - 0.8, 0.0, 0.5 * math.sin(frame / 7.0)]
            row: list[int | None] = []
            for camera_index, (name, camera) in enumerate(zip(NAMES, true_cameras, strict=True)):
                cam_T_world: np.ndarray = np.linalg.inv(world_T_rig @ TRUE_RIG_T_CAM[camera_index])
                in_camera: np.ndarray = walls @ cam_T_world[:3, :3].T + cam_T_world[:3, 3]
                ahead: np.ndarray = np.arccos(np.clip(in_camera[:, 2] / np.linalg.norm(in_camera, axis=1), -1.0, 1.0)) < np.radians(85.0)
                pixels: np.ndarray = np.full((len(walls), 2), -1.0)
                pixels[ahead] = camera.img_from_cam(in_camera[ahead])
                visible: np.ndarray = np.flatnonzero(
                    ahead & (pixels[:, 0] >= 0) & (pixels[:, 0] < WIDTH) & (pixels[:, 1] >= 0) & (pixels[:, 1] < HEIGHT)
                )
                image_id: int = database.write_image(pycolmap.Image(name=f"{name}/{frame:06d}", camera_id=camera_index + 1))
                keypoints: np.ndarray = (pixels[visible] + rng.normal(0.0, 0.3, size=(len(visible), 2))).astype(np.float32)
                database.write_keypoints(image_id, keypoints)
                observed[image_id] = {int(point): index for index, point in enumerate(visible)}
                row.append(image_id)
            image_ids.append(row)
        seeds: dict[int, pycolmap.Camera] = {camera.camera_id: camera for camera in database.read_all_cameras()}
        camera_of: dict[int, int] = {image.image_id: image.camera_id for image in database.read_all_images()}
        verification: pycolmap.TwoViewGeometryOptions = pycolmap.TwoViewGeometryOptions()
        for first, second in rig_window_pairs(image_ids, overlap):
            common: list[int] = sorted(observed[first].keys() & observed[second].keys())
            if len(common) < 30:
                continue
            matches: np.ndarray = np.asarray([[observed[first][point], observed[second][point]] for point in common], dtype=np.uint32)
            database.write_matches(first, second, matches)
            geometry: pycolmap.TwoViewGeometry = pycolmap.estimate_two_view_geometry(
                seeds[camera_of[first]],
                database.read_keypoints(first)[:, :2].astype(np.float64),
                seeds[camera_of[second]],
                database.read_keypoints(second)[:, :2].astype(np.float64),
                matches,
                verification,
            )
            database.write_two_view_geometry(first, second, geometry)


def test_solve_rig_recovers_a_known_fisheye_rig_from_seeds(tmp_path: Path) -> None:
    database_path: Path = tmp_path / "database.db"
    _synthetic_database(database_path)
    model: pycolmap.Reconstruction = solve_rig(database_path, NAMES, tmp_path)
    cameras = cameras_from_rig(
        model,
        {name: index + 1 for index, name in enumerate(NAMES)},
        {name: f"/world/rig_00/cam_{index:02d}" for index, name in enumerate(NAMES)},
        "OPENCV_FISHEYE",
    )
    assert model.num_reg_images() == 3 * 24
    np.testing.assert_array_equal(cameras[0].rig_T_cam, np.eye(4))
    scale: float = float(
        np.linalg.norm(TRUE_RIG_T_CAM[1][:3, 3] - TRUE_RIG_T_CAM[2][:3, 3])
        / np.linalg.norm(cameras[1].rig_T_cam[:3, 3] - cameras[2].rig_T_cam[:3, 3])
    )
    for camera, truth, params in zip(cameras, TRUE_RIG_T_CAM, TRUE_PARAMS, strict=True):
        rotation_deg: float = float(np.degrees(Rotation.from_matrix(truth[:3, :3].T @ camera.rig_T_cam[:3, :3]).magnitude()))
        assert rotation_deg < 0.1, camera.name
        np.testing.assert_allclose(camera.rig_T_cam[:3, 3] * scale, truth[:3, 3], atol=0.005, err_msg=camera.name)
        np.testing.assert_allclose(camera.k_matrix[[0, 1, 0, 1], [0, 1, 2, 2]], params[:4], rtol=0.005, err_msg=camera.name)


def test_the_cli_calibrates_a_segment() -> None:
    import tyro

    from colsfm.apis.run import CalibrateCommand, Command

    command = tyro.cli(Command, args=["calibrate", "--source.dataset", "robocap", "--source.segment", "seg", "--options.camera-model", "OPENCV"])
    assert isinstance(command, CalibrateCommand)
    assert (command.source.segment, command.options.camera_model, command.options.num_rig_frames) == ("seg", "OPENCV", 300)


def test_the_window_is_on_the_reference_packets_and_names_one_rig_frame_alike() -> None:
    from simplecv.catalog_video_reader import StreamIndex

    from colsfm.calibrate import window_streams

    def stream(entity: str, t_ns: list[int]) -> StreamIndex:
        times = np.asarray(t_ns, dtype=np.int64).astype("timedelta64[ns]")
        return StreamIndex(entity=entity, times=times, keyframes=np.zeros(len(t_ns), dtype=bool), codec="h264")

    reference: StreamIndex = stream("/world/rig_00/cam_00/pinhole/video", [0, 100, 200, 300, 400])
    other: StreamIndex = stream("/world/rig_00/cam_01/pinhole/video", [2, 101, 260, 302, 399])  # 260 is 60 ns off its rig frame
    entities: dict[str, str] = {"a": "/world/rig_00/cam_00", "b": "/world/rig_00/cam_01"}
    a, b = window_streams(entities, (reference, other), num_rig_frames=9, tolerance_ns=5, trims_s=(0.0, 0.0))
    assert (a.names, a.t_ns) == (["a/000000", "a/000001", "a/000002", "a/000003", "a/000004"], [0, 100, 200, 300, 400])
    assert (b.names, b.t_ns) == (["b/000000", "b/000001", "b/000003", "b/000004"], [2, 101, 302, 399])
    assert a.entity == "world/rig_00/cam_00/pinhole/video"


@pytest.mark.parametrize(
    ("other_t_ns", "tolerance_ns", "trims_s", "message"),
    [
        ([0, 100, 200, 300, 400], 5, (0.0, 1.0), "leave no frame"),
        ([50, 150, 250, 350, 450], 5, (0.0, 0.0), "b has no packet"),
        ([0, 100, 200, 300, 400], 100, (0.0, 0.0), "nearest of two rig frames"),
    ],
)
def test_a_window_that_cannot_be_sampled_is_refused_before_anything_is_written(
    other_t_ns: list[int], tolerance_ns: int, trims_s: tuple[float, float], message: str
) -> None:
    from simplecv.catalog_video_reader import StreamIndex

    from colsfm.calibrate import window_streams

    def stream(entity: str, t_ns: list[int]) -> StreamIndex:
        return StreamIndex(
            entity=entity, times=np.asarray(t_ns, dtype=np.int64).astype("timedelta64[ns]"), keyframes=np.zeros(len(t_ns), dtype=bool), codec="h264"
        )

    reference: StreamIndex = stream("/world/rig_00/cam_00/pinhole/video", [0, 100, 200, 300, 400])
    sparse: StreamIndex = stream("/world/rig_00/cam_01/pinhole/video", other_t_ns if message != "nearest of two rig frames" else [0, 300])
    with pytest.raises(ValueError, match=message):
        window_streams({"a": "/world/rig_00/cam_00", "b": "/world/rig_00/cam_01"}, (reference, sparse), 5, tolerance_ns, trims_s)
