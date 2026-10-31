"""SelfCap discovery and raw-csv parsing against synthetic fixtures (no corpus needed)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import rerun.blueprint as rrb
from jaxtyping import Float32, Float64
from numpy import ndarray

from dataforge import schema
from dataforge.datasets.selfcap import (
    CameraCalibration,
    EpisodePlan,
    HeadPoses,
    SelfcapConfig,
    SelfcapDataset,
    build_episode_plan,
    exo_devices,
    read_calibration,
    read_head_poses,
    read_imu_csv,
    read_session_task,
)
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import ImuChannel
from dataforge.writing import blueprint_views

SESSION_A: str = "25aeb66a-cf5a-44a3-845f-62606cefabe7"
"""Session uuid whose first eight characters become the identity's middle part."""
SESSION_B: str = "1dbe8c49-5f09-4b58-875e-b0a3315d9d4a"
"""Second session, on a different capture day."""
EXO_DEVICES: tuple[str, ...] = ("Dragon-413A7CC1", "Tiger-FE57138F", "Titanium-B00700F9", "Wolf-387E89E1")
"""Exo device directory names, which also form the video stems."""
UNREADABLE: int = 0o000
"""The mode the NAS sync leaves on much of the corpus."""


def make_episode(episode_dir: Path, *, unreadable: str | None = None) -> None:
    """Create every file the converter requires, optionally crippling one of them.

    Args:
        episode_dir: Target ``episodes/episode-<NNN>`` directory.
        unreadable: Path relative to ``episode_dir`` to chmod 000, mimicking the NAS sync.
    """
    (episode_dir / "ego").mkdir(parents=True)
    for stream in ("rgb", "left", "right"):
        (episode_dir / "ego" / f"{stream}.mp4").touch()
        (episode_dir / "ego" / f"{stream}.csv").write_text("ts_ns,frame_idx\n0,10\n")
    (episode_dir / "ego" / "imu.csv").touch()
    (episode_dir / "ego" / "calibration.json").touch()
    (episode_dir / "quest").mkdir()
    for stream in ("left", "right"):
        (episode_dir / "quest" / f"{stream}.mp4").touch()
    (episode_dir / "quest" / "head_pose.csv").touch()
    (episode_dir / "quest" / "calibration.json").touch()
    for device in EXO_DEVICES:
        (episode_dir / "exo" / device).mkdir(parents=True)
        (episode_dir / "exo" / device / f"{device}.mp4").touch()
        (episode_dir / "exo" / device / "calibration.json").touch()
    # The prior-art recording that dataforge deliberately rebuilds from raw.
    (episode_dir / f"{episode_dir.name}.rrd").touch()
    if unreadable is not None:
        (episode_dir / unreadable).chmod(UNREADABLE)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A fake SelfCap root: two days, one session each, one crippled episode."""
    session_a: Path = tmp_path / "cut" / "2025-12-20" / SESSION_A
    make_episode(session_a / "episodes" / "episode-001")
    make_episode(session_a / "episodes" / "episode-009")
    (session_a / "metadata.json").write_text(json.dumps({"task": "hotdog_in_ricecooker", "collector_name": "Adil", "location": {}}))

    session_b: Path = tmp_path / "cut" / "2025-12-21" / SESSION_B
    make_episode(session_b / "episodes" / "episode-002")
    make_episode(session_b / "episodes" / "episode-003", unreadable="quest/head_pose.csv")
    (session_b / "metadata.json").write_text("{}")

    # Loose files beside the day directories (the corpus keeps manifests there).
    (tmp_path / "cut" / "manifest.json").write_text("{}")
    return tmp_path


def test_discover_skips_episodes_with_unreadable_files(corpus: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset: SelfcapDataset = SelfcapDataset(SelfcapConfig(root=corpus))
    discovered: list[tuple[SequenceIdentity, Path]] = dataset.discover()
    assert [identity.sequence_key for identity, _ in discovered] == [
        f"2025-12-20/{SESSION_A[:8]}/ep001",
        f"2025-12-20/{SESSION_A[:8]}/ep009",
        f"2025-12-21/{SESSION_B[:8]}/ep002",
    ]
    assert "skipped 1 episode" in capsys.readouterr().out
    # sequences() is derived from discover(), so the two can never disagree.
    assert dataset.sequences() == [identity for identity, _ in discovered]


def test_discover_pairs_each_identity_with_its_episode_directory(corpus: Path) -> None:
    dataset: SelfcapDataset = SelfcapDataset(SelfcapConfig(root=corpus))
    pair: tuple[SequenceIdentity, Path] = dataset.discover()[1]
    assert pair[0].recording_id == f"selfcap__2025-12-20__{SESSION_A[:8]}__ep009"
    assert pair[1] == corpus / "cut" / "2025-12-20" / SESSION_A / "episodes" / "episode-009"


def test_download_verifies_the_local_corpus(corpus: Path, tmp_path: Path) -> None:
    SelfcapConfig(root=corpus).setup().download()
    empty_root: Path = tmp_path / "empty"
    empty_root.mkdir()
    with pytest.raises(FileNotFoundError, match="missing"):
        SelfcapConfig(root=empty_root).setup().download()


def test_exo_devices_requires_video_and_calibration_in_one_directory(tmp_path: Path) -> None:
    make_episode(tmp_path / "episode-001")
    split_brain: Path = tmp_path / "episode-001" / "exo" / "multiCam-Ghost-1._multicam._tcp.local."
    split_brain.mkdir()
    (split_brain / f"{split_brain.name}.mp4").touch()  # no calibration.json beside it
    (tmp_path / "episode-001" / "exo" / EXO_DEVICES[0] / "calibration.json").chmod(UNREADABLE)
    assert exo_devices(tmp_path / "episode-001") == list(EXO_DEVICES[1:])


def test_session_task_degrades_to_empty_when_metadata_is_unreadable(corpus: Path) -> None:
    metadata_path: Path = corpus / "cut" / "2025-12-20" / SESSION_A / "metadata.json"
    assert read_session_task(metadata_path) == {"task": "hotdog_in_ricecooker", "collector": "Adil"}
    metadata_path.chmod(UNREADABLE)
    assert read_session_task(metadata_path) == {}
    assert read_session_task(corpus / "cut" / "2025-12-21" / SESSION_B / "metadata.json") == {}


def test_imu_csv_keeps_accel_and_gyro_and_drops_mag_and_rot(tmp_path: Path) -> None:
    csv_path: Path = tmp_path / "imu.csv"
    csv_path.write_text(
        "ts_ns,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z,rot_i,rot_j,rot_k\n"
        "0,0.71875,-4.95703125,-8.796875,-0.13671875,-0.15625,-0.07421875,19.25,10.1875,73.0625,-0.85,-0.45,0.068\n"
        "10035000,0.76953125,-5.01171875,-8.88671875,-0.111328125,-0.154296875,-0.0703125,20.3125,4.8125,69.75,-0.851,-0.455,0.068\n"
    )
    channels: tuple[ImuChannel, ImuChannel] = read_imu_csv(csv_path)
    gyro: ImuChannel = channels[0]
    accel: ImuChannel = channels[1]
    np.testing.assert_array_equal(gyro.times_ns, np.array([0, 10035000], dtype=np.int64))
    np.testing.assert_array_equal(accel.times_ns, np.array([0, 10035000], dtype=np.int64))
    expected_gyro: Float64[ndarray, "2 3"] = np.array([[-0.13671875, -0.15625, -0.07421875], [-0.111328125, -0.154296875, -0.0703125]])
    np.testing.assert_allclose(gyro.values_xyz, expected_gyro)
    expected_accel: Float64[ndarray, "2 3"] = np.array([[0.71875, -4.95703125, -8.796875], [0.76953125, -5.01171875, -8.88671875]])
    np.testing.assert_allclose(accel.values_xyz, expected_accel)


def test_head_poses_take_the_left_eye_track_as_xyzw_quaternions(tmp_path: Path) -> None:
    csv_path: Path = tmp_path / "head_pose.csv"
    csv_path.write_text(
        "ts_ns,left_pos_x,left_pos_y,left_pos_z,left_quat_x,left_quat_y,left_quat_z,left_quat_w,"
        "right_pos_x,right_pos_y,right_pos_z,right_quat_x,right_quat_y,right_quat_z,right_quat_w\n"
        "0,-0.224171,1.953459,0.502586,0.448053,0.358166,-0.794444,0.199559,-0.257063,1.953081,0.448476,0.446841,0.35412,-0.793884,0.211378\n"
        "17181303,-0.224091,1.953531,0.502348,0.448075,0.358003,-0.794497,0.199592,-0.256979,1.95314,0.448235,0.44686,0.353958,-0.793937,0.211411\n"
    )
    poses: HeadPoses = read_head_poses(csv_path)
    np.testing.assert_array_equal(poses.times_ns, np.array([0, 17181303], dtype=np.int64))
    expected_translations: Float32[ndarray, "2 3"] = np.array([[-0.224171, 1.953459, 0.502586], [-0.224091, 1.953531, 0.502348]], dtype=np.float32)
    np.testing.assert_allclose(poses.translations_xyz, expected_translations)
    # xyzw, not the wxyz the csv header might suggest to a careless reader.
    np.testing.assert_allclose(poses.quaternions_xyzw[0], np.array([0.448053, 0.358166, -0.794444, 0.199559], dtype=np.float32))
    assert np.isclose(np.linalg.norm(poses.quaternions_xyzw, axis=-1), 1.0, atol=1e-3).all()


def test_calibration_keys_on_positional_layout_and_maps_14_coefficients(tmp_path: Path) -> None:
    coefficients: list[float] = [5.53, 1.66, 8.6e-05, 1.9e-04, 0.0259, 5.9, 3.36, 0.226, 0.0, 0.0, 0.0, 0.0, -0.00158, 0.0067]
    calibration_path: Path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "device_id": "synthetic",
                "intrinsics": [
                    {
                        "camera_id": "CAM_C",
                        "positional_layout": "left",
                        "lens_intrinsics": {
                            "focal_length_x": 572.2,
                            "focal_length_y": 572.1,
                            "principal_point_x": 648.4,
                            "principal_point_y": 356.7,
                            "available": True,
                        },
                        "distortion": {"coefficients": coefficients, "available": True},
                        "capture_resolution": {"width": 1280, "height": 720},
                        "sensor_info": {"pixel_array_size": {"width": 1280, "height": 720}},
                    },
                    {
                        "camera_id": "avcapturedevice:5",
                        "positional_layout": "center",
                        "lens_intrinsics": {
                            "focal_length_x": 700.0,
                            "focal_length_y": 700.0,
                            "principal_point_x": 960.0,
                            "principal_point_y": 540.0,
                            "available": True,
                        },
                        "distortion": {"available": False},
                        "capture_resolution": {"width": 1920, "height": 1080},
                        "sensor_info": {"pixel_array_size": {"width": 1920, "height": 1080}},
                    },
                    {
                        "camera_id": "broken",
                        "positional_layout": "right",
                        "lens_intrinsics": {
                            "focal_length_x": None,
                            "focal_length_y": None,
                            "principal_point_x": None,
                            "principal_point_y": None,
                            "available": False,
                        },
                        "capture_resolution": {"width": 0, "height": 0},
                        "sensor_info": {"pixel_array_size": {"width": 0, "height": 0}},
                        "distortion": {},
                    },
                ],
                "extrinsics": [],
            }
        )
    )
    cameras: dict[str, CameraCalibration] = read_calibration(calibration_path)
    assert sorted(cameras) == ["center", "left"]  # the intrinsics-free entry is dropped
    left: CameraCalibration = cameras["left"]
    assert (left.width, left.height) == (1280, 720)
    np.testing.assert_allclose(left.focal_length_xy, np.array([572.2, 572.1]))
    np.testing.assert_allclose(left.principal_point_xy, np.array([648.4, 356.7]))
    assert left.distortion_coefficients is not None
    np.testing.assert_allclose(left.distortion_coefficients, np.asarray(coefficients))
    assert cameras["center"].distortion_coefficients is None


def test_default_blueprint_is_the_calibration_review_layout() -> None:
    # The guard: :51235's default comes from this method on every dataforge-register run, and a
    # catalog restore once fell back to an uncalibrated grid that hid the calib layer's overlays.
    blueprint: rrb.Blueprint = SelfcapConfig().setup().default_blueprint()
    views: dict[str, rrb.View] = {str(view.name): view for view in blueprint_views(blueprint)}
    exo: list[str] = [f"exo rig_{rig:02d}" for rig in range(4)]
    assert sorted(views) == sorted(["World (Quest frame, calibrated)", "OAK rgb", "OAK left", "OAK right", "Quest left", "Quest right", *exo])
    assert str(views["World (Quest frame, calibrated)"].origin) == "/world"
    cameras: dict[str, tuple[int, int]] = {
        **{name: (rig, 0) for rig, name in enumerate(exo)},
        "OAK rgb": (4, 0),
        "OAK left": (4, 1),
        "OAK right": (4, 2),
        "Quest left": (5, 0),
        "Quest right": (5, 1),
    }
    for name, (rig, cam) in cameras.items():
        assert str(views[name].origin) == schema.pinhole_path(rig, cam)
        contents: list[str] = cast(list[str], views[name].contents)  # the view keeps the list it was built with
        pinhole = schema.pinhole_path(rig, cam)
        hidden = views[name].visualizer_overrides
        assert hidden[f"{pinhole}/pred"] == rrb.EntityBehavior(visible=False)
        assert hidden[f"{pinhole}/coco133_uv_projected"] == rrb.EntityBehavior(visible=False)
        assert hidden[f"{pinhole}/quest_coco133_uv_projected"] == rrb.EntityBehavior(visible=False)
        assert hidden["/world/gt"] == rrb.EntityBehavior(visible=False)
        assert hidden["/world/pred"] == rrb.EntityBehavior(visible=False)
        assert f"{pinhole}/**" in contents
        assert "/world/gt/**" in contents and "/world/pred/**" in contents
        assert ("/world/skeleton" in contents) == name.startswith("Quest")
    world = views["World (Quest frame, calibrated)"]
    assert world.visualizer_overrides["/world/pred"] == rrb.EntityBehavior(visible=False)
    assert world.visualizer_overrides["/world/gt"] == rrb.EntityBehavior(visible=False)
    for rig, cam in cameras.values():
        assert world.visualizer_overrides[f"{schema.pinhole_path(rig, cam)}/skeleton"] == rrb.EntityBehavior(visible=False)
    assert "/world/pred/soma_x/mesh" not in world.visualizer_overrides
    dev_views = blueprint_views(SelfcapConfig().setup().default_blueprint(show_soma_mesh=True))
    dev_world = next(view for view in dev_views if str(view.name) == "World (Quest frame, calibrated)")
    assert dev_world.visualizer_overrides["/world/pred/soma_x/mesh"] == rrb.EntityBehavior(visible=True)


def calibration_document(layouts: tuple[str, ...]) -> str:
    """A vendor-shaped calibration document with one usable camera per layout."""
    return json.dumps(
        {
            "device_id": "synthetic",
            "intrinsics": [
                {
                    "camera_id": layout,
                    "positional_layout": layout,
                    "lens_intrinsics": {
                        "focal_length_x": 700.0,
                        "focal_length_y": 700.0,
                        "principal_point_x": 640.0,
                        "principal_point_y": 360.0,
                        "available": True,
                    },
                    "capture_resolution": {"width": 1280, "height": 720},
                    "sensor_info": {"pixel_array_size": {"width": 1280, "height": 720}},
                    "distortion": {},
                }
                for layout in layouts
            ],
            "extrinsics": [],
        }
    )


def test_episode_plan_drops_cameras_without_intrinsics(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # The bug this guards: panes built from readability alone described cameras the
    # converter then skipped, so the blueprint and num_cameras over-reported.
    episode_dir: Path = tmp_path / "episode-001"
    make_episode(episode_dir)
    for device in EXO_DEVICES[:-1]:
        (episode_dir / "exo" / device / "calibration.json").write_text(calibration_document(("center",)))
    (episode_dir / "exo" / EXO_DEVICES[-1] / "calibration.json").write_text(calibration_document(()))
    (episode_dir / "ego" / "calibration.json").write_text(calibration_document(("center", "left")))  # no "right"
    (episode_dir / "quest" / "calibration.json").write_text(calibration_document(("left", "right")))

    plan: EpisodePlan = build_episode_plan(episode_dir)
    assert (plan.ego_rig, plan.quest_rig) == (4, 5)
    assert [(pane.rig, pane.cam, pane.kind) for pane in plan.panes] == [
        (0, 0, "exo"),
        (1, 0, "exo"),
        (2, 0, "exo"),
        (4, 0, "ego"),
        (4, 1, "ego"),
        (5, 0, "quest"),
        (5, 1, "quest"),
    ]
    # The intrinsics-less exo device keeps its rig node, honestly declaring zero cameras.
    assert [len(rig.cameras) for rig in plan.rigs] == [1, 1, 1, 0, 2, 2]
    assert len(plan.panes) == sum(len(rig.cameras) for rig in plan.rigs)
    assert {rig.reference for rig in plan.rigs} == {"cam_00"}
    warnings: str = capsys.readouterr().out
    assert "skipping Wolf-387E89E1" in warnings
    assert "skipping ego right" in warnings
    # Ego cameras carry the native-rate csv metadata; exo/quest leave it off entirely.
    assert [(camera.native_frames, camera.native_duration_ns) for camera in plan.rigs[4].cameras] == [(1, 0), (1, 0)]
    assert [camera.first_frame_idx for camera in plan.rigs[4].cameras] == [10, 10]
    assert plan.rigs[0].cameras[0].first_frame_idx is None
    assert plan.rigs[0].cameras[0].native_frames is None


def test_quest_hands_preserve_raw_clock_and_map_both_layouts(tmp_path: Path) -> None:
    from dataforge.datasets.selfcap import QuestHands, read_quest_hands

    episode: Path = tmp_path / "episode-001"
    make_episode(episode)
    for side, width in (("left", 3), ("right", 7)):
        header: list[str] = ["ts_ns", *(f"joint_{j}_{axis}" for j in range(26) for axis in ("px", "py", "pz", "qx", "qy", "qz", "qw")[:width])]
        values: list[str] = ["17181303", *(str(j * 10 + axis) for j in range(26) for axis in range(width))]
        (episode / "quest" / f"{side}_hand_poses.csv").write_text(",".join(header) + "\n" + ",".join(values) + "\n")
    hands: QuestHands = read_quest_hands(episode / "quest")
    np.testing.assert_array_equal(hands.times_ns, [17181303])
    assert hands.positions_xyz.shape == (1, 133, 3)
    assert np.isnan(hands.positions_xyz[:, :91]).all()
    np.testing.assert_array_equal(hands.positions_xyz[0, [91, 92, 96, 111]], [[10, 11, 12], [20, 21, 22], [70, 71, 72], [250, 251, 252]])
    np.testing.assert_array_equal(hands.positions_xyz[0, 91:112], hands.positions_xyz[0, 112:133])


FIXTURES: Path = Path(__file__).parent / "fixtures"
"""48-frame AV1 clips: one starting at 0, one remuxed with a 40 ms container start (an mp4 edit list)."""


def convert_fixture_episode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, overrides: dict[str, Path] | None = None) -> Path:
    """Convert a synthetic episode whose nine videos are the AV1 fixture, and return its base rrd.

    Args:
        tmp_path: Scratch directory for the episode and the output tree.
        monkeypatch: Redirects the dataforge output root into ``tmp_path``.
        overrides: Clip per video path relative to the episode, replacing the default fixture there.
    """
    import shutil

    from dataforge import paths

    episode: Path = tmp_path / "episode-001"
    make_episode(episode)
    clip: Path = FIXTURES / "av1_48f_192x160.mp4"
    if not clip.is_file():
        pytest.skip(f"AV1 fixture absent: {clip}")
    for video in episode.rglob("*.mp4"):
        shutil.copyfile(clip, video)
    for relative, source in (overrides or {}).items():
        shutil.copyfile(source, episode / relative)
    for calibration in episode.rglob("calibration.json"):
        calibration.write_text(calibration_document(("center", "left", "right")))
    (episode / "ego/imu.csv").write_text("ts_ns,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z\n17181303,1,2,3,4,5,6\n")
    (episode / "quest/head_pose.csv").write_text(
        "ts_ns,left_pos_x,left_pos_y,left_pos_z,left_quat_x,left_quat_y,left_quat_z,left_quat_w\n17181303,1,2,3,0,0,0,1\n"
    )
    for side, stamp in (("left", 17181303), ("right", 27181303)):
        (episode / "quest" / f"{side}_hand_poses.csv").write_text(
            "ts_ns,"
            + ",".join(f"joint_{j}_{axis}" for j in range(26) for axis in ("px", "py", "pz"))
            + "\n"
            + str(stamp)
            + ","
            + ",".join(str(j / 100) for j in range(78))
            + "\n"
        )
    monkeypatch.setattr(paths, "output_root", lambda: tmp_path / "output")
    identity: SequenceIdentity = SequenceIdentity(dataset="selfcap", parts=("2025-12-20", SESSION_A[:8], "ep001"))
    return SelfcapConfig().setup().convert(identity, episode, force=False)


def test_convert_preserves_capture_time_on_every_temporal_chunk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pyarrow as pa
    import rerun.chunk as rrc

    reader: rrc.RrdReader = rrc.RrdReader(convert_fixture_episode(tmp_path, monkeypatch))
    seen: set[str] = set()
    hand_rows: int = 0
    sample_rows: int = 0
    annotated: bool = False
    for chunk in reader.stream(store=reader.recordings()[0]):
        if chunk.is_static:
            if str(chunk.entity_path) == "/world/gt":
                annotated = "AnnotationContext:context" in chunk.to_record_batch().schema.names
            continue
        batch: pa.RecordBatch = chunk.to_record_batch()
        assert "capture_time" in chunk.timeline_names, str(chunk.entity_path)
        assert batch.column("capture_time").equals(batch.column(schema.TIMELINE))
        field: pa.Field = batch.schema.field("capture_time")
        assert field.type == pa.duration("ns")
        assert field.metadata is not None
        assert field.metadata[b"rerun:kind"] == b"index"
        assert field.metadata[b"rerun:index_name"] == b"capture_time"
        assert field.metadata[b"rerun:is_sorted"] == b"true"
        seen.add(str(chunk.entity_path))
        if "VideoStream:sample" in batch.schema.names:
            sample_rows += batch.num_rows
        if str(chunk.entity_path) == schema.coco133_xyz_path():
            hand_rows += batch.num_rows
            assert batch.column("capture_time").cast(pa.int64()).to_pylist() == [17181303, 27181303]
            points = np.asarray(batch.column("Points3D:positions").to_pylist(), dtype=np.float32)
            assert np.isnan(points[0, 112:]).all()
            assert np.isnan(points[1, 91:112]).all()
    assert annotated
    assert hand_rows == 2
    assert sample_rows == 9 * 48  # video stored once, despite its two clocks
    assert {schema.gyro_path(4, 0), schema.accel_path(4, 0), schema.rig_path(5), schema.coco133_xyz_path()} <= seen


def test_convert_keeps_each_video_container_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Half the cut mp4s present their first frame one frame late; Mp4Reader alone would put it at 0."""
    import av
    import pyarrow as pa
    import rerun.chunk as rrc

    late: Path = FIXTURES / "av1_48f_192x160_start40ms.mp4"
    with av.open(str(late)) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        assert time_base is not None
        presented_ns: list[int] = sorted(
            round(packet.pts * time_base * 1_000_000_000) for packet in container.demux(stream) if packet.pts is not None
        )
    reader: rrc.RrdReader = rrc.RrdReader(convert_fixture_episode(tmp_path, monkeypatch, overrides={"ego/left.mp4": late}))
    sample_ns: dict[str, list[int]] = {}
    first_indices: dict[str, int] = {}
    for chunk in reader.stream(store=reader.recordings()[0]):
        batch: pa.RecordBatch = chunk.to_record_batch()
        if chunk.is_static and "first_frame_idx" in batch.schema.names:
            values = batch.column("first_frame_idx").to_pylist()
            if values and values[0]:
                first_indices[str(chunk.entity_path)] = values[0][0]
        if not chunk.is_static and "VideoStream:sample" in batch.schema.names:
            sample_ns.setdefault(str(chunk.entity_path), []).extend(batch.column("capture_time").cast(pa.int64()).to_pylist())
    assert presented_ns[0] == 40_039_062
    # Mp4Reader rounds each rebased PTS to ns before the start is added back: 1 ns of slack.
    np.testing.assert_allclose(sorted(sample_ns.pop(schema.video_path(4, 1))), presented_ns, rtol=0, atol=1)
    assert len(sample_ns) == 8
    assert all(min(times) == 0 for times in sample_ns.values())
    assert first_indices == {schema.cam_path(4, cam): 10 for cam in range(3)}


@pytest.mark.parametrize("contents", ["", "ts_ns,bad\n1,2\n", "ts_ns," + ",".join(f"j{i}" for i in range(78)) + "\n1,2\n"])
def test_hand_reader_names_malformed_source(tmp_path: Path, contents: str) -> None:
    from dataforge.datasets.selfcap import read_quest_hands

    (tmp_path / "left_hand_poses.csv").write_text(contents)
    with pytest.raises(ValueError, match="left_hand_poses.csv"):
        read_quest_hands(tmp_path)


def test_header_only_hands_are_empty_and_missing_hand_is_nan(tmp_path: Path) -> None:
    from dataforge.datasets.selfcap import QuestHands, read_quest_hands

    header: str = "ts_ns," + ",".join(f"j{i}" for i in range(78)) + "\n"
    (tmp_path / "left_hand_poses.csv").write_text(header)
    empty: QuestHands = read_quest_hands(tmp_path)
    assert empty.positions_xyz.shape == (0, 133, 3)
    (tmp_path / "left_hand_poses.csv").write_text(header + "123," + ",".join(["0"] * 78) + "\n")
    hands: QuestHands = read_quest_hands(tmp_path)
    assert np.isnan(hands.positions_xyz[0, 112:]).all()
    assert np.isnan(hands.positions_xyz[0, 91:112]).all()  # an all-zero row is the CSV's "not tracked", not a hand at the origin


def test_raw_calibration_chunks_keep_verbatim_device_documents(tmp_path: Path) -> None:
    from dataforge.datasets.selfcap import raw_calibration_chunks

    documents = {}
    for rig, folder in enumerate(("exo/phone-a", "exo/phone-b", "ego", "quest")):
        path = tmp_path / folder / "calibration.json"
        path.parent.mkdir(parents=True)
        content = (' {"vendor": "' + folder + '", "opaque": [1, 2]}\r\n').encode()
        path.write_bytes(content)
        documents[f"/world/rig_{rig:02d}"] = content.decode()
    chunks = raw_calibration_chunks(tmp_path, ["phone-a", "phone-b"])
    assert len(chunks) == 4
    for chunk in chunks:
        assert chunk.is_static
        assert chunk.to_record_batch().column("raw_calibration_json").to_pylist() == [[documents[str(chunk.entity_path)]]]


def test_convert_logs_factory_documents_on_every_rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from rerun.chunk import RrdReader

    reader = RrdReader(convert_fixture_episode(tmp_path, monkeypatch))
    values = {
        str(c.entity_path): c.to_record_batch().column("raw_calibration_json")[0].as_py()[0]
        for c in reader.stream()
        if "raw_calibration_json" in c.to_record_batch().schema.names
    }
    assert set(values) == {f"/world/rig_{i:02d}" for i in range(6)}
    assert all(document == calibration_document(("center", "left", "right")) for document in values.values())


@pytest.mark.parametrize("side", ["left", "right"])
def test_hand_csv_refuses_nonadjacent_duplicate_timestamps(tmp_path: Path, side: str) -> None:
    from dataforge.datasets.selfcap import read_quest_hands

    source = tmp_path / f"{side}_hand_poses.csv"
    header = "ts_ns," + ",".join(f"j{i}" for i in range(78)) + "\n"
    source.write_text(header + "".join(f"{stamp}," + ",".join(["1"] * 78) + "\n" for stamp in (17, 27, 17)))
    with pytest.raises(ValueError, match=f"{side}_hand_poses.csv.*duplicate hand timestamp"):
        read_quest_hands(tmp_path)


def test_episode_plan_keeps_empty_native_clock(tmp_path: Path) -> None:
    episode: Path = tmp_path / "episode-001"
    make_episode(episode)
    for path in episode.rglob("calibration.json"):
        path.write_text(calibration_document(("center", "left", "right")))
    (episode / "ego/rgb.csv").write_text("ts_ns,frame_idx\n")
    plan: EpisodePlan = build_episode_plan(episode)
    camera = plan.rigs[plan.ego_rig].cameras[0]
    assert (camera.native_frames, camera.native_duration_ns, camera.first_frame_idx) == (0, 0, None)


@pytest.mark.parametrize("document", ["not JSON", '{"intrinsics": [{}]}'])
def test_calibration_decode_failure_names_its_file(tmp_path: Path, document: str) -> None:
    source = tmp_path / "bad-calibration.json"
    source.write_text(document)
    with pytest.raises(ValueError) as error:
        read_calibration(source)
    assert str(source) in str(error.value)


def test_calibration_keeps_unknown_vendor_fields_and_rejects_short_distortion(tmp_path: Path) -> None:
    document = json.loads(calibration_document(("center",)))
    document["future_vendor_field"] = {"opaque": True}
    document["intrinsics"][0]["distortion"]["coefficients"] = [0.0] * 13
    source = tmp_path / "calibration.json"
    source.write_text(json.dumps(document))
    result = read_calibration(source)["center"]
    assert result.width == 1280
    assert result.distortion_coefficients is None
