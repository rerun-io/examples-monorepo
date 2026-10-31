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

from dataforge import schema, transports
from dataforge.quest_joint_poses import quest_hand_path
from dataforge.datasets.selfcap import (
    DEVICE_MESHES,
    log_quest_hand_poses,
    read_hand_poses,
    write_device_asset,
    write_coco133_layer,
    CameraCalibration,
    CameraPane,
    EpisodePlan,
    HeadPoses,
    SelfcapConfig,
    SelfcapDataset,
    build_blueprint,
    build_episode_plan,
    exo_devices,
    read_calibration,
    read_head_poses,
    read_imu_csv,
    read_session_task,
)
from dataforge.datasets.selfcap_raw import OAK_MOUNT_OTHER, OAK_MOUNTS, assumed_phone_lens, quest_mesh_frame, FactoryCalibration, nominal_oak_mount, oak_stereo, quest_left_T_right
from dataforge.identity import SequenceIdentity
from dataforge.records import read_json
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


def test_download_fetches_the_pinned_cut_tree_then_verifies_it(corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fetches: list[tuple[str, list[str], Path, str]] = []
    monkeypatch.setattr(transports, "hf_fetch", lambda repo, *, allow_patterns, local_dir, revision: fetches.append((repo, list(allow_patterns), local_dir, revision)))
    SelfcapConfig(root=corpus).setup().download()
    assert fetches == [("pablovela5620/selfcap-hard26-raw", ["cut/**"], corpus, "84a01e79ae35aee923e6dd5dedc91cd695196445")]
    empty_root: Path = tmp_path / "empty"
    empty_root.mkdir()
    with pytest.raises(FileNotFoundError, match="missing"):
        SelfcapConfig(root=empty_root).setup().download()


def test_config_refuses_a_floating_revision() -> None:
    with pytest.raises(ValueError, match="full commit sha"):
        SelfcapConfig(revision="main")


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


def test_selfcap_layouts_open_on_the_aligned_clock() -> None:
    # video_time equals capture_time at ingest and is the aligned clock after selfcap-align, so it is the right
    # timeline at every stage; left unset, the viewer picked capture_time and showed the as-cut streams.
    panes = (CameraPane("Dragon", 0, 0, "exo"), CameraPane("oak rgb", 4, 0, "ego"), CameraPane("quest left", 5, 0, "quest"))
    for blueprint in (build_blueprint(panes, ego_rig=4), SelfcapDataset(SelfcapConfig()).default_blueprint()):
        assert blueprint.time_panel is not None
        assert str(blueprint.time_panel.timeline) == schema.TIMELINE


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
        assert hidden["/world/gt/sapiens2"] == rrb.EntityBehavior(visible=False)
        assert hidden[f"{pinhole}/sapiens2"] == rrb.EntityBehavior(visible=False)
        assert f"{pinhole}/camera_centers" not in hidden
        assert f"{pinhole}/exo_landmarks" not in hidden
        assert f"{pinhole}/**" in contents
        assert "/world/gt/**" in contents and "/world/pred/**" in contents
        assert ("/world/skeleton/**" in contents) == (rig != 4)
        if rig != 4:
            assert hidden["/world/pred/soma_x/mesh"] == rrb.EntityBehavior(visible=False)
    world = views["World (Quest frame, calibrated)"]
    assert world.visualizer_overrides["/world/pred"] == rrb.EntityBehavior(visible=False)
    assert world.visualizer_overrides["/world/gt"] == rrb.EntityBehavior(visible=False)
    for rig, cam in cameras.values():
        assert world.visualizer_overrides[f"{schema.pinhole_path(rig, cam)}/skeleton"] == rrb.EntityBehavior(visible=False)
        assert world.visualizer_overrides[f"{schema.pinhole_path(rig, cam)}/camera_centers"] == rrb.EntityBehavior(visible=False)
    assert "/world/pred/soma_x/mesh" not in world.visualizer_overrides
    dev_views = blueprint_views(SelfcapDataset(SelfcapConfig()).default_blueprint(show_soma_mesh=True))
    dev_world = next(view for view in dev_views if str(view.name) == "World (Quest frame, calibrated)")
    assert dev_world.visualizer_overrides["/world/pred/soma_x/mesh"] == rrb.EntityBehavior(visible=True)
    for view in dev_views:
        if str(view.name).startswith(("exo", "Quest")):
            assert view.visualizer_overrides["/world/pred/soma_x/mesh"] == rrb.EntityBehavior(visible=True)


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

    base: Path = convert_fixture_episode(tmp_path, monkeypatch)
    seen: set[str] = set()
    pose_rows: dict[str, list[int]] = {}
    sample_rows: int = 0
    for layer in (base, base.parent.parent / "gt" / base.name):
        reader: rrc.RrdReader = rrc.RrdReader(layer)
        for chunk in reader.stream(store=reader.recordings()[0]):
            if chunk.is_static:
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
            if "InstancePoses3D:translations" in batch.schema.names:
                pose_rows[str(chunk.entity_path)] = [len(row) for row in batch.column("InstancePoses3D:translations").to_pylist()]
            if str(chunk.entity_path) == schema.coco133_xyz_path():
                assert layer.parent.name == "gt"
                assert batch.column("capture_time").cast(pa.int64()).to_pylist() == [17181303, 27181303]
                points = np.asarray(batch.column("Points3D:positions").to_pylist(), dtype=np.float32)
                assert np.isnan(points[0, 112:]).all()
                assert np.isnan(points[1, 91:112]).all()
    assert pose_rows == {quest_hand_path("left"): [26], quest_hand_path("right"): [26]}  # every joint, palm and metacarpals too
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


def test_table_blueprint_shows_scene_and_one_exo_video_for_both_rig_layouts() -> None:
    views: list[rrb.View] = blueprint_views(SelfcapConfig().setup().table_blueprint())
    assert len(views) == 2
    scene, camera = views
    assert isinstance(scene, rrb.Spatial3DView)
    assert str(scene.origin) == "/world"
    assert isinstance(camera, rrb.Spatial2DView)
    assert str(camera.origin) == schema.pinhole_path(0, 0)
    scene_contents: list[str] = cast(list[str], scene.contents)
    camera_contents: list[str] = cast(list[str], camera.contents)
    for phone_count in (3, 4):
        slots = [(rig, 0) for rig in range(phone_count)] + [(phone_count, cam) for cam in range(3)] + [(phone_count + 1, cam) for cam in range(2)]
        for rig, cam in slots:
            assert f"- {schema.video_path(rig, cam)}/**" in scene_contents
    assert "/world/skeleton/**" in camera_contents
    assert "/world/pred/soma_x/mesh/**" in camera_contents
    assert "/world/pred/soma_x/mesh/**" in scene_contents
    assert "/world/gt/**" not in scene_contents


def test_table_fields_keep_the_capture_and_quality_summary_compact() -> None:
    fields = SelfcapConfig().setup().table_fields()
    assert [(field.column, field.name) for field in fields.cards] == [
        ("property:summary:quality_rank", "Quality rank"),
        ("property:calib:session", "Capture / session"),
        ("property:calib:exo_cameras", "Phones"),
        ("property:calib:calib_exo_refined", "Refined"),
        ("property:soma_quality:status", "SOMA quality"),
    ]
    assert len(fields.table) == 9
    assert {field.column for field in fields.cards} <= {field.column for field in fields.table}
    assert {field.name for field in fields.table} >= {"OAK posed", "OAK rotation", "Frames", "Session UUID"}


def factory_extrinsic(kind: str, rotation: list[list[float]], translation: list[float]) -> dict:
    return {"type": kind, "from": "a", "to": "b", "R": rotation, "T": translation}


def test_iphone_fallback_lens_is_not_measured(tmp_path: Path) -> None:
    # An iPhone writes a 700 px fallback with available=false; the base must not turn it into a Pinhole.
    document = json.loads(calibration_document(("center",)))
    document["intrinsics"][0]["lens_intrinsics"]["available"] = False
    (tmp_path / "calibration.json").write_text(json.dumps(document))
    assert not read_calibration(tmp_path / "calibration.json")["center"].available


def test_oak_stereo_is_rgb_relative_in_metres(tmp_path: Path) -> None:
    document = json.loads(calibration_document(("center", "left", "right")))
    identity = np.eye(3).tolist()
    document["extrinsics"] = [factory_extrinsic("left_to_right", identity, [7.5, 0.0, 0.0]),
                              factory_extrinsic("left_to_rgb", identity, [3.75, 0.0, 0.0])]
    (tmp_path / "calibration.json").write_text(json.dumps(document))
    stereo = oak_stereo(read_json(tmp_path / "calibration.json", FactoryCalibration))
    np.testing.assert_allclose(stereo["rgb_T_left"][:3, 3], [-0.0375, 0.0, 0.0])
    np.testing.assert_allclose(stereo["rgb_T_right"][:3, 3], [0.0375, 0.0, 0.0])


def test_quest_left_T_right_needs_a_healthy_headset(tmp_path: Path) -> None:
    document = json.loads(calibration_document(("left", "right")))
    identity = np.eye(3).tolist()
    document["extrinsics"] = [factory_extrinsic("device_to_left", identity, [-0.032, 0.0, 0.0]),
                              factory_extrinsic("device_to_right", identity, [0.032, 0.0, 0.0])]
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(document))
    np.testing.assert_allclose(quest_left_T_right(read_json(path, FactoryCalibration))[:3, 3], [0.064, 0.0, 0.0])
    document["intrinsics"][1]["lens_intrinsics"]["available"] = False  # faulty right camera: fixed later, not here
    path.write_text(json.dumps(document))
    assert quest_left_T_right(read_json(path, FactoryCalibration)) is None


def test_nominal_oak_mount_follows_the_oak_and_the_headset_layout(tmp_path: Path) -> None:
    document = json.loads(calibration_document(("left", "right")))
    identity = np.eye(3).tolist()
    document["extrinsics"] = [factory_extrinsic("device_to_left", identity, [-0.032, 0.0, 0.0]),
                              factory_extrinsic("device_to_right", identity, [0.032, 0.0, 0.0])]
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(document))
    healthy = read_json(path, FactoryCalibration)
    document["intrinsics"][1]["camera_id"] = "50"
    document["intrinsics"][1]["lens_intrinsics"]["available"] = False  # broken left camera: camera 50 borrowed
    path.write_text(json.dumps(document))
    borrowed = read_json(path, FactoryCalibration)
    known = FactoryCalibration(device_id=next(iter(OAK_MOUNTS)), intrinsics=())
    unseen = FactoryCalibration(device_id="OAK - never fitted", intrinsics=())

    assert nominal_oak_mount(known, healthy) is OAK_MOUNTS[known.device_id][0]
    assert nominal_oak_mount(known, borrowed) is OAK_MOUNTS[known.device_id][1]
    assert nominal_oak_mount(unseen, healthy) is OAK_MOUNT_OTHER[0]
    assert nominal_oak_mount(unseen, borrowed) is OAK_MOUNT_OTHER[1]


def test_quest_principal_point_is_in_video_pixels(tmp_path: Path) -> None:
    document = json.loads(calibration_document(("left", "right")))
    for entry in document["intrinsics"]:
        entry["capture_resolution"] = {"width": 1280, "height": 960}
        entry["sensor_info"] = {"pixel_array_size": {"width": 1280, "height": 1280}}
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(document))
    raw_cy = document["intrinsics"][0]["lens_intrinsics"]["principal_point_y"]

    assert read_calibration(path)["left"].principal_point_xy[1] == raw_cy - 160.0  # centred 1280x960 crop of a 1280x1280 sensor


def test_assumed_phone_lens_is_a_centred_undistorted_pinhole() -> None:
    lens = assumed_phone_lens(1920, 1080)

    assert lens.focal_length_xy.tolist() == [960.0, 960.0]  # 90 deg horizontal field of view
    assert lens.principal_point_xy.tolist() == [960.0, 540.0]
    assert lens.distortion_coefficients is None and not lens.available


def test_quest_mesh_turns_with_a_camera_50_eye_frame(tmp_path: Path) -> None:
    document = json.loads(calibration_document(("left", "right")))
    identity = np.eye(3).tolist()
    document["extrinsics"] = [factory_extrinsic("device_to_left", identity, [-0.032, 0.0, 0.0]),
                              factory_extrinsic("device_to_right", identity, [0.032, 0.0, 0.0])]
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(document))
    healthy = read_json(path, FactoryCalibration)
    document["intrinsics"][1]["camera_id"] = "50"
    document["intrinsics"][1]["lens_intrinsics"]["available"] = False
    path.write_text(json.dumps(document))
    borrowed = read_json(path, FactoryCalibration)
    oak = FactoryCalibration(device_id="OAK - never fitted", intrinsics=())

    turned = quest_mesh_frame(oak, borrowed)
    angle = np.degrees(np.arccos((np.trace(turned[:3, :3]) - 1) / 2))

    assert np.array_equal(quest_mesh_frame(oak, healthy), np.eye(4))
    assert 170 < angle < 180  # the camera-50 eye frame sits about 175 deg from the healthy one


def test_device_meshes_are_well_formed() -> None:
    for name in ("oak_d_pro_w", "quest3s"):
        mesh = np.load(DEVICE_MESHES / f"{name}.npz")
        positions, triangles = mesh["positions"], mesh["triangles"]
        assert positions.shape[1] == 3 and triangles.shape[1] == 3
        assert triangles.max() < len(positions)
        assert len(mesh["normals"]) == len(mesh["colors"]) == len(positions)
        assert np.ptp(positions, axis=0).max() < 0.4  # metres: a headset, not millimetres


def hand_csv(path: Path, width: int, rows: list[tuple[int, float]]) -> None:
    """A Quest hand CSV in the device's layout; a row with value 0 is the all-zero "not tracked" row."""
    axes = ("x", "y", "z", "qx", "qy", "qz", "qw")[:width]
    names = ("palm", "wrist", *(f"joint{j}" for j in range(24)))
    header = "ts_ns," + ",".join(f"{name}_{axis}" for name in names for axis in axes)
    lines = [f"{stamp}," + ",".join(f"{value * (k + 1) / 7:.6f}" for k in range(26 * width)) for stamp, value in rows]
    path.write_text(header + "\n" + "\n".join(lines) + "\n")


@pytest.mark.parametrize("width", [7, 3])
def test_quest_hand_poses_round_trip_through_the_base(tmp_path: Path, width: int) -> None:
    from dataforge import writing
    from dataforge.quest_joint_poses import read_quest_joint_poses

    hand_csv(tmp_path / "left.csv", width, [(1_000_000, 0.123456), (21_000_000, 0.0), (41_000_000, -0.654321)])
    raw = read_quest_joint_poses(tmp_path / "left.csv")
    with writing.recording_to(tmp_path / "base.rrd", recording_id="r") as recording:
        log_quest_hand_poses(recording, {"left": raw}, shift_ns=7_000_000)

    hands, shift = read_hand_poses(tmp_path / "base.rrd")
    back = hands["left"]

    assert shift == 7_000_000
    assert np.array_equal(back.times_ns, raw.times_ns)
    assert np.array_equal(back.positions, raw.positions, equal_nan=True)  # untracked row stays untracked
    assert np.isnan(back.positions[1]).all() and np.isfinite(back.positions[[0, 2]]).all()
    assert back.joint_names == raw.joint_names and back.joint_names[:2] == ("palm", "wrist")
    if width == 7:
        assert np.array_equal(back.orientations, raw.orientations, equal_nan=True)
    else:
        assert back.orientations is None


def test_gt_layer_from_a_base_is_the_coco133_view_of_its_hands(tmp_path: Path) -> None:
    import pyarrow as pa
    import rerun.chunk as rrc

    from dataforge import writing
    from dataforge.datasets.selfcap_raw import coco133_hands
    from dataforge.quest_joint_poses import read_quest_joint_poses

    hands = {}
    for side, stamp in (("left", 1_000_000), ("right", 2_000_000)):
        hand_csv(tmp_path / f"{side}.csv", 7, [(stamp, 0.25)])
        hands[side] = read_quest_joint_poses(tmp_path / f"{side}.csv")
    with writing.recording_to(tmp_path / "base.rrd", recording_id="r") as recording:
        log_quest_hand_poses(recording, hands, shift_ns=3_000_000)
    write_coco133_layer(*read_hand_poses(tmp_path / "base.rrd"), tmp_path / "gt.rrd", "r")

    table = rrc.ChunkStore.from_chunks(list(rrc.RrdReader(tmp_path / "gt.rrd").stream())).reader(index=schema.CAPTURE_TIMELINE, contents=schema.coco133_xyz_path()).to_arrow_table()
    table = table.filter(pa.compute.is_valid(table[f"{schema.coco133_xyz_path()}:Points3D:positions"]))
    points = np.asarray(table[f"{schema.coco133_xyz_path()}:Points3D:positions"].to_pylist(), dtype=np.float32)

    expected = coco133_hands(hands)
    assert table[schema.CAPTURE_TIMELINE].cast(pa.int64()).to_pylist() == expected.times_ns.tolist()
    assert (table[schema.TIMELINE].cast(pa.int64()).to_numpy() - expected.times_ns).tolist() == [3_000_000, 3_000_000]
    assert np.array_equal(points, expected.positions_xyz, equal_nan=True)


def test_device_asset_is_a_static_mesh_on_the_rig(tmp_path: Path) -> None:
    import rerun.chunk as rrc

    write_device_asset(tmp_path / "oak.rrd", "oak", 3)
    chunks = list(rrc.RrdReader(tmp_path / "oak.rrd").stream())

    assert {str(c.entity_path) for c in chunks if "Mesh3D:vertex_positions" in c.to_record_batch().schema.names} == {"/world/rig_03/device"}
    assert all(c.is_static for c in chunks if str(c.entity_path) == "/world/rig_03/device")
