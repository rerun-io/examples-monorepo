"""Assembly101 public source and dataset contracts, using synthetic inputs."""

import json
import shutil
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
from conftest import assert_raw_root_guarded, read_chunks
from simplecv.camera_parameters import BrownConradyDistortion, Fisheye62Parameters, KannalaBrandtDistortion, PinholeParameters
from simplecv.rerun_custom_types import PinholeWithDistortion

from dataforge import schema
from dataforge.actions import Segment, active_labels
from dataforge.datasets.assembly101 import Assembly101Config, Assembly101Dataset, Assembly101Source
from dataforge.datasets.assembly101_actions import Actions, read_actions
from dataforge.datasets.assembly101_calibration import Lens, camera_parameters, resolve_calibration
from dataforge.datasets.assembly101_layers import CameraVideo, camera_sources, read_poses, write_actions
from dataforge.datasets.assembly101_source import EXO_SERIALS, pose_path, read_confidence, read_hand_rows, read_pixels, read_transforms
from dataforge.hands import confidence_rule


def test_hand_rows_use_numeric_keys_and_shipped_confidence(tmp_path: Path) -> None:
    member = tmp_path / "landmarks.json"
    member.write_text(json.dumps({"10": {"0": [[1000, 2000, 3000]] * 21}, "2": {"1": [[4000, 5000, 6000]] * 21}}))
    confidence = {2: np.array([0.2, 0.7], dtype=np.float32), 10: np.array([0.0, 0.9], dtype=np.float32)}
    batches = list(read_hand_rows(member, confidence, dimensions=3, scale=0.001, frame_limit=None))
    frames, positions, scores = batches[0]
    assert frames.tolist() == [2, 10]
    np.testing.assert_allclose(positions[0, 112], [4, 5, 6])
    np.testing.assert_allclose(positions[1, 91], [1, 2, 3])
    assert scores[0, 112] == np.float32(0.7)
    assert scores[1, 91] == 0.0
    assert np.isnan(positions[0, 91]).all()
    assert scores[0, 91] == np.float32(0.2)
    _, normalized = confidence_rule(positions, scores)
    assert normalized[0, 91] == 0.0
    assert np.isnan(positions[:, 0]).all()


def test_calibration_parameters_preserve_coefficients_and_scale() -> None:
    lens = Lens("OpenCV", 1920, 1080, "C10095_rgb", 900.0, 900.0, 960.0, 540.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.4, 0.5)
    transform = np.eye(4)
    transform[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    transform[:3, 3] = [0.1, 0.2, 0.3]
    camera = camera_parameters(lens, transform, (1280, 720))
    assert isinstance(camera, PinholeParameters)
    np.testing.assert_array_equal(camera.extrinsics.world_T_cam, transform)
    assert camera.intrinsics.k_matrix is not None
    np.testing.assert_allclose(camera.intrinsics.k_matrix, [[600, 0, 640], [0, 600, 360], [0, 0, 1]])
    assert (camera.intrinsics.width, camera.intrinsics.height) == (1280, 720)
    assert camera.distortion == BrownConradyDistortion(0.1, 0.2, 0.4, 0.5, 0.3)
    fish = Lens("OVFishEye62", 636, 480, "84346135_mono10bit", 100.0, 101.0, 10.0, 20.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    camera = camera_parameters(fish, transform, (636, 480))
    assert isinstance(camera, Fisheye62Parameters)
    assert camera.distortion == KannalaBrandtDistortion(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    np.testing.assert_array_equal(camera.extrinsics.world_T_cam, transform)
    np.testing.assert_array_equal(camera.intrinsics.k_matrix, [[100, 0, 10], [0, 101, 20], [0, 0, 1]])


def test_actions_union_overlap_and_30_hz_boundaries(tmp_path: Path) -> None:
    fine = tmp_path / "fine-grained-annotations"
    fine.mkdir()
    (fine / "train.csv").write_text("video,start_frame,end_frame,action_cls\ns/cam1.mp4,30,90,hold\ns/cam2.mp4,30,90,hold\ns/cam1.mp4,60,120,turn\n")
    segments = read_actions(tmp_path, {"s"})["s"].fine
    frames, texts = active_labels(segments)
    assert frames.tolist() == [0, 30, 60, 90, 120]
    assert texts == ["", "hold", "hold\nturn", "turn", ""]


def test_discovery_sorted_serials_and_video_only(tmp_path: Path) -> None:
    for sequence in ("z", "a", "partial"):
        folder = tmp_path / "videos/av1-720-new" / sequence
        folder.mkdir(parents=True)
        for name in [
            *(f"{serial}_rgb_low" for serial in EXO_SERIALS),
            *(f"HMC_{serial}_mono10bit_low" for serial in (21179183, 21110305, 21176623, 21176875)),
        ]:
            if sequence != "partial" or name.startswith("C"):
                (folder / f"{name}.mp4").touch()
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests/sequences.csv").write_text("sequence_name,video_only\na,True\nz,True\npartial,True\n")
    dataset = Assembly101Config(root=tmp_path, annotations_root=tmp_path / "annotations").setup()
    assert [identity.recording_id for identity, _ in dataset.discover()] == ["assembly101__a", "assembly101__z"]
    cameras = camera_sources(tmp_path, "a")
    assert [cam.serial for cam in cameras] == [*EXO_SERIALS, "21110305", "21176623", "21176875", "21179183"]
    assert [(cam.rig, cam.cam) for cam in cameras] == [*((rig, 0) for rig in range(8)), *((8, cam) for cam in range(4))]
    assert dataset.targets(dataset.discover()[0][0]).keys() == {"base", "hand_pose", "actions"}


def test_calibration_sessions_do_not_borrow_across_days(tmp_path: Path) -> None:
    fixed = pose_path(tmp_path, "camera_extrinsics_fixed", "donor")
    fixed.parent.mkdir(parents=True)
    fixed.write_text(json.dumps({"C10095:rgb": np.eye(4).tolist()}))
    nimble = tmp_path / "assemblyhands-toolkit/calib/nimble_json_calib/donor.json"
    nimble.parent.mkdir(parents=True)
    camera = dict(
        DistortionModel="OpenCV",
        ImageSizeX=1920,
        ImageSizeY=1080,
        SerialNo="C10095_rgb",
        fx=900.0,
        fy=901.0,
        cx=960.0,
        cy=540.0,
        k1=0.0,
        k2=0.0,
        k3=0.0,
        k4=0.0,
        k5=0.0,
        k6=0.0,
        p1=0.0,
        p2=0.0,
    )
    ego = {**camera, "SerialNo": "84346135_mono10bit", "DistortionModel": "OVFishEye62", "ImageSizeX": 636, "ImageSizeY": 480}
    nimble.write_text(json.dumps([{"Camera": camera}, {"Camera": ego}]))
    transforms = read_transforms(fixed)
    calibrated = resolve_calibration(tmp_path, transforms)
    assert calibrated["C10095:rgb"].source == "nimble:donor"
    assert calibrated["C10095:rgb"].lens.fy == 901.0
    transforms["C10095:rgb"][0, 3] += 1.0
    uncovered = resolve_calibration(tmp_path, transforms)
    assert "C10095:rgb" not in uncovered
    assert "84346135:mono10bit" in uncovered
    fixed.unlink()
    with pytest.raises(ValueError, match="camera_extrinsics_fixed/donor.json"):
        resolve_calibration(tmp_path, transforms)


def test_shipped_pixels_scaled_once_all_views(tmp_path: Path) -> None:
    path = tmp_path / "pixels.json"
    path.write_text(json.dumps({"12": {"C10095:rgb": {"0": [[900, 600]] * 21}, "84346135:mono10bit": {"1": [[300, 200]] * 21}}}))
    rows = dict(read_pixels(path, {12: np.array([0.3, 0.8], dtype=np.float32)}, {"C10095:rgb": 2.0 / 3.0, "84346135:mono10bit": 1.0}, None))
    assert rows["C10095:rgb"][0].tolist() == [12]
    np.testing.assert_allclose(rows["C10095:rgb"][1][0, 91], [600, 400])
    np.testing.assert_allclose(rows["84346135:mono10bit"][1][0, 112], [300, 200])


@pytest.mark.integration
def test_video_only_conversion_remuxes_own_lengths_and_actions(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "raw"
    folder = root / "videos/av1-720-new/s"
    folder.mkdir(parents=True)
    fixture = Path(__file__).parent / "fixtures/av1_48f_192x160.mp4"
    if not fixture.is_file():
        pytest.skip(f"Synthetic AV1 fixture absent: {fixture}")
    manifest = root / "manifests/sequences.csv"
    manifest.parent.mkdir()
    manifest.write_text("sequence_name,video_only\ns,True\n")
    for serial in EXO_SERIALS:
        shutil.copyfile(fixture, folder / f"{serial}_rgb_low.mp4")
    for serial in (21110305, 21176623, 21176875, 21179183):
        shutil.copyfile(fixture, folder / f"HMC_{serial}_mono10bit_low.mp4")
    annotations = tmp_path / "annotations/fine-grained-annotations"
    annotations.mkdir(parents=True)
    (annotations / "train.csv").write_text("video,start_frame,end_frame,action_cls\ns/cam.mp4,1,3,turn\n")
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    dataset = Assembly101Config(root=root, annotations_root=annotations.parent, frame_limit=12).setup()
    identity, source = dataset.discover()[0]
    dataset.convert(identity, source, force=True)
    targets = dataset.targets(identity)
    assert not targets["hand_pose"].exists()
    assert targets["actions"].exists()
    chunks = read_chunks(targets["base"])
    assert not any(
        "Transform3D:translation" in c.to_record_batch().schema.names or "Pinhole:image_from_camera" in c.to_record_batch().schema.names
        for c in chunks
    )
    video = [c for c in chunks if "VideoStream:sample" in c.to_record_batch().schema.names]
    assert sum(c.num_rows for c in video) == 12 * 12
    for chunk in video:
        batch = chunk.to_record_batch()
        frames = np.array(batch.column("frame_index").to_pylist())
        np.testing.assert_array_equal(batch.column("video_time").cast(pa.int64()).to_numpy(), np.rint(frames / 60 * 1e9).astype(np.int64))
    assert any(c.entity_path == schema.video_path(8, 3) for c in video)
    assert dataset.timer.capture_s == 12 / 60
    targets["actions"].unlink()
    dataset.convert(identity, source, force=False)
    assert dataset.timer.capture_s == 12 / 60
    stamps = {layer: path.stat().st_mtime_ns for layer, path in targets.items() if path.exists()}
    dataset.convert(identity, source, force=False)
    assert stamps == {layer: path.stat().st_mtime_ns for layer, path in targets.items() if path.exists()}
    empty = Assembly101Config(root=root, annotations_root=tmp_path / "empty-annotations", frame_limit=12).setup()
    empty.convert(identity, source, force=True)
    assert not targets["actions"].exists()
    assert not targets["hand_pose"].exists()


@pytest.mark.parametrize("video_only", [None, "False", "True", "missing-row"])
def test_manifest_prevents_missing_poses_becoming_video_only(tmp_path: Path, video_only: str | None) -> None:
    folder = tmp_path / "videos/av1-720-new/s"
    folder.mkdir(parents=True)
    for name in [*(f"{serial}_rgb_low" for serial in EXO_SERIALS), *(f"HMC_{serial}_mono10bit_low" for serial in (1, 2, 3, 4))]:
        (folder / f"{name}.mp4").touch()
    manifest = tmp_path / "manifests/sequences.csv"
    manifest.parent.mkdir()
    if video_only is not None:
        manifest.write_text(
            f"sequence_name,video_only\n{'other' if video_only == 'missing-row' else 's'},{'True' if video_only == 'missing-row' else video_only}\n"
        )
    dataset = Assembly101Dataset(Assembly101Config(root=tmp_path))
    if video_only == "True":
        assert [source for _, source in dataset.discover()] == [Assembly101Source("s", has_poses=False)]
    else:
        assert dataset.discover() == []
        with pytest.raises(FileNotFoundError, match="incomplete sequences"):
            Assembly101Dataset(Assembly101Config(root=tmp_path, sequences=("s",))).discover()


def test_moving_headset_uses_one_track_and_static_camera_offsets(tmp_path: Path) -> None:
    folder = tmp_path / "videos/av1-720-new/s"
    folder.mkdir(parents=True)
    for name in ("C10095_rgb_low", "HMC_20_mono10bit_low", "HMC_10_mono10bit_low"):
        (folder / f"{name}.mp4").touch()
    reference = np.eye(4)
    reference[:3, 3] = [100, 200, 300]
    other = reference.copy()
    other[0, 3] += 40
    later = reference.copy()
    later[1, 3] += 100
    later_other = other.copy()
    later_other[1, 3] += 100
    values = {
        "camera_extrinsics_fixed": {"C10095:rgb": np.eye(4).tolist()},
        "timestamp": {"2": 10.033, "0": 10.0},
        "camera_extrinsics_ego": {
            "2": {"10:mono10bit": later.tolist(), "20:mono10bit": later_other.tolist()},
            "0": {"10:mono10bit": reference.tolist(), "20:mono10bit": other.tolist()},
        },
    }
    for member, value in values.items():
        path = pose_path(tmp_path, member, "s")
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(value))
    poses = read_poses(tmp_path, "s", camera_sources(tmp_path, "s"), None)
    assert poses.frames.tolist() == [0, 2]
    assert poses.timestamp_t0 == 10.0
    np.testing.assert_allclose(poses.world_T_rig[:, :3, 3], [[0.1, 0.2, 0.3], [0.1, 0.3, 0.3]])
    np.testing.assert_allclose(poses.rig_T_cam["20:mono10bit"][:3, 3], [0.04, 0, 0])
    assert not poses.calibration


def test_timestamp_t0_when_pose_keys_start_after_zero(tmp_path: Path) -> None:
    folder = tmp_path / "videos/av1-720-new/s"
    folder.mkdir(parents=True)
    for name in ("C10095_rgb_low", "HMC_20_mono10bit_low"):
        (folder / f"{name}.mp4").touch()
    values = {
        "camera_extrinsics_fixed": {"C10095:rgb": np.eye(4).tolist()},
        "timestamp": {"2": 10.033, "3": 10.05},
        "camera_extrinsics_ego": {key: {"20:mono10bit": np.eye(4).tolist()} for key in ("2", "3")},
    }
    for member, value in values.items():
        path = pose_path(tmp_path, member, "s")
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(value))
    poses = read_poses(tmp_path, "s", camera_sources(tmp_path, "s"), None)
    assert poses.frames.tolist() == [2, 3]
    assert poses.timestamp_t0 == 10.0


def test_write_actions_on_60_hz_frames_with_limit_and_initial_empty_state(tmp_path: Path) -> None:
    target = tmp_path / "actions.rrd"
    with rr.RecordingStream("test", recording_id="actions") as recording:
        recording.save(target)
        write_actions(recording, Actions(coarse=[Segment(30, 90, "hold")], fine=[Segment(0, 30, "hold")]), frame_limit=None)
        write_actions(recording, Actions(coarse=[], fine=[Segment(30, 90, "turn")]), frame_limit=12)
    rows: dict[str, list[tuple[list[int], list[list[str]]]]] = {}
    for chunk in read_chunks(target):
        if str(chunk.entity_path).startswith("/task/actions/"):
            batch = chunk.to_record_batch()
            rows.setdefault(str(chunk.entity_path), []).append(
                (batch.column(schema.FRAME_INDEX).to_pylist(), batch.column("TextDocument:text").to_pylist())
            )
    assert rows == {
        "/task/actions/coarse": [([0, 60, 180], [[""], ["hold"], [""]])],
        "/task/actions/fine": [([0, 60], [["hold"], [""]]), ([0], [[""]])],
    }


@pytest.mark.parametrize("rig,stored,expected", [(0, (1280, 720), 2.0 / 3.0), (8, (636, 480), 1.0)])
def test_video_scale_matches_intrinsics(rig: int, stored: tuple[int, int], expected: float) -> None:
    camera = CameraVideo(Path("video.mp4"), "serial", rig, 0, stored, 60, 60)
    width, height = camera.source_resolution
    lens = Lens("OpenCV" if rig == 0 else "OVFishEye62", width, height, "serial", 900.0, 901.0, 300.0, 200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert camera.scale == expected
    matrix = lens.matrix(stored)
    np.testing.assert_allclose([matrix[0, 0] / lens.fx, matrix[1, 1] / lens.fy], [camera.scale, camera.scale])


@pytest.mark.parametrize("model", ["OpenCV", "OVFishEye62"])
def test_camera_pinhole_components_match_legacy_values(model: Literal["OpenCV", "OVFishEye62"]) -> None:
    lens = Lens(model, 1920, 1080, "camera", 900.0, 901.0, 960.0, 540.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    transform = np.array([[0.0, -1.0, 0.0, 0.1], [1.0, 0.0, 0.0, 0.2], [0.0, 0.0, 1.0, 0.3], [0.0, 0.0, 0.0, 1.0]])
    camera = camera_parameters(lens, transform, (1280, 720))
    # log_camera_node uses child-from-parent; its inverse must be the old parent-from-child pose.
    np.testing.assert_allclose(np.linalg.inv(camera.extrinsics.cam_T_world), transform)
    actual = PinholeWithDistortion.from_camera(camera, image_plane_distance=0.05)
    legacy = rr.Pinhole(
        image_from_camera=lens.matrix((1280, 720)), resolution=(1280, 720), camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=0.05
    )
    for new, old in zip(actual.pinhole.as_component_batches(), legacy.as_component_batches(), strict=True):
        assert new.component_descriptor() == old.component_descriptor()
        assert new.as_arrow_array() == old.as_arrow_array()
    assert actual.distortion is not None


def test_confidence_record_preserves_shipped_values(tmp_path: Path) -> None:
    path = tmp_path / "confidence.json"
    path.write_text('{"2": {"0": 0.25, "1": 0.0}}')
    np.testing.assert_array_equal(read_confidence(path)[2], np.array([0.25, 0.0], dtype=np.float32))


def test_output_beneath_the_raw_root_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert_raw_root_guarded("assembly101", Assembly101Source("sequence", has_poses=True), tmp_path, monkeypatch)


def test_default_roots_follow_the_raw_root_and_annotations_follow_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAFORGE_RAW_ROOT", "/raw")
    assert Assembly101Config().root == Path("/raw/assembly101")
    assert Assembly101Config().annotations == Path("/raw/assembly101/official/annotations")
    assert Assembly101Config(root=Path("/elsewhere")).annotations == Path("/elsewhere/official/annotations")
    assert Assembly101Config(annotations_root=Path("/labels/annotations")).annotations == Path("/labels/annotations")
