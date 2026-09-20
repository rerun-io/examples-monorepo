"""SHOW3D annotation boundaries and layer round trips."""

import json
import shutil
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
import rerun.blueprint as rrb
from conftest import SHOW3D_RAW, Show3dSceneInputs, blueprint_views, index_row, read_back, read_chunks, recording_properties
from jaxtyping import Bool, Float32, Float64
from numpy import ndarray
from serde import from_dict
from simplecv.camera_parameters import PinholeParameters, perspective_projection
from simplecv.umetrack_temp.generic_hand_model_numpy import NUM_JOINTS_PER_HAND, NUM_LANDMARKS_PER_HAND

from dataforge import schema, transports
from dataforge.datasets.base import DataforgeDataset
from dataforge.datasets.show3d import Show3dConfig, Show3dDataset, base_files
from dataforge.datasets.show3d_calibration import pinhole
from dataforge.datasets.show3d_captions import Caption, write_properties_layer
from dataforge.datasets.show3d_hands import (
    HAND_CONFIDENCE,
    HAND_SIDES,
    HandFrame,
    HandPose,
    high_confidence_coverage,
    read_hand_frames,
    write_hand_pose_layer,
)
from dataforge.datasets.show3d_layers import Scene
from dataforge.datasets.show3d_source import (
    CAPTIONS_VERSION,
    HAND_POSE_VERSION,
    HEADSET_CAMERAS,
    OBJECT_POSE_VERSION,
    FrameClock,
    FrameInfo,
    IndexRow,
    RecordingInfo,
    caption_file,
    hand_pose_file,
    hand_profile_file,
)
from dataforge.identity import SequenceIdentity


def test_hand_schema_preserves_null_world_and_null_uv_landmarks() -> None:
    pose: dict = dict(
        confidence=0.0,
        joint_angles=[0.0] * NUM_JOINTS_PER_HAND,
        wrist_rotation=None,
        wrist_translation=None,
        landmarks_3d_mm=None,
        landmarks_3d_mm_local=[[1.0, 2.0, 3.0]] * NUM_LANDMARKS_PER_HAND,  # source field the layer deliberately drops
        landmarks_2d=None,
        extra="allowed",
    )
    frame: HandFrame = from_dict(HandFrame, dict(index=0, agt_frame_id=20, timestamp=1.0, missing_cameras=[], hand_poses={"0": pose, "1": pose}))
    assert frame.hand_poses["0"].landmarks_3d_mm is None
    assert frame.hand_poses["0"].joint_angles is not None
    assert frame.hand_poses["0"].joint_angles.dtype == np.float32
    pose["landmarks_2d"] = {"headset0": [None] + [[3.0, 4.0]] * (NUM_LANDMARKS_PER_HAND - 1)}
    frame = from_dict(HandFrame, dict(index=0, agt_frame_id=20, timestamp=1.0, missing_cameras=[], hand_poses={"0": pose, "1": pose}))
    assert frame.hand_poses["0"].landmarks_2d is not None
    assert frame.hand_poses["0"].landmarks_2d["headset0"][0] is None
    assert frame.hand_poses["0"].landmarks_2d["headset0"][1] == [3.0, 4.0]


def test_caption_schema_and_episode_properties_have_stable_types(tmp_path: Path) -> None:
    caption: Caption = from_dict(
        Caption,
        dict(
            object_alias="ignored",
            action_hint="ignored",
            hand="both",
            interaction_description="Lift",
            start_state="Rest",
            end_state="Held",
            intent="Move",
            scene_description="Room",
            additional_observations="",
            overall_caption="Lift the toy.",
            extra=42,
        ),
    )
    assert caption.markdown().startswith("Lift the toy.\n\nObject alias\n:   ignored")
    schemas: list[pa.Schema] = []
    for present in (True, False):
        source: IndexRow = index_row(
            subject_id="S",
            scene_id="toy_pick_up_abcd",
            num_frames=4,
            has_object_pose=present,
            split="train",
            has_hand_pose=present,
            has_caption=present,
        )
        target: Path = tmp_path / f"{present}.rrd"
        write_properties_layer(SequenceIdentity("show3d", ("S", source.scene_id)), source, caption if present else None, target)
        props: dict[str, object] = recording_properties(read_back(target), "episode")
        assert props == dict(
            subject_id="S",
            split="train",
            object_alias="toy",
            action="pick_up",
            hand="both" if present else "",
            overall_caption="Lift the toy." if present else "",
            hand_pose_version=HAND_POSE_VERSION if present else "",
            object_pose_version=OBJECT_POSE_VERSION if present else "",
            captions_version=CAPTIONS_VERSION if present else "",
        )
        chunks: list[rr.experimental.Chunk] = read_chunks(target)
        assert len(chunks) == 1
        schemas.append(chunks[0].to_record_batch().schema.remove_metadata())
    assert schemas[0] == schemas[1]


def test_coverage_counts_strictly_above_half() -> None:
    assert high_confidence_coverage([0.0, 0.5, 0.51, 1.0]) == 0.5


class AnnotationBuild(NamedTuple):
    """A written hand layer and its source records."""

    scene: Scene
    frames: list[HandFrame]
    target: Path
    chunks: list[rr.experimental.Chunk]
    identity: SequenceIdentity


@pytest.fixture(scope="module")
def annotation_scene(show3d_scene_inputs: Show3dSceneInputs, tmp_path_factory: pytest.TempPathFactory) -> AnnotationBuild:
    inputs: Show3dSceneInputs = show3d_scene_inputs
    target: Path = tmp_path_factory.mktemp(inputs.identity.parts[0]) / "hand_pose.rrd"
    write_hand_pose_layer(inputs.identity, inputs.scene, inputs.hands, inputs.profile.text, target)
    return AnnotationBuild(inputs.scene, inputs.hands, target, read_chunks(target), inputs.identity)


@pytest.mark.integration
def test_real_scene_annotation_layers(annotation_scene: AnnotationBuild) -> None:
    scene, frames, target, chunks, identity = annotation_scene
    profile: Path = SHOW3D_RAW / hand_profile_file(identity.parts[0])
    assert rr.experimental.RrdReader(target).recordings()[0].recording_id == identity.recording_id
    for chunk in chunks:
        if not chunk.is_static:
            assert set(chunk.timeline_names) == {"video_time", "frame_index"}
    props: dict[str, object] = recording_properties(read_back(target), "hand_pose")
    assert props["version"] == HAND_POSE_VERSION
    assert recording_properties(read_back(target), "capture") == {}
    for side in HAND_SIDES:
        poses: list[HandPose] = [frame.hand_poses[side.key] for frame in frames]
        for suffix, component, expected in (
            ("joint_angles", "joint_angles", sum(p.joint_angles is not None for p in poses)),
            ("wrist", "Transform3D:translation", sum(p.wrist_translation is not None for p in poses)),
            ("confidence", "Scalars:scalars", scene.info.num_frames),
        ):
            rows: list[rr.experimental.Chunk] = [
                c
                for c in chunks
                if str(c.entity_path) == f"{schema.hands_path(side.name)}/{suffix}" and component in c.to_record_batch().schema.names
            ]
            assert sum(c.num_rows for c in rows) == expected
        assert props[f"coverage_{side.name}_high_conf"] == pytest.approx(sum(p.confidence > 0.5 for p in poses) / scene.info.num_frames)
    for path, component, confidence_component, dimensions in (
        (schema.coco133_xyz_path(), "Points3D:positions", "simplecv.KeypointConfidence3D:confidences", 3),
        (schema.coco133_uv_path(1, 0), "Points2D:positions", "simplecv.KeypointConfidence2D:confidences", 2),
        (schema.coco133_uv_path(1, 1), "Points2D:positions", "simplecv.KeypointConfidence2D:confidences", 2),
    ):
        keypoint_rows: list[dict] = [
            row for chunk in chunks if str(chunk.entity_path) == path and not chunk.is_static for row in chunk.to_record_batch().to_pylist()
        ]
        keypoint_rows.sort(key=lambda row: row["frame_index"])
        assert len(keypoint_rows) == len(frames)
        points: Float64[ndarray, "n 133 d"] = np.array([row[component] for row in keypoint_rows])
        confidence: Float64[ndarray, "n 133"] = np.array([row[confidence_component] for row in keypoint_rows])
        assert points.shape == (len(frames), 133, dimensions)
        assert confidence.shape == (len(frames), 133)
        assert np.isfinite(confidence).all()
        for index, frame in enumerate(frames):
            assert keypoint_rows[index]["frame_index"] == frame.index
            for hand_index, side in enumerate(HAND_SIDES):
                pose: HandPose = frame.hand_poses[side.key]
                offset: int = 91 + 21 * hand_index
                # Source fingertip 0 maps to COCO thumb4, independent of interpolation.
                placed: bool = pose.confidence > HAND_CONFIDENCE  # Hub README default threshold
                if dimensions == 3:
                    if pose.landmarks_3d_mm is None or not placed:
                        assert np.isnan(points[index, offset : offset + 21]).all()
                        assert (confidence[index, offset : offset + 21] == 0.0).all()
                    else:
                        np.testing.assert_allclose(points[index, offset + 4], pose.landmarks_3d_mm[0] * np.float32(0.001))
                        assert confidence[index, offset + 4] == pytest.approx(pose.confidence)
                else:
                    camera_name: str = "headset0" if path == schema.coco133_uv_path(1, 0) else "headset1"
                    pixels: list[list[float] | None] | None = (pose.landmarks_2d or {}).get(camera_name)
                    if pixels is None or pixels[0] is None or not placed:
                        assert np.isnan(points[index, offset + 4]).all()
                        assert confidence[index, offset + 4] == 0.0
                    else:
                        np.testing.assert_allclose(points[index, offset + 4], pixels[0])
                        assert confidence[index, offset + 4] == pytest.approx(pose.confidence)
    assert {str(chunk.entity_path) for chunk in chunks if str(chunk.entity_path).endswith("/coco133_uv")} == {
        schema.coco133_uv_path(1, 0),
        schema.coco133_uv_path(1, 1),
    }
    profile_text: list = next(
        c.to_record_batch().column("TextDocument:text").to_pylist() for c in chunks if str(c.entity_path) == schema.hand_profile_path()
    )
    assert profile_text[0][0] == profile.read_text()
    chunk_schema: pa.Schema = next(c.to_record_batch().schema for c in chunks if str(c.entity_path).endswith("/joint_angles"))
    assert chunk_schema.field("joint_angles").type.value_type.value_type == pa.float32()


@pytest.mark.golden
def test_hand_landmarks_reproject_through_base_camera_chain(
    annotation_scene: AnnotationBuild,
) -> None:
    """Project published hands through the package camera and projection chain."""
    scene: Scene = annotation_scene.scene
    chunks: list[rr.experimental.Chunk] = annotation_scene.chunks
    transforms: dict[int, Float64[ndarray, "4 4"]] = {
        pose.index: pose.T_WorldFromCamera for pose in scene.poses if pose.T_WorldFromCamera is not None
    }
    for hand_index, side in enumerate(HAND_SIDES):
        offset: int = 91 + 21 * hand_index
        world: dict[int, Float64[ndarray, "21 3"]] = {
            row["frame_index"]: np.array(row["Points3D:positions"])[offset : offset + 21]
            for c in chunks
            if str(c.entity_path) == schema.coco133_xyz_path() and not c.is_static
            for row in c.to_record_batch().to_pylist()
        }
        for camera in [c for c in scene.cameras if c.camera in HEADSET_CAMERAS]:
            errors: list[float] = []
            path: str = schema.coco133_uv_path(camera.camera.rig, camera.camera.cam)
            for chunk in chunks:
                if str(chunk.entity_path) != path or chunk.is_static:
                    continue
                for row in chunk.to_record_batch().to_pylist():
                    index: int = row["frame_index"]
                    if index not in world or index not in transforms:
                        continue
                    world_T_rig: Float64[ndarray, "4 4"] = transforms[index].copy()
                    world_T_rig[:3, 3] *= 0.001
                    parameters: PinholeParameters = pinhole(camera.camera.source_name, camera.calibration, world_T_rig @ camera.rig_T_cam)
                    assert parameters.extrinsics.cam_R_world is not None
                    assert parameters.extrinsics.cam_t_world is not None
                    assert parameters.intrinsics.k_matrix is not None
                    xyz: Float64[ndarray, "21 3"] = world[index] @ parameters.extrinsics.cam_R_world.T + parameters.extrinsics.cam_t_world
                    projected: Float64[ndarray, "21 2"] = perspective_projection(xyz, parameters.intrinsics.k_matrix.astype(np.float64))
                    shipped: Float64[ndarray, "21 2"] = np.array(row["Points2D:positions"])[offset : offset + 21]
                    valid: Bool[ndarray, "21"] = np.isfinite(shipped).all(axis=1) & np.isfinite(projected).all(axis=1)
                    errors.extend(np.linalg.norm(projected[valid] - shipped[valid], axis=1).tolist())
            assert len(errors) > 100
            median: float = float(np.median(errors))
            print(f"{side.name}/{camera.camera.source_name}: {len(errors)} points, median error {median:.6f} px")
            assert median < 0.5


@pytest.mark.integration
def test_convert_rebuilds_each_annotation_without_videos_or_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    raw: Path = SHOW3D_RAW
    key: str = "SPI102/keyboard_toss-away_83ef"
    if not (raw / hand_pose_file(key)).is_file():
        pytest.skip(f"SHOW3D keyboard sidecars absent: {raw}")
    root: Path = tmp_path / "raw"
    for relative in (
        f"scenes/{key}",
        str(Path(hand_pose_file(key)).parent),
        str(Path(hand_profile_file("SPI102")).parent),
        str(Path(caption_file(key)).parent),
    ):
        shutil.copytree(raw / relative, root / relative, ignore=shutil.ignore_patterns("*.mp4"))
    shutil.rmtree(root / "scenes" / key / "camera_calibration")
    shutil.rmtree(root / "scenes" / key / "blur_info")
    retained_video: Path = root / "scenes" / key / "headset0.mp4"
    retained_video.write_bytes(b"keep raw when only annotations are rebuilt")
    output: Path = tmp_path / "rrd"
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(output))
    identity: SequenceIdentity = SequenceIdentity("show3d", tuple(key.split("/")))
    base: Path = output / "base" / f"{identity.recording_id}.rrd"
    base.parent.mkdir(parents=True)
    base.write_bytes(b"existing base must not be read or rewritten")
    source: IndexRow = index_row(
        subject_id="SPI102",
        scene_id="keyboard_toss-away_83ef",
        num_frames=1,
        has_object_pose=True,
        split="train",
        has_hand_pose=True,
        has_caption=True,
    )
    # This test isolates the PR3 layers: the later layers already exist.
    for layer in ("object_pose", "object_mesh", "hand_mesh"):
        existing: Path = output / layer / base.name
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"existing later layer")
    dataset: DataforgeDataset = Show3dConfig(root=root).setup()
    assert dataset.convert(identity, source, force=False) == base
    assert capsys.readouterr().out.splitlines()[-1] == f"done {key}: hand_pose, captions, properties"
    assert "commit_sha" not in dataset.__dict__  # No Hub call is needed for retained sidecars.
    targets: list[Path] = [output / layer / f"{identity.recording_id}.rrd" for layer in ("hand_pose", "captions", "properties")]
    for missing in targets:
        before: dict[Path, int] = {p: p.stat().st_mtime_ns for p in [base, *targets]}
        missing.unlink()
        dataset.convert(identity, source, force=False)
        assert capsys.readouterr().out.splitlines()[-1] == f"done {key}: {missing.parent.name}"
        assert missing.is_file()
        assert all(p.stat().st_mtime_ns == stamp for p, stamp in before.items() if p != missing)
    assert recording_properties(read_back(targets[-1]), "episode")["action"] == "toss-away"
    assert base.read_bytes() == b"existing base must not be read or rewritten"
    assert retained_video.read_bytes() == b"keep raw when only annotations are rebuilt"


@pytest.mark.parametrize("fault", ["count", "index", "timestamp"])
def test_hand_reader_rejects_census_or_clock_mismatch(tmp_path: Path, fault: str) -> None:
    scene: FrameClock = FrameClock(
        RecordingInfo(20, 1, 60.0, {}),
        [FrameInfo(0, 20, 1.0, [])],
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int64),
    )
    hand: dict = dict(
        confidence=0.0,
        joint_angles=None,
        wrist_rotation=None,
        wrist_translation=None,
        landmarks_3d_mm=None,
        landmarks_2d=None,
    )
    records: dict = (
        {}
        if fault == "count"
        else {
            "0": dict(
                index=1 if fault == "index" else 0,
                agt_frame_id=20,
                timestamp=2.0 if fault == "timestamp" else 1.0,
                missing_cameras=[],
                hand_poses={"0": hand, "1": hand},
            )
        }
    )
    source: Path = tmp_path / "hands.json"
    source.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="census|disagrees"):
        read_hand_frames(source, scene)


def test_properties_only_conversion_needs_no_scene_sidecars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path))
    identity: SequenceIdentity = SequenceIdentity("show3d", ("S", "none_wave-hands_abcd"))
    base: Path = tmp_path / "base" / f"{identity.recording_id}.rrd"
    base.parent.mkdir()
    base.write_bytes(b"existing")
    dataset: DataforgeDataset = Show3dConfig(root=tmp_path / "absent-raw").setup()
    dataset.convert(
        identity,
        index_row(
            subject_id="S", scene_id="none_wave-hands_abcd", num_frames=4, has_object_pose=False, split="test", has_hand_pose=False, has_caption=False
        ),
        force=False,
    )
    props: dict[str, object] = recording_properties(read_back(tmp_path / "properties" / base.name), "episode")
    assert props["action"] == "wave-hands"
    assert props["hand"] == props["overall_caption"] == props["captions_version"] == ""


def test_default_blueprint_includes_instruction_below_ego_panes() -> None:
    views: list[rrb.View] = blueprint_views(Show3dConfig().setup().default_blueprint())
    assert [view.name for view in views[:4]] == ["Back rig frame", "headset0", "headset1", "Instruction"]
    assert isinstance(views[3], rrb.TextDocumentView)
    assert views[3].origin == schema.instruction_path()


@pytest.mark.parametrize(("scene_id", "alias", "action"), [("none_clap-hands_a702", "none", "clap-hands"), ("toy_pick_up_abcd", "toy", "pick_up")])
def test_index_scene_id_parts(scene_id: str, alias: str, action: str) -> None:
    row: IndexRow = index_row(
        subject_id="S", scene_id=scene_id, num_frames=1, has_object_pose=False, split="train", has_hand_pose=False, has_caption=False
    )
    assert (row.object_alias, row.action) == (alias, action)


def test_index_rejects_incomplete_scene_id() -> None:
    with pytest.raises(ValueError, match="scene id"):
        index_row(
            subject_id="S", scene_id="none_clap-hands", num_frames=1, has_object_pose=False, split="train", has_hand_pose=False, has_caption=False
        )


def test_base_input_plan_excludes_absent_cameras() -> None:
    source: IndexRow = index_row(
        subject_id="S",
        scene_id="none_wave_abcd",
        num_frames=1,
        has_object_pose=False,
        split="test",
        has_hand_pose=False,
        has_caption=False,
        has_rig0=False,
    )
    files: list[str] = base_files(source, "S/none_wave_abcd")
    assert not any("/rig0." in name for name in files)
    assert "scenes/S/none_wave_abcd/rig1.mp4" in files


def test_fetch_missing_preserves_retained_raw_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    retained: Path = tmp_path / "retained.json"
    retained.write_text("keep")
    dataset: Show3dDataset = Show3dDataset(Show3dConfig(root=tmp_path))
    dataset.__dict__["commit_sha"] = "test-sha"
    calls: list[list[str]] = []

    def fetch(repo_id: str, paths: list[str], *, local_dir: Path, revision: str) -> Path:
        calls.append(paths)
        return local_dir

    monkeypatch.setattr(transports, "hf_fetch_files", fetch)
    dataset.fetch_missing(["retained.json", "absent.json"])
    assert calls == [["absent.json"]]
    assert retained.read_text() == "keep"


def test_hand_layer_writes_dense_coco133_with_shipped_confidence_and_pixels(tmp_path: Path) -> None:
    clock: FrameClock = FrameClock(
        RecordingInfo(20, 2, 60.0, {}),
        [FrameInfo(0, 20, 1.0, []), FrameInfo(1, 21, 2.0, [])],
        np.array([0, 1_000_000_000], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
    )
    landmarks: Float32[ndarray, "21 3"] = np.zeros((21, 3), dtype=np.float32)
    landmarks[0] = [1000.0, 2000.0, 3000.0]
    landmarks[5] = [2000.0, 4000.0, 6000.0]
    landmarks[6] = [4000.0, 6000.0, 8000.0]
    pixels: list[list[float] | None] = [[30.0, 40.0] for _ in range(21)]
    pixels[5] = [10.0, 20.0]
    pixels[6] = [50.0, 60.0]
    pixels[1] = None
    left: HandPose = HandPose(0.6, None, None, None, landmarks, {"headset0": pixels})
    right: HandPose = HandPose(1.0, None, None, None, landmarks, {"headset1": pixels})
    absent: HandPose = HandPose(0.75, None, None, None, None, None)
    low: HandPose = HandPose(0.5, None, None, None, landmarks, {"headset0": pixels})  # at the threshold: shipped but not placed
    frames: list[HandFrame] = [
        HandFrame(0, 20, 1.0, [], {"0": left, "1": right}),
        HandFrame(1, 21, 2.0, [], {"0": low, "1": right}),
    ]
    del absent
    target: Path = tmp_path / "hand_pose.rrd"
    write_hand_pose_layer(SequenceIdentity("show3d", ("S", "none_wave_abcd")), clock, frames, "{}", target)
    chunks: list[rr.experimental.Chunk] = read_chunks(target)
    temporal: list[rr.experimental.Chunk] = [chunk for chunk in chunks if not chunk.is_static]
    assert {str(chunk.entity_path) for chunk in temporal} == {
        "/world/gt/hands/left/confidence",
        "/world/gt/hands/right/confidence",
        "/world/gt/coco133_xyz",
        "/world/rig_01/cam_00/pinhole/coco133_uv",
        "/world/rig_01/cam_01/pinhole/coco133_uv",
    }
    for path, component, confidence_component, dimensions in (
        (schema.coco133_xyz_path(), "Points3D:positions", "simplecv.KeypointConfidence3D:confidences", 3),
        (schema.coco133_uv_path(1, 0), "Points2D:positions", "simplecv.KeypointConfidence2D:confidences", 2),
        (schema.coco133_uv_path(1, 1), "Points2D:positions", "simplecv.KeypointConfidence2D:confidences", 2),
    ):
        rows: list[dict] = [row for chunk in temporal if str(chunk.entity_path) == path for row in chunk.to_record_batch().to_pylist()]
        rows.sort(key=lambda row: row["frame_index"])
        assert [row["frame_index"] for row in rows] == [0, 1]
        assert all("video_time" in row for row in rows)
        points: Float64[ndarray, "2 133 d"] = np.array([row[component] for row in rows])
        confidence: Float64[ndarray, "2 133"] = np.array([row[confidence_component] for row in rows])
        assert points.shape == (2, 133, dimensions)
        assert confidence.shape == (2, 133)
        assert np.isfinite(confidence).all()
        assert np.isnan(points[:, :9]).all()
        assert np.isnan(points[:, 11:91]).all()
        assert (confidence[:, :9] == 0.0).all()
        assert (confidence[:, 11:91] == 0.0).all()
        assert np.isnan(points[1, 91:112]).all()
        assert np.isnan(points[1, 9]).all()
        assert (confidence[1, 91:112] == 0.0).all()
        assert confidence[1, 9] == 0.0
        if dimensions == 3:
            np.testing.assert_allclose(points[0, 95], [1.0, 2.0, 3.0])
            np.testing.assert_allclose(points[0, [9, 91]], [[2.0, 4.0, 6.0]] * 2)
            np.testing.assert_allclose(points[0, 92], [3.0, 5.0, 7.0])
            assert confidence[0, 91:112] == pytest.approx(0.6)
            assert confidence[0, 9] == pytest.approx(0.6)
            assert np.isnan(points[1, 91:112]).all()  # confidence 0.5 is not above the README default
            assert (confidence[1, 91:112] == 0.0).all()
            assert (confidence[:, 112:133] == 1.0).all()
            assert (confidence[:, 10] == 1.0).all()
            colors: list[int] = rows[0]["Points3D:colors"]
            assert len(colors) == 133
            assert colors[116] == 0x00FF00FF  # 1.0 confidence: green RGBA.
        else:
            offset: int = 91 if path == schema.coco133_uv_path(1, 0) else 112
            np.testing.assert_allclose(points[0, offset + 4], [30.0, 40.0])
            np.testing.assert_allclose(points[0, offset + 1], [30.0, 40.0])
            assert np.isnan(points[0, offset + 8]).all()  # Null index fingertip.
            assert confidence[0, offset + 8] == 0.0
            assert confidence[0, offset + 4] == pytest.approx(0.6 if offset == 91 else 1.0)
            assert (confidence[~np.isfinite(points).all(axis=2)] == 0.0).all()
    assert not any(str(chunk.entity_path).endswith(("/landmarks", "/uv")) for chunk in chunks)
