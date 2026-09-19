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
from conftest import SHOW3D_RAW, blueprint_views, index_row, read_back, read_chunks, recording_properties
from jaxtyping import Bool, Float64
from numpy import ndarray
from serde import from_dict
from simplecv.camera_parameters import PinholeParameters, perspective_projection
from simplecv.umetrack_temp.generic_hand_model_numpy import NUM_JOINTS_PER_HAND, NUM_LANDMARKS_PER_HAND

from dataforge import schema, transports
from dataforge.datasets.base import DataforgeDataset
from dataforge.datasets.show3d import Show3dConfig, Show3dDataset, base_files
from dataforge.datasets.show3d_annotation_source import HAND_SIDES, Caption, HandFrame, HandPose, HandProfile, read_hand_frames
from dataforge.datasets.show3d_annotations import high_confidence_coverage, write_hand_pose_layer, write_properties_layer
from dataforge.datasets.show3d_layers import Scene, pinhole, read_scene
from dataforge.datasets.show3d_source import (
    CAPTIONS_VERSION,
    HAND_POSE_VERSION,
    FrameClock,
    FrameInfo,
    IndexRow,
    RecordingInfo,
    caption_file,
    hand_pose_file,
    hand_profile_file,
    read_frame_clock,
    read_json,
)
from dataforge.identity import SequenceIdentity


def test_hand_schema_preserves_null_world_and_null_uv_landmarks() -> None:
    pose: dict = dict(
        confidence=0.0,
        joint_angles=[0.0] * NUM_JOINTS_PER_HAND,
        wrist_rotation=None,
        wrist_translation=None,
        landmarks_3d_mm=None,
        landmarks_3d_mm_local=[[1.0, 2.0, 3.0]] * NUM_LANDMARKS_PER_HAND,
        landmarks_2d=None,
        extra="allowed",
    )
    frame: HandFrame = from_dict(HandFrame, dict(index=0, agt_frame_id=20, timestamp=1.0, missing_cameras=[], hand_poses={"0": pose, "1": pose}))
    assert frame.hand_poses["0"].landmarks_3d_mm is None
    assert frame.hand_poses["0"].joint_angles is not None
    assert frame.hand_poses["0"].joint_angles.dtype == np.float32
    assert frame.hand_poses["0"].landmarks_3d_mm_local is not None
    assert frame.hand_poses["0"].landmarks_3d_mm_local.shape == (NUM_LANDMARKS_PER_HAND, 3)
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
            object_pose_version="v1" if present else "",
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


@pytest.fixture(scope="module", params=["SPI102/keyboard_toss-away_83ef", "LYA722/birdhousetoy_shaking_8eca"])
def annotation_scene(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> AnnotationBuild:
    key: str = request.param
    hand_path: Path = SHOW3D_RAW / hand_pose_file(key)
    subject: str = key.split("/")[0]
    required: list[Path] = [
        hand_path,
        SHOW3D_RAW / hand_profile_file(subject),
        SHOW3D_RAW / caption_file(key),
        SHOW3D_RAW / "scenes" / key / "metadata/recording_info.json",
        SHOW3D_RAW / "scenes" / key / "metadata/frame_info.json",
    ]
    for path in required:
        if not path.is_file():
            pytest.skip(f"SHOW3D annotation asset absent: {path}")
    scene_dir: Path = SHOW3D_RAW / "scenes" / key
    clock: FrameClock = read_frame_clock(scene_dir, key)
    for camera in clock.info.resolution:
        for relative in (f"camera_calibration/{camera}.json", f"blur_info/{camera}.mp4.json"):
            path: Path = scene_dir / relative
            if not path.is_file():
                pytest.skip(f"SHOW3D scene asset absent: {path}")
    scene: Scene = read_scene(scene_dir, scene_key=key)
    frames: list[HandFrame] = read_hand_frames(hand_path, scene)
    identity: SequenceIdentity = SequenceIdentity("show3d", tuple(key.split("/")))
    target: Path = tmp_path_factory.mktemp(subject) / "hand_pose.rrd"
    profile_path: Path = SHOW3D_RAW / hand_profile_file(subject)
    profile_text: str = profile_path.read_text()
    read_json(profile_path, HandProfile, text=profile_text)
    write_hand_pose_layer(identity, scene, frames, profile_text, target)
    return AnnotationBuild(scene, frames, target, read_chunks(target), identity)


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
    for hand, side in HAND_SIDES:
        poses: list[HandPose] = [frame.hand_poses[hand] for frame in frames]
        for suffix, component, expected in (
            ("landmarks", "Points3D:positions", sum(p.landmarks_3d_mm is not None for p in poses)),
            ("landmarks_local", "Points3D:positions", sum(p.landmarks_3d_mm_local is not None for p in poses)),
            ("joint_angles", "joint_angles", sum(p.joint_angles is not None for p in poses)),
            ("wrist", "Transform3D:translation", sum(p.wrist_translation is not None for p in poses)),
            ("confidence", "Scalars:scalars", scene.info.num_frames),
        ):
            rows: list[rr.experimental.Chunk] = [
                c for c in chunks if str(c.entity_path) == f"{schema.hands_path(side)}/{suffix}" and component in c.to_record_batch().schema.names
            ]
            assert sum(c.num_rows for c in rows) == expected
        assert props[f"coverage_{side}_high_conf"] == pytest.approx(sum(p.confidence > 0.5 for p in poses) / scene.info.num_frames)
        for camera in (0, 1):
            uv_chunks: list[rr.experimental.Chunk] = [
                c for c in chunks if str(c.entity_path) == schema.hand_uv_path(1, camera, side) and not c.is_static
            ]
            uv: Float64[ndarray, "n 21 2"] = np.concatenate(
                [np.array(c.to_record_batch().column("Points2D:positions").to_pylist()) for c in uv_chunks]
            )
            source_rows: list[list[list[float] | None]] = [
                p.landmarks_2d[f"headset{camera}"] for p in poses if p.landmarks_2d is not None and f"headset{camera}" in p.landmarks_2d
            ]
            assert len(uv) == len(source_rows)
            assert np.isnan(uv).sum() == 2 * sum(point is None for row in source_rows for point in row)
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
    scene, _, _, chunks, _ = annotation_scene
    transforms: dict[int, Float64[ndarray, "4 4"]] = {
        pose.index: pose.T_WorldFromCamera for pose in scene.poses if pose.T_WorldFromCamera is not None
    }
    for _, side in HAND_SIDES:
        world: dict[int, Float64[ndarray, "21 3"]] = {
            row["frame_index"]: np.array(row["Points3D:positions"])
            for c in chunks
            if str(c.entity_path) == f"{schema.hands_path(side)}/landmarks" and not c.is_static
            for row in c.to_record_batch().to_pylist()
        }
        for camera in [c for c in scene.cameras if c.camera.rig == 1]:
            errors: list[float] = []
            path: str = schema.hand_uv_path(camera.camera.rig, camera.camera.cam, side)
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
                    shipped: Float64[ndarray, "21 2"] = np.array(row["Points2D:positions"])
                    valid: Bool[ndarray, "21"] = np.isfinite(shipped).all(axis=1) & np.isfinite(projected).all(axis=1)
                    errors.extend(np.linalg.norm(projected[valid] - shipped[valid], axis=1).tolist())
            assert len(errors) > 100
            median: float = float(np.median(errors))
            print(f"{side}/{camera.camera.source_name}: {len(errors)} points, median error {median:.6f} px")
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

    scene: Scene = Scene(
        RecordingInfo(20, 1, 60.0, {}),
        [FrameInfo(0, 20, 1.0, [])],
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int64),
        (),
        [],
        np.array([], dtype=np.int64),
    )
    hand: dict = dict(
        confidence=0.0,
        joint_angles=None,
        wrist_rotation=None,
        wrist_translation=None,
        landmarks_3d_mm=None,
        landmarks_3d_mm_local=None,
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


def test_hand_layer_omits_all_absent_measurements(tmp_path: Path) -> None:
    clock: FrameClock = FrameClock(
        RecordingInfo(20, 1, 60.0, {}),
        [FrameInfo(0, 20, 1.0, [])],
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int64),
    )
    pose: HandPose = HandPose(0.0, None, None, None, None, None, None)
    frame: HandFrame = HandFrame(0, 20, 1.0, [], {"0": pose, "1": pose})
    target: Path = tmp_path / "hand_pose.rrd"
    write_hand_pose_layer(SequenceIdentity("show3d", ("S", "none_wave_abcd")), clock, [frame], "{}", target)
    temporal: list[rr.experimental.Chunk] = [chunk for chunk in read_chunks(target) if not chunk.is_static]
    assert {str(chunk.entity_path) for chunk in temporal} == {
        "/world/gt/hands/left/confidence",
        "/world/gt/hands/right/confidence",
    }
