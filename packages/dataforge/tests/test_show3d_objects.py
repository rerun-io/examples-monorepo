"""SHOW3D object boundaries, derived geometry, and recording contracts."""

import json
import shutil
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import rerun as rr
from conftest import SHOW3D_RAW, Show3dSceneInputs, index_row, read_back, read_chunks, recording_properties
from jaxtyping import Float32
from numpy import ndarray
from serde import from_dict
from simplecv.umetrack_temp.generic_hand_model_numpy import HandModelNumpy, skin_landmarks, wrist_for_hand

from dataforge import paths, schema, transports
from dataforge.datasets.show3d import Show3dConfig, Show3dDataset
from dataforge.datasets.show3d_calibration import HeadsetCalibration, HeadsetPose
from dataforge.datasets.show3d_hands import HAND_SIDES, HandFrame, HandPose, write_hand_mesh_layer
from dataforge.datasets.show3d_mesh_source import MESH_REPO, MeshAsset, MeshInfo, download_meshes, mesh_ids, strip_texture_transform, stripped_mesh
from dataforge.datasets.show3d_object_source import CLOCK_TOLERANCE_S, ObjectFrame, ObjectTrack, read_object_frames
from dataforge.datasets.show3d_objects import HIDDEN_MESH_SCALE, ObjectSanity, object_sanity, write_object_mesh_layer, write_object_pose_layer
from dataforge.datasets.show3d_source import (
    HEADSET_CAMERAS,
    OBJECT_POSE_VERSION,
    OBJECTS,
    FrameClock,
    FrameInfo,
    IndexRow,
    RecordingInfo,
    caption_file,
    hand_pose_file,
    hand_profile_file,
    mesh_name,
    object_pose_file,
    read_json,
    scene_id_parts,
)
from dataforge.identity import SequenceIdentity


def test_object_record_accepts_empty_unposed_rows_and_checks_proper_rotation() -> None:
    row: dict = dict(index=0, agt_frame_id=20, timestamp=1.0, missing_cameras=[], R=[], t=[], confidence=0.0, extra=True)
    assert from_dict(ObjectFrame, row).R == []
    row.update(confidence=1.0, R=np.eye(3).tolist(), t=[[1000.0], [0.0], [0.0]])
    assert from_dict(ObjectFrame, row).t[0][0] == 1000.0
    row["R"][0][0] = -1.0
    with pytest.raises(ValueError, match="proper rigid"):
        from_dict(ObjectFrame, row)


def test_mesh_aliases_are_unique_and_mapping_is_disjoint() -> None:
    mapped: set[str] = {alias for alias, name in OBJECTS.items() if name is not None}
    unmapped: set[str] = {alias for alias, name in OBJECTS.items() if name is None}
    assert mapped.isdisjoint(unmapped)
    assert len(mapped) == 22
    assert len(unmapped - {"none"}) == 5
    assert len(mapped | unmapped) == len(OBJECTS)


@pytest.mark.integration
def test_mapped_names_resolve_in_bop_census() -> None:
    path: Path = SHOW3D_RAW / "assets/hot3d_bop/object_models/models_info.json"
    if not path.is_file():
        pytest.skip(f"SHOW3D model census absent: {path}")
    records: dict[str, MeshInfo] = read_json(path, dict[str, MeshInfo])
    names: set[str] = {record.name for record in records.values()}
    assert all(name in names for name in OBJECTS.values() if name is not None)


def test_object_table_covers_both_indexes() -> None:
    indexes: list[Path] = [SHOW3D_RAW / f"dataset_index_{split}.parquet" for split in ("train", "test")]
    for path in indexes:
        if not path.is_file():
            pytest.skip(f"SHOW3D index absent: {path}")
    aliases: set[str] = set(OBJECTS)
    for path in indexes:
        assert {scene_id_parts(scene)[0] for scene in pq.read_table(path, columns=["scene_id"]).column("scene_id").to_pylist()} <= aliases


def test_glb_strip_preserves_binary_scale_and_other_extensions() -> None:
    document: dict = dict(
        asset={"version": "2.0"},
        nodes=[{"scale": [0.001] * 3}],
        extensionsUsed=["KHR_texture_transform", "KEEP"],
        extensionsRequired=["KHR_texture_transform"],
        materials=[
            {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0, "extensions": {"KHR_texture_transform": {"offset": [0, 0]}, "KEEP": {}}}}}
        ],
    )
    payload: bytes = json.dumps(document).encode()
    payload += b" " * (-len(payload) % 4)
    binary: bytes = struct.pack("<I4s", 4, b"BIN\x00") + b"1234"
    glb: bytes = struct.pack("<4sII", b"glTF", 2, 20 + len(payload) + len(binary)) + struct.pack("<I4s", len(payload), b"JSON") + payload + binary
    stripped: bytes = strip_texture_transform(glb)
    length: int = struct.unpack_from("<I", stripped, 12)[0]
    result: dict = json.loads(stripped[20 : 20 + length])
    assert result["nodes"] == document["nodes"]
    assert result["extensionsUsed"] == ["KEEP"]
    assert "KHR_texture_transform" not in str(result)
    assert stripped[20 + length :] == binary
    assert struct.unpack_from("<I", stripped, 8)[0] == len(stripped)
    assert strip_texture_transform(stripped) == stripped


def test_object_sanity_uses_depth_bounds_nearest_palm_and_posed_denominator() -> None:
    frames: list[ObjectFrame] = [
        ObjectFrame(i, i, float(i), [], np.eye(3).tolist(), [[x], [0.0], [z]], 1.0)
        for i, (x, z) in enumerate([(0.0, 1000.0), (0.0, -1000.0), (3000.0, 1000.0)])
    ]
    frames.append(ObjectFrame(3, 3, 3.0, [], [], [], 0.0))
    camera: HeadsetCalibration = HeadsetCalibration(
        100, 100, 100.0, 100.0, 50.0, 50.0, "PinholePlane", {str(i): HeadsetPose(i, i, float(i), np.eye(4), False) for i in range(4)}
    )
    palms: Float32[ndarray, "21 3"] = np.zeros((21, 3), dtype=np.float32)
    palms[20] = [0.0, 0.0, 900.0]
    hand: HandPose = HandPose(1.0, None, None, None, palms, None)
    absent: HandPose = HandPose(0.0, None, None, None, None, None)
    hands: list[HandFrame] = [HandFrame(i, i, float(i), [], {"0": hand if i == 0 else absent, "1": absent}) for i in range(4)]
    result: ObjectSanity = object_sanity(frames, [camera], hands)
    assert result.coverage == 0.75
    assert result.in_ego_fov_fraction == pytest.approx(1 / 3)
    assert result.palm_dist_median_m == pytest.approx(0.1)
    empty: ObjectSanity = object_sanity([frames[-1]], [], [])
    assert empty.coverage == 0.0
    assert np.isnan(empty.in_ego_fov_fraction)
    assert np.isnan(empty.palm_dist_median_m)


def mesh_vertex_counts(chunks: Sequence[rr.experimental.Chunk]) -> dict[int, int]:
    """Vertex count per frame_index across temporal Mesh3D chunks."""
    rows: dict[int, int] = {}
    for c in chunks:
        batch: pa.RecordBatch = c.to_record_batch()
        for index, vertices in zip(batch.column("frame_index").to_pylist(), batch.column("Mesh3D:vertex_positions").to_pylist(), strict=True):
            rows[int(index)] = len(vertices)
    return rows


@pytest.mark.integration
def test_hand_mesh_skins_only_above_default_confidence_and_clears_other_frames(show3d_scene_inputs: Show3dSceneInputs, tmp_path: Path) -> None:
    """Wrists ship at confidence 0 and below the README default 0.5; the derived mesh draws neither, and other frames hold nothing."""
    inputs: Show3dSceneInputs = show3d_scene_inputs
    good: HandPose = next(
        f.hand_poses["1"]
        for f in inputs.hands
        if f.hand_poses["1"].wrist_rotation is not None and f.hand_poses["1"].trusted
    )
    low: HandPose = HandPose(0.3, good.joint_angles, good.wrist_rotation, good.wrist_translation, good.landmarks_3d_mm, None)
    lost: HandPose = HandPose(0.0, good.joint_angles, good.wrist_rotation, good.wrist_translation, None, None)
    absent: HandPose = HandPose(0.0, good.joint_angles, None, None, None, None)
    base: list[FrameInfo] = inputs.scene.frames[:4]
    hands: list[HandFrame] = [
        HandFrame(f.index, f.agt_frame_id, f.timestamp, f.missing_cameras, {"0": absent, "1": pose})
        for f, pose in zip(base, [good, low, lost, absent], strict=True)
    ]
    target: Path = tmp_path / "hand_mesh.rrd"
    write_hand_mesh_layer(inputs.identity, inputs.scene, hands, inputs.profile.model, target)
    chunks: list[rr.experimental.Chunk] = read_chunks(target)
    right: dict[int, int] = mesh_vertex_counts([c for c in chunks if str(c.entity_path) == schema.hand_mesh_path("right") and not c.is_static])
    left: dict[int, int] = mesh_vertex_counts([c for c in chunks if str(c.entity_path) == schema.hand_mesh_path("left") and not c.is_static])
    assert right == {base[0].index: len(inputs.profile.model.mesh_vertices), base[1].index: 0, base[2].index: 0, base[3].index: 0}
    assert left == {f.index: 0 for f in base}


class ObjectBuild(NamedTuple):
    """Written layers and their independently shipped reference inputs."""

    identity: SequenceIdentity
    frames: list[ObjectFrame]
    hands: list[HandFrame]
    profile: HandModelNumpy
    output: Path
    metrics: ObjectSanity


@pytest.fixture(scope="module")
def object_scene(show3d_scene_inputs: Show3dSceneInputs, tmp_path_factory: pytest.TempPathFactory) -> ObjectBuild:
    inputs: Show3dSceneInputs = show3d_scene_inputs
    identity: SequenceIdentity = inputs.identity
    path: Path = SHOW3D_RAW / object_pose_file(identity.sequence_key)
    if not path.is_file():
        pytest.skip(f"SHOW3D object asset absent: {path}")
    alias: str = scene_id_parts(identity.parts[1])[0]
    try:
        asset: MeshAsset = stripped_mesh(SHOW3D_RAW, alias)
    except FileNotFoundError as error:
        pytest.skip(f"SHOW3D mesh asset absent: {error}")
    track: ObjectTrack = read_object_frames(path, inputs.scene)
    frames: list[ObjectFrame] = track.frames
    metrics: ObjectSanity = object_sanity(frames, list(inputs.scene.headsets.values()), inputs.hands)
    output: Path = tmp_path_factory.mktemp(identity.parts[0] + "-objects")
    write_object_pose_layer(identity, alias, inputs.scene, frames, metrics, output / "object_pose.rrd", clock_offset_s=track.clock_offset_s)
    write_object_mesh_layer(identity, alias, inputs.scene, frames, asset.mesh_id, asset.path, output / "object_mesh.rrd")
    write_hand_mesh_layer(identity, inputs.scene, inputs.hands, inputs.profile.model, output / "hand_mesh.rrd")
    return ObjectBuild(identity, frames, inputs.hands, inputs.profile.model, output, metrics)


@pytest.mark.integration
def test_real_scene_object_and_mesh_layers(object_scene: ObjectBuild) -> None:
    build: ObjectBuild = object_scene
    alias: str = scene_id_parts(build.identity.parts[1])[0]
    chunks: list[rr.experimental.Chunk] = read_chunks(build.output / "object_pose.rrd")
    for entity_path, component, expected in [
        (schema.objects_path(alias), "Transform3D:translation", sum(frame.confidence > 0.0 for frame in build.frames)),
        (schema.object_confidence_path(alias), "Scalars:scalars", len(build.frames)),
    ]:
        selected: list[rr.experimental.Chunk] = [
            c for c in chunks if str(c.entity_path) == entity_path and component in c.to_record_batch().schema.names
        ]
        assert sum(c.num_rows for c in selected) == expected
        assert all(set(c.timeline_names) == {"video_time", "frame_index"} for c in selected)
    props: dict[str, object] = recording_properties(read_back(build.output / "object_pose.rrd"), "object_pose")
    assert props["version"] == OBJECT_POSE_VERSION
    assert props["coverage"] == pytest.approx(sum(f.confidence > 0.0 for f in build.frames) / len(build.frames))
    mesh_chunks: list[rr.experimental.Chunk] = read_chunks(build.output / "object_mesh.rrd")
    assert any(
        str(c.entity_path) == schema.object_mesh_path(alias) and c.is_static and "Asset3D:blob" in c.to_record_batch().schema.names
        for c in mesh_chunks
    )
    # The static mesh would otherwise persist at the last pose through unposed frames; a dense scale on the
    # mesh entity hides it there while the shipped pose stream on the parent stays sparse and untouched.
    scale_rows: dict[int, float] = {}
    for c in mesh_chunks:
        if str(c.entity_path) == schema.object_mesh_path(alias) and not c.is_static:
            batch: pa.RecordBatch = c.to_record_batch()
            assert set(c.timeline_names) == {"video_time", "frame_index"}
            for index, scale in zip(batch.column("frame_index").to_pylist(), batch.column("Transform3D:scale").to_pylist(), strict=True):
                scale_rows[int(index)] = float(scale[0][0])
    assert len(scale_rows) == len(build.frames)
    assert any(not f.posed for f in build.frames), "fixture lacks an unposed object frame"
    assert all(scale_rows[f.index] == float(np.float32(1.0 if f.posed else HIDDEN_MESH_SCALE)) for f in build.frames)
    assert recording_properties(read_back(build.output / "object_mesh.rrd"), "object_mesh") == {
        "mesh_id": 28 if alias == "keyboard" else 26,
        "mesh_source": "bop-benchmark/hot3d",
    }
    hand_chunks: list[rr.experimental.Chunk] = read_chunks(build.output / "hand_mesh.rrd")
    for side in HAND_SIDES:
        temporal: list[rr.experimental.Chunk] = [c for c in hand_chunks if str(c.entity_path) == schema.hand_mesh_path(side.name) and not c.is_static]
        # One row per frame: skinned vertices where Meta trusts the hand, an empty row otherwise so the viewer holds nothing.
        assert sum(c.num_rows for c in temporal) == len(build.hands)
        trusted: list[bool] = [f.hand_poses[side.key].wrist_rotation is not None and f.hand_poses[side.key].trusted for f in build.hands]
        rows: dict[int, int] = mesh_vertex_counts(temporal)
        assert [rows[f.index] > 0 for f in build.hands] == trusted
        assert {n for n in rows.values() if n} == {len(build.profile.mesh_vertices)}
        assert all(set(c.timeline_names) == {"video_time", "frame_index"} for c in temporal)
        assert any(
            c.is_static and str(c.entity_path) == schema.hand_mesh_path(side.name) and "Mesh3D:triangle_indices" in c.to_record_batch().schema.names
            for c in hand_chunks
        )
    for layer in ("object_pose", "object_mesh", "hand_mesh"):
        path: Path = build.output / f"{layer}.rrd"
        assert rr.experimental.RrdReader(path).recordings()[0].recording_id == build.identity.recording_id
        assert recording_properties(read_back(path), "capture") == {}
    assert recording_properties(read_back(build.output / "hand_mesh.rrd"), "hand_mesh") == {}


@pytest.mark.golden
def test_skinning_matches_shipped_landmarks_both_hands(object_scene: ObjectBuild) -> None:
    build: ObjectBuild = object_scene
    for side in HAND_SIDES:
        poses: list[HandPose] = [
            frame.hand_poses[side.key]
            for frame in build.hands
            if frame.hand_poses[side.key].wrist_rotation is not None and frame.hand_poses[side.key].landmarks_3d_mm is not None
        ]
        assert poses
        assert all(pose.joint_angles is not None for pose in poses)
        angles: Float32[ndarray, "n 22"] = np.asarray([pose.joint_angles for pose in poses], dtype=np.float32)
        wrists: Float32[ndarray, "n 4 4"] = np.zeros((len(poses), 4, 4), dtype=np.float32)
        wrists[:, :3, :3] = np.asarray([pose.wrist_rotation for pose in poses])
        wrists[:, :3, 3] = np.asarray([pose.wrist_translation for pose in poses])
        wrists[:, 3, 3] = 1.0
        predicted: Float32[ndarray, "n 21 3"] = skin_landmarks(build.profile, angles, wrist_for_hand(wrists, side.model_index))
        shipped: Float32[ndarray, "n 21 3"] = np.asarray([pose.landmarks_3d_mm for pose in poses], dtype=np.float32)
        error: float = float(np.linalg.norm(predicted - shipped, axis=-1).max())
        print(f"{build.identity.sequence_key} hand {side.key}: max skinning error {error:.8f} mm")
        assert error < 0.01


@pytest.mark.golden
def test_object_sanity_regression_bands(object_scene: ObjectBuild) -> None:
    build: ObjectBuild = object_scene
    metrics: ObjectSanity = build.metrics
    print(f"{build.identity.sequence_key}: {metrics}")
    if scene_id_parts(build.identity.parts[1])[0] == "keyboard":
        assert metrics.in_ego_fov_fraction < 0.05
        assert metrics.palm_dist_median_m > 1.0
    else:
        assert metrics.in_ego_fov_fraction > 0.9
        assert metrics.palm_dist_median_m < 0.2


@pytest.mark.integration
def test_convert_rebuilds_each_mesh_and_object_layer_without_video(
    object_scene: ObjectBuild, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    build: ObjectBuild = object_scene
    key: str = build.identity.sequence_key
    raw: Path = tmp_path / "raw"
    for relative in (
        f"scenes/{key}",
        str(Path(object_pose_file(key)).parent),
        str(Path(hand_pose_file(key)).parent),
        str(Path(caption_file(key)).parent),
        str(Path(hand_profile_file(build.identity.parts[0])).parent),
        "assets/hot3d_bop",
    ):
        shutil.copytree(SHOW3D_RAW / relative, raw / relative, ignore=shutil.ignore_patterns("*.mp4"))
    output: Path = tmp_path / "rrd"
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(output))
    for layer in ("base",):
        target: Path = output / layer / f"{build.identity.recording_id}.rrd"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"preexisting layer")
    dataset: Show3dDataset = Show3dDataset(Show3dConfig(root=raw))
    source: IndexRow = index_row(subject_id=build.identity.parts[0], scene_id=build.identity.parts[1])
    dataset.convert(build.identity, source, force=False)
    assert capsys.readouterr().out.splitlines()[-1] == (
        f"done {key}: hand_pose, captions, properties, object_pose, object_mesh, hand_mesh"
    )
    assert "commit_sha" not in dataset.__dict__
    targets: list[Path] = [output / layer / f"{build.identity.recording_id}.rrd" for layer in dataset.layers]
    for layer in ("object_pose", "object_mesh", "hand_mesh"):
        missing: Path = output / layer / targets[0].name
        before: dict[Path, int] = {p: p.stat().st_mtime_ns for p in targets}
        missing.unlink()
        dataset.convert(build.identity, source, force=False)
        assert missing.is_file()
        assert all(p.stat().st_mtime_ns == stamp for p, stamp in before.items() if p != missing)
    assert not list(raw.rglob("*.mp4"))


@pytest.mark.parametrize("fault", ["count", "key", "frame_id", "drift"])
def test_object_reader_rejects_census_or_alignment_faults(tmp_path: Path, fault: str) -> None:
    frames: list[FrameInfo] = [FrameInfo(0, 20, 1.0, []), FrameInfo(1, 21, 1.5, [])]
    clock: FrameClock = FrameClock(RecordingInfo(20, 2, 60.0, {}), frames, np.array([0, 500_000_000], dtype=np.int64), np.array([0, 1], dtype=np.int64))
    records: dict[str, dict[str, object]] = {
        str(i): dict(index=i, agt_frame_id=20 + i, timestamp=frame.timestamp, missing_cameras=[], R=[], t=[], confidence=0.0) for i, frame in enumerate(frames)
    }
    if fault == "count":
        records.pop("1")
    elif fault == "key":
        records["7"] = records.pop("1")
    elif fault == "frame_id":
        records["1"]["agt_frame_id"] = 99
    elif fault == "drift":
        records["1"]["timestamp"] = 1.5 + 0.25  # offset differs between frames
    path: Path = tmp_path / "object_pose.json"
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="census|disagrees|drift"):
        read_object_frames(path, clock)


def test_object_reader_tolerates_a_constant_clock_offset(tmp_path: Path) -> None:
    """keyboard_fix-sticky-key_910a stamps object_pose.json 2318.58 s before frame_info; index is the join key."""
    frames: list[FrameInfo] = [FrameInfo(0, 20, 1.0, []), FrameInfo(1, 21, 1.5, [])]
    clock: FrameClock = FrameClock(RecordingInfo(20, 2, 60.0, {}), frames, np.array([0, 500_000_000], dtype=np.int64), np.array([0, 1], dtype=np.int64))
    records: dict[str, dict[str, object]] = {
        str(i): dict(index=i, agt_frame_id=20 + i, timestamp=frame.timestamp - 2318.583333, missing_cameras=[], R=[], t=[], confidence=0.0)
        for i, frame in enumerate(frames)
    }
    path: Path = tmp_path / "object_pose.json"
    path.write_text(json.dumps(records))
    track: ObjectTrack = read_object_frames(path, clock)
    assert [frame.index for frame in track.frames] == [0, 1]
    assert track.clock_offset_s == pytest.approx(-2318.583333, abs=CLOCK_TOLERANCE_S)


def test_unposed_object_layer_retains_confidence_and_typed_nan_metrics(tmp_path: Path) -> None:
    clock: FrameClock = FrameClock(
        RecordingInfo(20, 1, 60.0, {}), [FrameInfo(0, 20, 1.0, [])], np.array([0], dtype=np.int64), np.array([0], dtype=np.int64)
    )
    target: Path = tmp_path / "object_pose.rrd"
    write_object_pose_layer(
        SequenceIdentity("show3d", ("S", "toy_hold_abcd")),
        "toy",
        clock,
        [ObjectFrame(0, 20, 1.0, [], [], [], 0.0)],
        ObjectSanity(0.0, float("nan"), float("nan")),
        target,
        clock_offset_s=0.0,
    )
    chunks: list[rr.experimental.Chunk] = read_chunks(target)
    assert [str(c.entity_path) for c in chunks if not c.is_static] == [schema.object_confidence_path("toy")]
    props: dict[str, object] = recording_properties(read_back(target), "object_pose")
    assert props["coverage"] == 0.0
    assert props["clock_offset_s"] == 0.0
    for name in ("in_ego_fov_fraction", "palm_dist_median_m"):
        value = props[name]
        assert isinstance(value, float)
        assert np.isnan(value)
    batch: pa.RecordBatch = next(c.to_record_batch() for c in chunks if c.is_static)
    for name in ("coverage", "in_ego_fov_fraction", "palm_dist_median_m"):
        assert batch.schema.field(name).type.value_type == pa.float64()


def test_mesh_download_matches_names_and_strips_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    census: dict = {
        str(100 + i): {"name": name} for i, name in enumerate(reversed([name for name in OBJECTS.values() if name is not None]))
    }
    payload: bytes = b'{"asset":{"version":"2.0"},"extensionsUsed":["KHR_texture_transform"]}'
    payload += b" " * (-len(payload) % 4)
    glb: bytes = struct.pack("<4sIII4s", b"glTF", 2, 20 + len(payload), len(payload), b"JSON") + payload
    fetched: list[list[str]] = []

    def revision(repo_id: str, revision: str | None) -> str:
        assert repo_id == MESH_REPO
        return "bop-test-sha"

    def fetch(repo_id: str, paths: Sequence[str], *, local_dir: Path, revision: str) -> Path:
        assert repo_id == MESH_REPO and revision == "bop-test-sha"
        fetched.append(list(paths))
        for name in paths:
            path: Path = local_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(json.dumps(census).encode() if name.endswith(".json") else glb)
        return local_dir

    monkeypatch.setattr(transports, "repo_revision", revision)
    monkeypatch.setattr(transports, "hf_fetch_files", fetch)
    download_meshes(tmp_path)
    assert fetched[0] == ["object_models/models_info.json"]
    assert {name for batch in fetched[1:] for name in batch} == {f"object_models/obj_{int(key):06d}.glb" for key in census}
    root: Path = tmp_path / "assets/hot3d_bop"
    assert mesh_ids(root)["keyboard"] == next(int(key) for key, value in census.items() if value["name"] == "keyboard")
    outputs: list[Path] = list((root / "stripped").glob("*.glb"))
    assert len(outputs) == 22
    assert not list((root / "object_models").glob("*.glb"))
    assert all(b"KHR_texture_transform" not in p.read_bytes() for p in outputs)
    before: dict[Path, int] = {p: p.stat().st_mtime_ns for p in outputs}
    download_meshes(tmp_path)
    assert len(fetched) == 23
    assert all(p.stat().st_mtime_ns == stamp for p, stamp in before.items())


@pytest.mark.parametrize(("alias", "has_object_pose"), [("keyboard", False), ("keyboard2", False), ("keyboard2", True)])
def test_convert_without_pending_object_pose_omits_mesh_and_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], alias: str, has_object_pose: bool
) -> None:
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "rrd"))
    identity: SequenceIdentity = SequenceIdentity("show3d", ("S", f"{alias}_hold_abcd"))
    dataset: Show3dDataset = Show3dDataset(Show3dConfig(root=tmp_path / "raw"))
    for layer in ("base", "properties", "object_pose"):
        target: Path = paths.rrd_path(paths.output_root(), layer=layer, identity=identity)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"existing")
    dataset.convert(
        identity,
        index_row(scene_id=identity.parts[1], has_object_pose=has_object_pose, has_hand_pose=False, has_caption=False),
        force=False,
    )
    assert "no object_mesh" not in capsys.readouterr().out
    assert not (tmp_path / "rrd/object_mesh").exists()


@pytest.mark.parametrize("census", [{"bad-id": {"name": "keyboard"}}, {"1": {"name": "unmapped-name"}}])
def test_mesh_census_error_names_source(tmp_path: Path, census: dict[str, dict[str, str]]) -> None:
    path: Path = tmp_path / "object_models/models_info.json"
    path.parent.mkdir()
    path.write_text(json.dumps(census))
    with pytest.raises(ValueError, match="models_info.json"):
        mesh_ids(tmp_path)



@pytest.mark.parametrize("alias", ["none", "keyboard2"])
def test_listed_unmapped_alias_has_no_mesh(alias: str) -> None:
    assert mesh_name(alias) is None


def test_unknown_mesh_alias_names_the_error() -> None:
    with pytest.raises(ValueError, match="unknown SHOW3D object alias: typo"):
        mesh_name("typo")


def test_convert_pending_unmapped_object_pose_prints_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "rrd"))
    identity: SequenceIdentity = SequenceIdentity("show3d", ("S", "keyboard2_hold_abcd"))
    raw: Path = tmp_path / "raw"
    scene: Path = raw / "scenes" / identity.sequence_key
    metadata: Path = scene / "metadata"
    metadata.mkdir(parents=True)
    frame: dict = dict(index=0, agt_frame_id=20, timestamp=1.0, missing_cameras=[])
    (metadata / "recording_info.json").write_text(json.dumps(dict(start_frame_id=20, num_frames=1, fps=60.0, resolution={})))
    (metadata / "frame_info.json").write_text(json.dumps([frame]))
    calibration: Path = scene / "camera_calibration"
    calibration.mkdir()
    for camera in HEADSET_CAMERAS:
        (calibration / f"{camera.source_name}.json").write_text(json.dumps(dict(
            ImageSizeX=100, ImageSizeY=100, fx=100.0, fy=100.0, cx=50.0, cy=50.0,
            DistortionModel="PinholePlane", T_WorldFromCamera_by_index={},
        )))
    poses: Path = raw / object_pose_file(identity.sequence_key)
    poses.parent.mkdir(parents=True)
    poses.write_text(json.dumps({"0": dict(**frame, R=[], t=[], confidence=0.0)}))
    for layer in ("base", "properties"):
        target: Path = paths.rrd_path(paths.output_root(), layer=layer, identity=identity)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"existing")
    dataset: Show3dDataset = Show3dDataset(Show3dConfig(root=raw))
    dataset.convert(
        identity, index_row(scene_id=identity.parts[1], has_hand_pose=False, has_caption=False), force=False
    )
    assert capsys.readouterr().out.splitlines() == [
        f"{identity.sequence_key}: no object_mesh: alias 'keyboard2' has no HOT3D mesh mapping",
        f"done {identity.sequence_key}: object_pose",
    ]
    assert paths.rrd_path(paths.output_root(), layer="object_pose", identity=identity).is_file()
    assert not paths.rrd_path(paths.output_root(), layer="object_mesh", identity=identity).exists()
