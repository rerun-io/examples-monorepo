"""Driver-run HOT3D integration and golden checks on the two staged captures."""

import csv
import json
import os
import struct
from dataclasses import replace
from itertools import islice, product
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pytest
import rerun.chunk as rrc
from conftest import read_chunks
from scipy.spatial.transform import Rotation

from dataforge import aria, paths, schema, writing
from dataforge.datasets.hot3d import Hot3dAriaConfig, Hot3dQuest3Config
from dataforge.datasets.hot3d_hands import evaluate_hands, mesh_vertices
from dataforge.datasets.hot3d_layers import LayerWriter
from dataforge.datasets.hot3d_source import DEVICES, URL_LIST_DATE, Device, Hot3dSource, Metadata
from dataforge.datasets.hot3d_vrs import Scene, read_scene
from dataforge.datasets.show3d_hands import read_hand_profile
from dataforge.datasets.show3d_source import read_json
from dataforge.identity import SequenceIdentity
from dataforge.jpeg import decode_jpeg_frames, jpeg_frame_source
from dataforge.objects import strip_texture_transform
from dataforge.timing import SequenceTimer
from dataforge.video_encoding import AV1_CQ, AV1_GOP, encode_frames_to_mp4
from dataforge.vrs import VrsImageReader


@pytest.fixture(params=[("aria", "P0001_4bf4e21a"), ("quest3", "P0002_5a9cfa51")])
def hot3d_source(request: pytest.FixtureRequest) -> tuple[Device, Path]:
    device, sequence = request.param
    root = paths.raw_root() / "hot3d"
    source = root / device / sequence
    required = [
        source / name
        for name in (
            "recording.vrs",
            "metadata.json",
            "camera_models.json",
            "headset_trajectory.csv",
            "dynamic_objects.csv",
            "umetrack_hand_pose_trajectory.jsonl",
            "umetrack_hand_user_profile.json",
            "mano_hand_pose_trajectory.jsonl",
        )
    ]
    required += [
        source / f"masks/mask_{name}.csv"
        for name in (
            "good_exposure",
            "hand_pose_available",
            "hand_visible",
            "headset_pose_available",
            "object_pose_available",
            "object_visible",
            "qa_pass",
        )
    ]
    if DEVICES[device].has_timecode_mapping:
        required.append(source / "timecode_devicetime_mapping.csv")
    required += [root / "assets/instance.json", root / f"Hot3D{DEVICES[device].url_label}_download_urls-{URL_LIST_DATE}.json"]
    for path in required:
        if not path.is_file():
            pytest.skip(f"HOT3D asset absent: {path}")
    metadata = read_json(source / "metadata.json", Metadata)
    for alias in metadata.object_uids:
        path = root / f"assets/{alias}.glb"
        if not path.is_file():
            pytest.skip(f"HOT3D asset absent: {path}")
    return device, source


@pytest.mark.integration
def test_first_sixty_frames_all_layers(
    hot3d_source: tuple[Device, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc_ffmpeg: Path
) -> None:
    device, source = hot3d_source
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("DATAFORGE_FFMPEG", str(nvenc_ffmpeg))
    config_type = Hot3dAriaConfig if device == "aria" else Hot3dQuest3Config
    dataset = config_type(root=source.parent.parent, sequences=(source.name,), frame_limit=60).setup()
    identity, found = dataset.discover()[0]
    dataset.convert(identity, found, force=True)
    targets = dataset.targets(identity)
    assert set(targets) == {"base", "hand_pose", "hand_mesh", "projections", "object_pose", "object_mesh"}
    scene = read_scene(Hot3dSource(source, read_json(source / "metadata.json", Metadata)), device, 60)
    for layer, target in targets.items():
        assert rrc.RrdReader(target).recordings()[0].recording_id == identity.recording_id
        chunks = read_chunks(target)
        assert chunks
        if layer not in ("base", "projections"):
            assert not any(chunk.entity_path.startswith("/__properties") for chunk in chunks)
        for chunk in chunks:
            if chunk.is_static:
                continue
            assert "video_time" in chunk.timeline_names
            # Native IMU samples use their own clock with no invented frame index.
            if "/imu_" not in chunk.entity_path:
                assert "frame_index" in chunk.timeline_names
                assert all(0 <= frame < 60 for frame in chunk.to_record_batch().column("frame_index").to_pylist())
    base = read_chunks(targets["base"])
    for index, camera in enumerate(scene.cameras):
        chunks = [
            chunk
            for chunk in base
            if chunk.entity_path == schema.video_path(0, index) and "VideoStream:sample" in chunk.to_record_batch().schema.names
        ]
        stamps = np.concatenate([chunk.to_record_batch().column("video_time").cast(pa.int64()).to_numpy() for chunk in chunks])
        np.testing.assert_array_equal(np.sort(stamps), camera.times_ns)
        assert len(stamps) == 60
    pose = read_chunks(targets["hand_pose"])
    assert not any("coco133_uv" in chunk.entity_path for chunk in pose)
    points = [chunk for chunk in pose if chunk.entity_path == schema.coco133_xyz_path() and not chunk.is_static]
    assert sum(chunk.num_rows for chunk in points) == len(scene.times_ns)
    assert len(scene.hands) > 170 if device == "aria" else len(scene.hands) == 60
    assert any("/imu_00/" in chunk.entity_path for chunk in base) == (device == "aria")
    assert any("/imu_01/" in chunk.entity_path for chunk in base) == (device == "aria")
    if device == "quest3":
        rig = [chunk.to_record_batch() for chunk in base if chunk.entity_path == schema.rig_path(0) and not chunk.is_static]
        rows = [row for batch in rig for row in batch.to_pylist() if row["frame_index"] == 5]
        assert len(rows) == 1
        assert np.isnan(rows[0]["Transform3D:translation"]).all()
    original = {layer: target.stat().st_mtime_ns for layer, target in targets.items()}
    dataset.convert(identity, found, force=False)
    assert original == {layer: target.stat().st_mtime_ns for layer, target in targets.items()}


@pytest.mark.integration
def test_nvenc_native_color_gray_rotation_and_no_b_frames(hot3d_source: tuple[Device, Path], tmp_path: Path, nvenc_ffmpeg: Path) -> None:
    device, source = hot3d_source
    scene = read_scene(Hot3dSource(source, read_json(source / "metadata.json", Metadata)), device, 3)
    for index, camera in enumerate(scene.cameras):
        model = camera.model
        images = []
        for _, image in aria.iter_frames(scene.provider, model.stream_id):
            images.append(image)
            if len(images) == 3:
                break
        clip = tmp_path / f"{index}.mp4"
        jpegs = [record.image for record in islice(VrsImageReader(source / "recording.vrs", model.stream_id).images(), 3)]
        frame_source = jpeg_frame_source(jpegs[0])
        count = encode_frames_to_mp4(
            decode_jpeg_frames(jpegs, source=frame_source),
            clip,
            source=frame_source,
            fps=30,
            cq=AV1_CQ,
            gop=AV1_GOP,
            rotate_cw_quarter_turns=1,
            filter_threads=1,
            ffmpeg=nvenc_ffmpeg,
        )
        assert count == 3
        with av.open(str(clip)) as container:
            stream = container.streams.video[0]
            assert stream.codec_context.codec.canonical_name == "av1"
            packets = [packet for packet in container.demux(stream) if packet.pts is not None]
            assert len(packets) == 3
            assert all(packet.pts == packet.dts for packet in packets)
        with av.open(str(clip)) as container:
            decoded = next(container.decode(video=0))
            assert (decoded.width, decoded.height) == (model.height, model.width)
            pixels = decoded.to_ndarray(format="rgb24").astype(np.float32)
            if model.stream_id == "214-1":
                assert np.mean(np.abs(pixels[:, :, 0] - pixels[:, :, 2])) > 1.0
            else:
                assert np.max(np.abs(pixels[:, :, 0] - pixels[:, :, 2])) <= 2.0
            native = images[0].astype(np.float32)
            if native.ndim == 2:
                native = np.repeat(native[:, :, None], 3, axis=2)
            assert np.mean(np.abs(pixels - np.rot90(native, -1))) < 15.0


@pytest.mark.golden
def test_fk_matches_simplecv_at_matching_stamps(hot3d_source: tuple[Device, Path]) -> None:
    device, source = hot3d_source
    reference_root: Path = Path(os.environ.get("DATAFORGE_HOT3D_REFERENCE_ROOT", "/mnt/nas/datasets/exoego-forge-catalog-rig"))
    reference = reference_root / f"hot3d-{device}" / f"{source.name}.rrd"
    if not reference.is_file():
        pytest.skip(f"HOT3D simplecv reference asset absent: {reference}")
    try:
        with reference.open("rb") as stream:
            stream.read(1)
    except PermissionError:
        pytest.skip(f"HOT3D simplecv reference asset unreadable: {reference}")
    scene = read_scene(Hot3dSource(source, read_json(source / "metadata.json", Metadata)), device)
    profile = read_hand_profile(source / "umetrack_hand_user_profile.json")
    actual = evaluate_hands(profile.model, scene.hands, scene.times_ns)
    origin = int(scene.cameras[0].times_ns[0])
    # Raw rows determine presence independently of FK output and the reference.
    raw_presence = np.zeros((len(scene.times_ns), 133), dtype=np.bool_)
    for index, stamp in enumerate(scene.times_ns):
        row = scene.hands.get(int(stamp))
        for side, wrist, start in (("0", 9, 91), ("1", 10, 112)):
            if row is not None and side in row.hand_poses:
                raw_presence[index, wrist] = True
                raw_presence[index, start : start + 21] = True
    np.testing.assert_array_equal(np.isfinite(actual.positions).all(axis=2), raw_presence)
    compared = 0
    unmatched_reference = 0
    matched = set()
    carried_joints = 0
    max_error = 0.0
    for chunk in rrc.RrdReader(reference).stream():
        if chunk.entity_path != schema.coco133_xyz_path() or chunk.is_static:
            continue
        batch = chunk.to_record_batch()
        if "Points3D:positions" not in batch.schema.names:
            continue
        times = batch.column("video_time").cast(pa.int64()).to_pylist()
        for stamp, positions in zip(times, batch.column("Points3D:positions").to_pylist(), strict=True):
            shifted = int(stamp) + origin
            right = int(np.searchsorted(scene.times_ns, shifted))
            candidates = [i for i in (right - 1, right) if 0 <= i < len(scene.times_ns)]
            nearest = min(candidates, key=lambda i: abs(int(scene.times_ns[i]) - shifted))
            if abs(int(scene.times_ns[nearest]) - shifted) > 1:
                unmatched_reference += 1
                continue
            matched.add(nearest)
            expected = np.asarray(positions, dtype=np.float32)
            ours = actual.positions[nearest]
            reference_present = np.isfinite(expected).all(axis=1)
            present = raw_presence[nearest]
            # Facts §11: simplecv carried hand landmarks forward on dropout.
            # This exception applies only to slots of a hand absent in the raw row.
            allowed_carry = np.zeros(133, dtype=np.bool_)
            for wrist, start in ((9, 91), (10, 112)):
                if not present[wrist]:
                    allowed_carry[wrist] = True
                    allowed_carry[start : start + 21] = True
            assert not np.any(present & ~reference_present), f"reference lost raw joints at {scene.times_ns[nearest]}"
            assert not np.any(reference_present & ~present & ~allowed_carry), f"unexplained reference joints at {scene.times_ns[nearest]}"
            carried_joints += int(np.sum(reference_present & ~present))
            valid = present & reference_present
            if np.any(valid):
                error = float(np.linalg.norm(ours[valid] - expected[valid], axis=1).max())
                max_error = max(max_error, error)
                compared += 1
    print(
        f"{device}/{source.name}: {compared} matching rows, max FK error {max_error:.9f} m; "
        f"unmatched reference stamps={unmatched_reference}, unmatched census stamps={len(scene.times_ns) - len(matched)}, "
        f"documented carry-forward joints={carried_joints}"
    )
    assert compared > len(scene.cameras[0].times_ns) * 0.9
    assert max_error <= 1e-4


@pytest.mark.golden
def test_mesh_skinning_and_fk_landmarks_share_world_frame(hot3d_source: tuple[Device, Path]) -> None:
    device, source = hot3d_source
    scene = read_scene(Hot3dSource(source, read_json(source / "metadata.json", Metadata)), device, 60)
    profile = read_hand_profile(source / "umetrack_hand_user_profile.json")
    batch = evaluate_hands(profile.model, scene.hands, scene.times_ns)
    model = profile.model
    # Append each exact rest landmark as a mesh probe, retaining its sparse skin
    # weights. Compare the mesh path with FK; anatomical landmarks need not lie
    # exactly on a surface vertex, so nearest-vertex distance is not this test.
    weights = np.zeros((21, model.dense_bone_weights.shape[1]), dtype=np.float32)
    for landmark, (indices, values) in enumerate(zip(model.landmark_rest_bone_indices, model.landmark_rest_bone_weights, strict=True)):
        np.add.at(weights[landmark], indices, values)
    augmented = replace(
        model,
        mesh_vertices=np.concatenate([model.mesh_vertices, model.landmark_rest_positions]),
        dense_bone_weights=np.concatenate([model.dense_bone_weights, weights]),
    )
    for side, wrist_slot in [(0, 91), (1, 112)]:
        present = np.isfinite(batch.angles[side]).all(axis=1)
        assert np.any(present)
        vertices = mesh_vertices(augmented, batch.angles[side, present], batch.wrists[side, present], side)
        regular = mesh_vertices(model, batch.angles[side, present], batch.wrists[side, present], side)
        np.testing.assert_allclose(vertices[:, : len(model.mesh_vertices)], regular, atol=1e-4, rtol=0.0)
        # UmeTrack landmark 5 is the wrist, mapped into COCO's first hand slot.
        np.testing.assert_allclose(vertices[:, -21 + 5], batch.positions[present, wrist_slot], atol=1e-4, rtol=0.0)
        from dataforge.hands import coco133_from_hands

        for index, points in enumerate(vertices[:, -21:]):
            pair = np.full((2, 21, 3), np.nan, dtype=np.float32)
            pair[side] = points
            mapped, _ = coco133_from_hands(pair, np.ones(2, dtype=np.float32))
            valid = np.isfinite(mapped).all(axis=1)
            np.testing.assert_allclose(mapped[valid], batch.positions[present][index, valid], atol=1e-4, rtol=0.0)


@pytest.mark.golden
def test_object_mesh_placement_matches_raw_one_frame(hot3d_source: tuple[Device, Path], tmp_path: Path) -> None:
    device, source = hot3d_source
    scene: Scene = read_scene(Hot3dSource(source, read_json(source / "metadata.json", Metadata)), device, 1)
    identity = SequenceIdentity(f"hot3d-{device}", (source.name,))
    assets = source.parent.parent / "assets"
    outputs = {}
    writer = LayerWriter(scene, identity, SequenceTimer(), assets)
    for layer in ("object_pose", "object_mesh"):
        target = tmp_path / f"{layer}.rrd"
        with writing.atomic_recording(target, recording_id=identity.recording_id, send_properties=False) as recording:
            writer.write_layer(layer, recording)
        outputs[layer] = read_chunks(target)
    with (source / "dynamic_objects.csv").open() as stream:
        raw = next(csv.DictReader(stream))
    alias = raw["object_uid"]
    expected_rotation = Rotation.from_quat([float(raw[key]) for key in ("q_wo_x", "q_wo_y", "q_wo_z", "q_wo_w")]).as_matrix()
    expected_translation = np.array([float(raw[key]) for key in ("t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]")])
    stamp = int(raw["timestamp[ns]"])
    if DEVICES[device].has_timecode_mapping:
        with (source / "timecode_devicetime_mapping.csv").open() as stream:
            mapping = {int(row["timecode_ns"]): int(row["devicetime_ns"]) for row in csv.DictReader(stream)}
        stamp = mapping[stamp]
    pose_rows = []
    for chunk in outputs["object_pose"]:
        if chunk.entity_path == schema.objects_path(alias) and not chunk.is_static:
            table = chunk.to_record_batch()
            for time, row in zip(table.column("video_time").cast(pa.int64()).to_pylist(), table.to_pylist(), strict=True):
                if time == stamp and "Transform3D:translation" in row:
                    pose_rows.append(row)
    assert len(pose_rows) == 1
    row = pose_rows[0]
    translation = np.asarray(row["Transform3D:translation"]).reshape(3)
    rotation = Rotation.from_quat(np.asarray(row["Transform3D:quaternion"]).reshape(4)).as_matrix()
    blobs = [
        chunk.to_record_batch().column("Asset3D:blob")[0].as_py()
        for chunk in outputs["object_mesh"]
        if chunk.entity_path == schema.object_mesh_path(alias) and "Asset3D:blob" in chunk.to_record_batch().schema.names
    ]
    assert len(blobs) == 1
    native = (assets / f"{alias}.glb").read_bytes()
    # An Arrow row holds a list of blobs, each itself a list<uint8>.
    assert len(blobs[0]) == 1
    assert np.asarray(blobs[0][0], dtype=np.uint8).tobytes() == strip_texture_transform(native)
    # The asset is byte-preserved apart from the unsupported extension, so its
    # internal node transforms and units cannot change. Check placement on the
    # eight shipped POSITION-bound corners (no optional mesh package needed).
    length = struct.unpack_from("<I", native, 12)[0]
    document = json.loads(native[20 : 20 + length])
    primitive = document["meshes"][0]["primitives"][0]
    accessor = document["accessors"][primitive["attributes"]["POSITION"]]
    vertices = np.array(list(product(*zip(accessor["min"], accessor["max"], strict=True))))
    assert vertices.shape == (8, 3)
    expected = vertices @ expected_rotation.T + expected_translation
    placed = vertices @ rotation.T + translation
    np.testing.assert_allclose(placed, expected, atol=1e-4, rtol=0.0)


@pytest.mark.integration
def test_projection_preview_entities_properties_and_bounds(
    hot3d_source: tuple[Device, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc_ffmpeg: Path
) -> None:
    device, source = hot3d_source
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("DATAFORGE_FFMPEG", str(nvenc_ffmpeg))
    config_type = Hot3dAriaConfig if device == "aria" else Hot3dQuest3Config
    dataset = config_type(root=source.parent.parent, sequences=(source.name,), frame_limit=30).setup()
    identity, found = dataset.discover()[0]
    dataset.convert(identity, found, force=True)
    target = dataset.targets(identity)[paths.PROJECTIONS_LAYER]
    assert target.is_file()
    chunks = read_chunks(target)
    scene = read_scene(found, device, 30)
    entities = {chunk.entity_path for chunk in chunks if "coco133_uv" in chunk.entity_path}
    assert entities == {schema.coco133_uv_projected_path(0, cam) for cam in range(len(scene.cameras))}
    assert len(entities) == (3 if device == "aria" else 2)
    properties = [chunk for chunk in chunks if chunk.entity_path == "/__properties/projections"]
    assert properties and all(chunk.is_static for chunk in properties)
    fields = {key: value for chunk in properties for row in chunk.to_record_batch().to_pylist() for key, value in row.items()}
    assert fields["derived_from"] == ["coco133_xyz"]
    assert fields["camera_model"] == ["FISHEYE624"]
    finite_count = 0
    for cam, camera in enumerate(scene.cameras):
        rows = [chunk.to_record_batch() for chunk in chunks if chunk.entity_path == schema.coco133_uv_projected_path(0, cam) and not chunk.is_static]
        stamps = np.concatenate([row.column("video_time").cast(pa.int64()).to_numpy() for row in rows])
        order = np.argsort(stamps)
        np.testing.assert_array_equal(stamps[order], scene.times_ns)
        frames = np.concatenate([row.column("frame_index").to_numpy() for row in rows])
        np.testing.assert_array_equal(frames[order], scene.frame_indices)
        pixels = np.concatenate([np.asarray(row.column("Points2D:positions").to_pylist()) for row in rows])
        finite = np.isfinite(pixels).all(axis=-1)
        valid = pixels[finite]
        width, height = camera.calibration.get_image_size()
        assert np.all((valid[:, 0] >= 0) & (valid[:, 0] < width))
        assert np.all((valid[:, 1] >= 0) & (valid[:, 1] < height))
        finite_count += len(valid)
    assert finite_count > 0
