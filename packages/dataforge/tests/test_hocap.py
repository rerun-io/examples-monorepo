"""HOCap archive discovery and source-to-recording contracts."""

import io
import json
import struct
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import rerun as rr
import yaml
from conftest import blueprint_views, read_chunks
from jaxtyping import Float32, Float64, Int64
from numpy import ndarray
from PIL import Image

from dataforge import objects, writing
from dataforge.datasets.hocap import HocapConfig, HocapDataset
from dataforge.datasets.hocap_layers import write_base, write_hands
from dataforge.datasets.hocap_mesh import textured_glb
from dataforge.datasets.hocap_source import (
    EGO_RIG,
    EXO_RIGS,
    CalibrationExtrinsics,
    Color,
    HocapCamera,
    HoloLens,
    Metadata,
    RealSense,
    SequenceData,
    frame_times,
    pose_matrices,
    read_sequence,
)
from dataforge.hands import coco133_from_coco_hands, confidence_rule
from dataforge.identity import SequenceIdentity
from dataforge.timing import SequenceTimer
from dataforge.video_encoding import FrameSource


def scene(
    *,
    world_T_cam: Float64[ndarray, "e 4 4"] | None = None,
    mano: Float32[ndarray, "2 n 51"] | None = None,
    pv: Float32[ndarray, "n 7"] | None = None,
    times_ns: Int64[ndarray, "n"] | None = None,
    frame_indices: Int64[ndarray, "n"] | None = None,
) -> SequenceData:
    """Small source scene with explicit keyword overrides."""
    serials = tuple(f"cam{i}" for i in EXO_RIGS)
    return SequenceData(
        meta=Metadata(RealSense(list(serials), 640, 480), HoloLens("pv", 640, 480), "unused", "subject_5", ["a", "b", "c", "d"], ["right"], 2, 1),
        cameras=(*[HocapCamera(serial, rig, "exo") for rig, serial in zip(EXO_RIGS, serials, strict=True)], HocapCamera("pv", EGO_RIG, "ego")),
        intrinsics=(Color(640, 480, 100.0, 100.0, 320.0, 240.0),) * (len(EXO_RIGS) + 1),
        world_T_cam=np.tile(np.eye(4), (len(EXO_RIGS), 1, 1)) if world_T_cam is None else world_T_cam,
        mano=np.zeros((2, 2, 51), dtype=np.float32) if mano is None else mano,
        objects=np.zeros((4, 2, 7), dtype=np.float32),
        pv=np.zeros((2, 7), dtype=np.float32) if pv is None else pv,
        betas=np.zeros(10, dtype=np.float32),
        times_ns=np.array([0, 33333333], dtype=np.int64) if times_ns is None else times_ns,
        frame_indices=np.arange(2, dtype=np.int64) if frame_indices is None else frame_indices,
    )


def test_discovery_uses_poses_and_skips_absent_subject_archives(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with ZipFile(tmp_path / "poses.zip", "w") as archive:
        for key in ("subject_5/b", "subject_1/a", "subject_5/a"):
            archive.writestr(f"{key}/poses_m.npy", b"")
    with ZipFile(tmp_path / "subject_5.zip", "w"):
        pass
    (tmp_path / "subject_1/a").mkdir(parents=True)
    dataset = HocapConfig(root=tmp_path).setup()
    assert [identity.recording_id for identity, _ in dataset.discover()] == ["hocap__subject_5__a", "hocap__subject_5__b"]
    assert f"skip subject_1/a: missing image archive {tmp_path / 'subject_1.zip'}" in capsys.readouterr().out
    with pytest.raises(FileNotFoundError, match="subject_1.zip"):
        HocapConfig(root=tmp_path, sequences=("subject_1/a",)).setup().discover()


def test_clock_and_direct_hand_order() -> None:

    np.testing.assert_array_equal(frame_times(4), [0, 33333333, 66666667, 100000000])
    joints = np.full((2, 21, 3), np.nan, dtype=np.float32)
    joints[0] = np.arange(63, dtype=np.float32).reshape(21, 3)
    points, scores = confidence_rule(coco133_from_coco_hands(joints[::-1])[None], None)
    np.testing.assert_array_equal(points[0, 112:133], joints[0])
    assert np.isnan(points[0, :112]).all()
    assert (scores[0, :112] == 0).all()
    assert (scores[0, 112:] == 1).all()


def test_jpeg_pipe_keeps_encoded_input() -> None:

    assert FrameSource("jpeg").input_args(fps=30) == ["-f", "image2pipe", "-framerate", "30", "-c:v", "mjpeg", "-i", "pipe:0"]


def test_metadata_selects_extrinsics_and_checks_full_counts(tmp_path: Path) -> None:

    serials = [f"cam{i}" for i in reversed(EXO_RIGS)]
    meta = dict(
        realsense=dict(serials=serials, width=640, height=480),
        hololens=dict(serial="pv", pv_width=640, pv_height=480),
        extrinsics="chosen.yaml",
        subject_id="subject_5",
        object_ids=["a", "b", "c", "d"],
        mano_sides=["right"],
        num_frames=2,
        task_id=1,
    )
    with ZipFile(tmp_path / "s.zip", "w") as subject, ZipFile(tmp_path / "c.zip", "w") as calibration, ZipFile(tmp_path / "p.zip", "w") as poses:
        subject.writestr("subject_5/test/meta.yaml", yaml.safe_dump(meta))
        for serial in [*serials, "pv"]:
            for i in range(2):
                subject.writestr(f"subject_5/test/{serial}/color_{i:06d}.jpg", b"unused")
            calibration.writestr(
                f"calibration/intrinsics/{serial}.yaml", yaml.safe_dump(dict(color=dict(width=640, height=480, fx=100, fy=100, ppx=320, ppy=240)))
            )
        calibration.writestr("calibration/mano/subject_5.yaml", yaml.safe_dump(dict(betas=[0.0] * 10)))
        tag = [1, 0, 0, 1, 0, 1, 0, 2, 0, 0, 1, 3]
        camera = [1, 0, 0, 4, 0, 1, 0, 6, 0, 0, 1, 8]
        calibration.writestr(
            "calibration/extrinsics/chosen.yaml", yaml.safe_dump(dict(extrinsics={"tag_1": tag, **{serial: camera for serial in serials}}))
        )
        for name, shape in (("m", (2, 2, 51)), ("o", (4, 2, 7)), ("pv", (2, 7))):
            buffer = io.BytesIO()
            np.save(buffer, np.zeros(shape, dtype=np.float32))
            poses.writestr(f"subject_5/test/poses_{name}.npy", buffer.getvalue())
        scene = read_sequence(subject, calibration, poses, "subject_5/test", frame_limit=1, members=frozenset(subject.namelist()))
        assert tuple(camera.serial for camera in scene.cameras) == (*sorted(serials), "pv")
        assert scene.count == 1
        np.testing.assert_array_equal(scene.world_T_cam[0, :3, 3], [3, 4, 5])
        with pytest.raises(ValueError, match="frame_limit"):
            read_sequence(subject, calibration, poses, "subject_5/test", frame_limit=0, members=frozenset(subject.namelist()))
        calibration.writestr("calibration/extrinsics/missing.yaml", yaml.safe_dump(dict(extrinsics={"tag_1": tag})))
        meta["extrinsics"] = "missing.yaml"
        with ZipFile(tmp_path / "missing-subject.zip", "w") as missing_subject:
            missing_subject.writestr("subject_5/test/meta.yaml", yaml.safe_dump(meta))
            with pytest.raises(ValueError, match="subject_5/test: extrinsics missing.yaml lacks cam0"):
                read_sequence(missing_subject, calibration, poses, "subject_5/test", members=frozenset(subject.namelist()))
        subject.writestr("subject_5/test/pv/color_000002.jpg", b"extra")
        with pytest.raises(ValueError, match="color frame indices"):
            read_sequence(subject, calibration, poses, "subject_5/test", frame_limit=1, members=frozenset(subject.namelist()))


def test_invalid_pose_rows_do_not_inherit_a_previous_pose() -> None:

    rows = np.array([[0, 0, 0, 1, 1, 2, 3], [-1] * 7, [np.nan] * 7], dtype=np.float32)
    result = pose_matrices(rows)
    np.testing.assert_array_equal(result[0, :3, 3], [1, 2, 3])
    assert np.isnan(result[1:]).all()
    assert np.isnan(pose_matrices(rows[1:])).all()


def test_missing_camera_labels_remain_missing_but_world_joints_use_another_camera(tmp_path: Path) -> None:

    transforms = np.tile(np.eye(4), (len(EXO_RIGS), 1, 1))
    transforms[1, :3, 3] = [10, 20, 30]
    mano_rows = np.zeros((2, 2, 51), dtype=np.float32)
    mano_rows[:, 1] = -1
    data = scene(world_T_cam=transforms, mano=mano_rows)
    joints = np.full((2, 21, 3), -1, dtype=np.float32)
    joints[0] = [1, 2, 3]
    pixels = np.full((2, 21, 2), -1, dtype=np.int64)
    pixels[0] = [100, 200]
    buffer = io.BytesIO()
    np.savez(buffer, hand_joints_3d=joints, hand_joints_2d=pixels)
    target = tmp_path / "hand.rrd"
    with ZipFile(tmp_path / "labels.zip", "w") as archive:
        archive.writestr("subject_5/test/cam1/label_000000.npz", buffer.getvalue())
        with writing.atomic_recording(target, recording_id="hocap__subject_5__test", send_properties=False) as recording:
            write_hands(recording, data, archive, "subject_5/test", members=frozenset(archive.namelist()))
    chunks = read_chunks(target)
    xyz = next(c.to_record_batch() for c in chunks if c.entity_path == "/world/gt/coco133_xyz" and not c.is_static)
    np.testing.assert_array_equal(xyz.column("Points3D:positions").to_pylist()[0][112], [11, 22, 33])
    assert np.isnan(xyz.column("Points3D:positions").to_pylist()[1]).all()
    assert xyz.column(next(name for name in xyz.schema.names if name.endswith(":confidences"))).to_pylist()[0][112] == 1.0
    assert xyz.column(next(name for name in xyz.schema.names if name.endswith(":confidences"))).to_pylist()[1] == [0.0] * 133
    assert not any(f"rig_{EGO_RIG:02}" in c.entity_path for c in chunks)
    for c in chunks:
        if not c.is_static:
            assert set(c.timeline_names) == {"frame_index", "video_time"}
    uv = next(c.to_record_batch() for c in chunks if "rig_00" in c.entity_path and not c.is_static)
    assert np.isnan(uv.column("Points2D:positions").to_pylist()).all()
    mano = next(c.to_record_batch() for c in chunks if c.entity_path == "/world/gt/hands/right/mano" and not c.is_static)
    assert len(mano.column("pca_coefficients").to_pylist()[0][0]) == 45
    assert np.isnan(mano.column("pca_coefficients").to_pylist()[1]).all()


def test_textured_obj_preserves_uv_seams(tmp_path: Path) -> None:

    texture = io.BytesIO()
    Image.fromarray(np.full((2, 2, 3), [255, 0, 0], dtype=np.uint8)).save(texture, format="PNG")
    with ZipFile(tmp_path / "models.zip", "w") as archive:
        archive.writestr(
            "models/a/textured_mesh.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 2\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1/1 2/2/1 3/3/1\nf 1/3/1 2/2/1 3/1/1\n"
        )
        archive.writestr("models/a/textured_mesh.mtl", "map_Kd textured_mesh_0.png\n")
        archive.writestr("models/a/textured_mesh_0.png", texture.getvalue())
        glb = textured_glb(archive, "a")
    assert struct.unpack_from("<4sII", glb) == (b"glTF", 2, len(glb))
    size, kind = struct.unpack_from("<I4s", glb, 12)
    assert kind == b"JSON" and size % 4 == 0
    doc = json.loads(glb[20 : 20 + size])
    bin_size, kind = struct.unpack_from("<I4s", glb, 20 + size)
    binary = glb[28 + size :]
    assert kind == b"BIN\0" and bin_size == len(binary) and bin_size % 4 == 0
    arrays = []
    for accessor in doc["accessors"]:
        view = doc["bufferViews"][accessor["bufferView"]]
        assert view["byteOffset"] % 4 == 0
        dtype = "<u4" if accessor["componentType"] == 5125 else "<f4"
        arrays.append(np.frombuffer(binary[view["byteOffset"] : view["byteOffset"] + view["byteLength"]], dtype=dtype))
    vertices, normals, uv, indices = arrays
    vertices, normals, uv = vertices.reshape(-1, 3), normals.reshape(-1, 3), uv.reshape(-1, 2)
    assert len(vertices) == 5
    np.testing.assert_array_equal(vertices[:2], [[0, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(uv[:2], [[0, 1], [0, 0]])
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)
    assert len(indices) == 6 and indices.max() < len(vertices)
    assert doc["accessors"][0]["min"] == [0, 0, 0]
    assert doc["accessors"][0]["max"] == [1, 1, 0]
    view = doc["bufferViews"][doc["images"][0]["bufferView"]]
    assert binary[view["byteOffset"] : view["byteOffset"] + view["byteLength"]] == texture.getvalue()
    assert doc["materials"][0]["pbrMetallicRoughness"]["metallicFactor"] == 0.0
    with writing.atomic_recording(tmp_path / "mesh.rrd", recording_id="test", send_properties=False) as recording:
        objects.log_object_mesh(
            recording,
            "a",
            times_ns=np.array([0], dtype=np.int64),
            frame_indices=np.array([0], dtype=np.int64),
            asset=rr.Asset3D(contents=glb, media_type="model/gltf-binary"),
            confidence=np.ones(1, dtype=np.float32),
            posed=np.ones(1, dtype=bool),
            trust_threshold=0.0,
        )
    batches = [c.to_record_batch() for c in read_chunks(tmp_path / "mesh.rrd")]
    asset = next(b for b in batches if "Asset3D:blob" in b.schema.names)
    assert bytes(asset.column("Asset3D:blob").to_pylist()[0][0]) == glb
    assert asset.column("Asset3D:media_type").to_pylist() == [["model/gltf-binary"]]
    assert any("Scalars:scalars" in b.schema.names for b in batches)


def test_converter_refuses_output_under_raw_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "converted"))
    dataset = HocapConfig(root=tmp_path).setup()
    with pytest.raises(ValueError, match="refusing to write beneath raw root"):
        dataset.convert(SequenceIdentity("hocap", ("subject_5", "test")), "subject_5/test", force=True)


def test_blueprints_have_nine_cameras_and_table_excludes_other_video() -> None:

    dataset = HocapConfig().setup()
    assert len(blueprint_views(dataset.default_blueprint())) == 10
    assert len(blueprint_views(dataset.table_blueprint())) == 2


def test_prefix_targets_never_overlap_full_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path))
    identity = SequenceIdentity("hocap", ("subject_5", "test"))
    full = HocapConfig().setup().targets(identity)
    limited = HocapConfig(frame_limit=2).setup().targets(identity)
    assert set(full.values()).isdisjoint(limited.values())
    assert {p.parent for p in full.values()}.isdisjoint(p.parent for p in limited.values())
    assert all(p.is_relative_to(tmp_path / "preview-first2") for p in limited.values())


@pytest.mark.parametrize("dimensions", [2, 3])
def test_shared_coco_adapter_copies_only_shipped_slots(dimensions: int) -> None:
    joints = np.ones((2, 21, dimensions), dtype=np.float32)
    joints[1] = 2.0
    joints[0, 3] = np.nan
    result = coco133_from_coco_hands(joints)
    np.testing.assert_array_equal(result[91:112], joints[0])
    np.testing.assert_array_equal(result[112:], joints[1])
    assert np.isnan(result[:91]).all()


@pytest.mark.parametrize("rows, message", [({}, "tag_1"), ({"tag_1": [0.0]}, "12 floats")])
def test_calibration_rejects_malformed_extrinsics(rows: dict[str, list[float]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CalibrationExtrinsics(rows)


def test_pv_missing_rows_are_dense_nan_quaternions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = np.array([[0, 0, 0, 1, 1, 2, 3], [-1] * 7, [np.nan] * 7, [0] * 7], dtype=np.float32)
    data = scene(pv=rows, times_ns=frame_times(4), frame_indices=np.arange(4, dtype=np.int64))
    monkeypatch.setattr("dataforge.datasets.hocap_layers.encode_frames_to_mp4", lambda *_args, **_kwargs: 4)
    monkeypatch.setattr("dataforge.datasets.hocap_layers.log_video_stream", lambda _recording, clip, *_args, **_kwargs: clip.touch())
    target = tmp_path / "base.rrd"
    with ZipFile(tmp_path / "subject.zip", "w") as archive, writing.atomic_recording(target, recording_id="pv") as recording:
        write_base(recording, data, archive, "subject_5/test", SequenceIdentity("hocap", ("subject_5", "test")), SequenceTimer(), tmp_path / "work")
    batch = next(c.to_record_batch() for c in read_chunks(target) if c.entity_path == f"/world/rig_{EGO_RIG:02}" and not c.is_static)
    quaternions = batch.column("Transform3D:quaternion").to_pylist()
    np.testing.assert_array_equal(quaternions[0], [[0, 0, 0, 1]])
    assert np.isnan(quaternions[1:]).all()
    assert np.isnan(batch.column("Transform3D:translation").to_pylist()[1:]).all()
    assert list((tmp_path / "work").iterdir()) == []


def test_archive_handle_and_member_index_are_reused(tmp_path: Path) -> None:
    with ZipFile(tmp_path / "poses.zip", "w") as archive:
        archive.writestr("subject_5/test/poses_m.npy", b"")
    with ZipFile(tmp_path / "subject_5.zip", "w"):
        pass
    dataset = HocapConfig(root=tmp_path).setup()
    assert isinstance(dataset, HocapDataset)
    first = dataset.archive("poses")
    dataset.discover()
    dataset.discover()
    assert dataset.archive("poses") is first
    assert first.members == frozenset({"subject_5/test/poses_m.npy"})
