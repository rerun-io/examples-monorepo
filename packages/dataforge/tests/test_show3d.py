"""SHOW3D's file boundaries and the recording a consumer reads."""

from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import rerun as rr
from conftest import SHOW3D_RAW, index_row, read_back, read_chunks, recording_properties
from jaxtyping import Float64, UInt8
from numpy import ndarray
from serde import SerdeError, from_dict

from dataforge import schema
from dataforge.datasets.base import DataforgeDataset
from dataforge.datasets.show3d import Show3dConfig, pane_contents, world_contents
from dataforge.datasets.show3d_calibration import HeadsetCalibration, HeadsetRig, RigCalibration, headset_rig
from dataforge.datasets.show3d_layers import write_base_layer
from dataforge.datasets.show3d_source import CAMERAS, BlurInfo, IndexRow
from dataforge.identity import SequenceIdentity


def test_source_schemas_accept_legacy_and_new_pose_contracts() -> None:
    pose: dict = dict(index=0, agt_frame_id=20, timestamp=1.0, T_WorldFromCamera=np.eye(4).tolist(), is_synthesized=False)
    calibration: dict = dict(
        ImageSizeX=64,
        ImageSizeY=48,
        fx=30.0,
        fy=30.0,
        cx=32.0,
        cy=24.0,
        DistortionModel="PinholePlane",
        T_WorldFromCamera_by_index={"0": pose},
        upstream_extra=1,
    )
    old: HeadsetCalibration = from_dict(HeadsetCalibration, calibration)
    assert old.pose_contract_version is None
    assert old.T_WorldFromCamera_by_index["0"].is_pose_valid is None
    pose.update(pose_source="mocap", is_pose_valid=True)
    calibration["pose_contract_version"] = 1
    new: HeadsetCalibration = from_dict(HeadsetCalibration, calibration)
    assert new.T_WorldFromCamera_by_index["0"].is_pose_valid is True
    assert new.T_WorldFromCamera_by_index["0"].pose_source == "mocap"
    assert from_dict(BlurInfo, {"blur_boxes": {}, "extra": 3}).blur_boxes == {}


def test_discovery_orders_object_scenes_then_train_then_test_and_skips_empty(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for split, rows in (("train", [("z", 5, False), ("object", 4, True), ("empty", 0, False)]), ("test", [("a", 3, False)])):
        pq.write_table(
            pa.Table.from_pylist(
                [
                    dict(
                        subject_id="AZH822",
                        scene_id=f"{name}_action_abcd",
                        num_frames=count,
                        has_object_pose=objects,
                        **{f"has_{camera}": True for camera in ("headset0", "headset1", *(f"rig{i}" for i in range(8)))},
                        has_hand_pose=True,
                        has_caption=False,
                    )
                    for name, count, objects in rows
                ]
            ),
            tmp_path / f"dataset_index_{split}.parquet",
        )
    dataset: DataforgeDataset = Show3dConfig(root=tmp_path).setup()
    pairs: list[tuple[SequenceIdentity, IndexRow]] = dataset.discover()
    assert [identity.parts[1] for identity, _ in pairs] == ["object_action_abcd", "z_action_abcd", "a_action_abcd"]
    assert [source.split for _, source in pairs] == ["train", "train", "test"]
    assert "empty" in capsys.readouterr().out


def test_headset_rig_fits_mm_and_rejects_nonrigid_scene() -> None:
    camera: dict = dict(ImageSizeX=64, ImageSizeY=48, fx=30.0, fy=30.0, cx=32.0, cy=24.0, DistortionModel="PinholePlane")
    left: dict = {}
    right: dict = {}
    for index in range(4):
        pose: Float64[ndarray, "4 4"] = np.eye(4)
        pose[1, 3] = index * 10.0
        left[str(index)] = dict(index=index, agt_frame_id=index, timestamp=float(index), T_WorldFromCamera=pose.tolist(), is_synthesized=False)
        pose[0, 3] = 63.8
        right[str(index)] = dict(index=index, agt_frame_id=index, timestamp=float(index), T_WorldFromCamera=pose.tolist(), is_synthesized=False)
    lhs: HeadsetCalibration = from_dict(HeadsetCalibration, dict(**camera, T_WorldFromCamera_by_index=left))
    rhs: HeadsetCalibration = from_dict(HeadsetCalibration, dict(**camera, T_WorldFromCamera_by_index=right))
    rig: HeadsetRig = headset_rig(lhs, rhs, scene="synthetic")
    assert rig.cam0_T_cam1[0, 3] == pytest.approx(0.0638)
    assert rig.translation_std_m == pytest.approx(0.0)
    right["3"]["T_WorldFromCamera"][0][3] = 80.0
    rhs = from_dict(HeadsetCalibration, dict(**camera, T_WorldFromCamera_by_index=right))
    with pytest.raises(ValueError, match="synthetic.*not rigid"):
        headset_rig(lhs, rhs, scene="synthetic")


@pytest.mark.integration
@pytest.mark.parametrize(("subject", "scene", "count"), [("SPI102", "keyboard_toss-away_83ef", 30), ("LYA722", "birdhousetoy_shaking_8eca", 30)])
def test_real_scene_base(tmp_path: Path, subject: str, scene: str, count: int) -> None:
    source: Path = SHOW3D_RAW / "scenes" / subject / scene
    if not (source / "headset0.mp4").is_file():
        pytest.skip(f"SHOW3D scene videos absent: {source}")
    target: Path = tmp_path / "base.rrd"
    write_base_layer(
        SequenceIdentity("show3d", (subject, scene)),
        source,
        target,
        index=index_row(subject_id=subject, scene_id=scene),
        work_dir=tmp_path,
        frame_limit=count,
        hf_revision="test-sha",
    )
    videos: dict[str, int] = {}
    pose_rows: int = 0
    for chunk in read_chunks(target):
        batch: pa.RecordBatch = chunk.to_record_batch()
        if "VideoStream:sample" in batch.schema.names:
            videos[str(chunk.entity_path)] = videos.get(str(chunk.entity_path), 0) + chunk.num_rows
            assert set(chunk.timeline_names) == {"video_time", schema.FRAME_INDEX}
        if str(chunk.entity_path) == "/world/rig_01" and "Transform3D:translation" in batch.schema.names:
            pose_rows += chunk.num_rows
    assert len(videos) == 10
    assert set(videos.values()) == {count}
    assert pose_rows == count


@pytest.fixture
def tiny_scene(tmp_path: Path) -> Path:
    """Two B-frame videos and four source rows, including a zero-box row."""

    scene: Path = tmp_path / "scene"
    for directory in ("metadata", "camera_calibration", "blur_info"):
        (scene / directory).mkdir(parents=True)
    (scene / "metadata/recording_info.json").write_text(
        json.dumps(dict(start_frame_id=20, num_frames=4, fps=60, resolution={"headset0": [160, 256], "headset1": [160, 256]}))
    )
    frames: list[dict] = [dict(index=i, agt_frame_id=20 + i, timestamp=100.0 + i / 60, missing_cameras=[]) for i in range(4)]
    (scene / "metadata/frame_info.json").write_text(json.dumps(frames))
    for camera in range(2):
        poses: dict = {}
        for frame in frames:
            transform: Float64[ndarray, "4 4"] = np.eye(4)
            transform[0, 3] = camera * 63.8
            transform[1, 3] = frame["index"] * 10.0
            poses[str(frame["index"])] = dict(
                index=frame["index"],
                agt_frame_id=frame["agt_frame_id"],
                timestamp=frame["timestamp"],
                T_WorldFromCamera=transform.tolist(),
                is_synthesized=frame["index"] == 2,
                pose_source="mocap",
                is_pose_valid=True,
            )
        calibration: dict = dict(
            ImageSizeX=256,
            ImageSizeY=160,
            fx=30.0,
            fy=30.0,
            cx=32.0,
            cy=24.0,
            DistortionModel="PinholePlane",
            pose_contract_version=1,
            T_WorldFromCamera_by_index=poses,
        )
        (scene / f"camera_calibration/headset{camera}.json").write_text(json.dumps(calibration))
        (scene / f"blur_info/headset{camera}.mp4.json").write_text(
            json.dumps(dict(blur_boxes={"0": [[2.0, 4.0, 10.0, 16.0]], "1": []} if camera == 0 else {}))
        )
        with av.open(str(scene / f"headset{camera}.mp4"), "w") as container:
            stream: av.video.stream.VideoStream = container.add_stream(
                "libx264", rate=60, width=256, height=160, pix_fmt="yuv420p", options={"bf": "2", "x264-params": "b-adapt=0:keyint=60"}
            )
            for index in range(4):
                pixels: UInt8[ndarray, "160 256"] = (np.arange(160 * 256, dtype=np.uint16).reshape(160, 256) + index * 31).astype(np.uint8)
                video_frame: av.VideoFrame = av.VideoFrame.from_ndarray(pixels, format="gray")
                video_frame.pts = index
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        with av.open(str(scene / f"headset{camera}.mp4")) as container:
            assert any(packet.pts != packet.dts for packet in container.demux(video=0) if packet.pts is not None)
    return scene


@pytest.mark.integration
def test_synthetic_base_roundtrip(tmp_path: Path, tiny_scene: Path) -> None:
    target: Path = tmp_path / "synthetic.rrd"
    write_base_layer(
        SequenceIdentity("show3d", ("subject", "toy_pick-up_abcd")),
        tiny_scene,
        target,
        index=index_row(subject_id="subject", scene_id="toy_pick-up_abcd", split="test"),
        work_dir=tmp_path / "work",
        hf_revision="test-sha",
    )
    chunks: list[rr.experimental.Chunk] = read_chunks(target)
    videos: dict[str, int] = {}
    for chunk in chunks:
        batch: pa.RecordBatch = chunk.to_record_batch()
        if not chunk.is_static:
            assert set(chunk.timeline_names) == {schema.FRAME_INDEX, "video_time"}
        if "VideoStream:sample" in batch.schema.names:
            videos[str(chunk.entity_path)] = videos.get(str(chunk.entity_path), 0) + chunk.num_rows
    assert videos == {"/world/rig_01/cam_00/pinhole/video": 4, "/world/rig_01/cam_01/pinhole/video": 4}
    # §13: shipped face boxes sit at pinhole/boxes/face with the FACE class id and their reason as metadata.
    faces: list[rr.experimental.Chunk] = [chunk for chunk in chunks if str(chunk.entity_path) == schema.boxes_path(1, 0, "face")]
    assert not any("blur" in str(chunk.entity_path) for chunk in chunks)
    temporal_faces: list[rr.experimental.Chunk] = [chunk for chunk in faces if not chunk.is_static]
    assert len(temporal_faces) == 1 and temporal_faces[0].num_rows == 2
    face_batch: pa.RecordBatch = temporal_faces[0].to_record_batch()
    assert face_batch.column("Boxes2D:centers").to_pylist() == [[[6.0, 10.0]], []]
    assert face_batch.column("Boxes2D:class_ids").to_pylist() == [[103], []]
    static_face: pa.RecordBatch = next(chunk.to_record_batch() for chunk in faces if chunk.is_static)
    assert static_face.column("source").to_pylist() == [["blur_info"]]
    # The root AnnotationContext lives in base so boxes resolve their classes without any other layer.
    context: pa.RecordBatch = next(
        chunk.to_record_batch()
        for chunk in chunks
        if str(chunk.entity_path) == "/" and "AnnotationContext:context" in chunk.to_record_batch().schema.names
    )
    entries: list[dict] = context.column("AnnotationContext:context").to_pylist()[0][0]
    assert [entry["class_id"] for entry in entries] == [0, 100, 101, 102, 103]
    assert [entry["class_description"]["info"]["label"] for entry in entries[1:]] == ["left_hand", "right_hand", "full_body", "face"]
    frame_batch: pa.RecordBatch = next(chunk.to_record_batch() for chunk in chunks if str(chunk.entity_path) == "/frames")
    assert frame_batch.column("source_frame_id").to_pylist() == [[20], [21], [22], [23]]
    assert "string" in str(frame_batch.schema.field("missing_cameras").type)
    properties = recording_properties(read_back(target), "capture")
    assert properties["hf_revision"] == "test-sha"
    assert properties["schema"] == "dataforge:v1"
    assert properties["num_frames"] == 4
    assert properties["num_cameras"] == 2
    assert properties["num_synthesized_headset_poses"] == 1
    assert properties["source_start_frame_id"] == 20
    assert properties["source_start_time_s"] == 100.0
    assert recording_properties(read_back(target), "convert")["version"] == "1"
    assert recording_properties(read_back(target), "episode") == dict(subject_id="subject", split="test", object_alias="toy", action="pick-up")
    assert not list((tmp_path / "work").glob("*.mp4"))


def test_rig_calibration_is_typed_and_rejects_reflections() -> None:
    source: dict = dict(
        ImageSizeX=64, ImageSizeY=48, fx=30.0, fy=30.0, cx=32.0, cy=24.0, DistortionModel="PinholePlane", T_WorldFromCamera=np.eye(4).tolist()
    )
    camera: RigCalibration = from_dict(RigCalibration, source)
    assert camera.T_WorldFromCamera.dtype == np.float64
    source["T_WorldFromCamera"][0][0] = -1.0
    with pytest.raises((ValueError, SerdeError), match="proper rigid transform"):
        from_dict(RigCalibration, source)


def test_blur_boxes_normalize_inverted_corners() -> None:
    blur = BlurInfo({"255": [[454.64, 438.24, 452.30, 436.15], [1.0, 2.0, 3.0, 4.0]]})
    assert blur.blur_boxes["255"] == [[452.30, 436.15, 454.64, 438.24], [1.0, 2.0, 3.0, 4.0]]
    assert blur.num_normalized_boxes == 1


def test_camera_indices_survive_missing_rig0() -> None:
    assert [(camera.source_name, camera.rig, camera.cam) for camera in CAMERAS] == [
        ("headset0", 1, 0),
        ("headset1", 1, 1),
        ("rig0", 0, 0),
        ("rig1", 0, 1),
        ("rig2", 0, 2),
        ("rig3", 0, 3),
        ("rig4", 0, 4),
        ("rig5", 0, 5),
        ("rig6", 0, 6),
        ("rig7", 0, 7),
    ]
    present = [camera for camera in CAMERAS if camera.source_name != "rig0"]
    assert sum(camera.rig == 0 for camera in present) == 7
    assert [camera.cam for camera in present if camera.rig == 0] == [1, 2, 3, 4, 5, 6, 7]


def test_pane_contents_keep_only_this_cameras_image_space() -> None:
    """Each 2D pane excludes its own face boxes and shipped UV, plus every other pinhole subtree."""
    for camera in CAMERAS:
        contents: list[str] = pane_contents(camera)
        own: str = schema.pinhole_path(camera.rig, camera.cam)
        assert contents[0] == "+ /world/**"
        assert f"- {own}/boxes/face" in contents
        assert f"- {schema.coco133_uv_path(camera.rig, camera.cam)}" in contents
        assert f"- {own}/**" not in contents
        others: set[str] = {f"- {schema.pinhole_path(c.rig, c.cam)}/**" for c in CAMERAS if c is not camera}
        assert others <= set(contents)
        assert len(contents) == 3 + len(others)
        assert not any("**" in rule and not rule.endswith("/**") for rule in contents)


def test_world_contents_exclude_face_boxes_and_shipped_uv_by_explicit_path() -> None:
    """Rerun content filters ignore mid-path wildcards, so face boxes and shipped UV use exact paths."""
    contents: list[str] = world_contents()
    assert contents[0] == "+ /world/**"
    assert set(contents[1:]) == {
        f"- {path}"
        for camera in CAMERAS
        for path in (schema.boxes_path(camera.rig, camera.cam, "face"), schema.coco133_uv_path(camera.rig, camera.cam))
    }
    assert not any("*" in rule for rule in contents[1:])
