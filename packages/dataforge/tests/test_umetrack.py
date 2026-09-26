"""UmeTrack raw-source contracts and published layer behavior."""

import json
import math
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from conftest import column_rows, read_back, recording_properties
from jaxtyping import Float64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from simplecv.umetrack_temp.generic_hand_model_numpy import SingleHandPose, landmarks_from_hand_pose, skin_landmarks, wrist_for_hand

from dataforge import paths, schema, writing
from dataforge.apis.compare_layers import ColumnKey, component_rows
from dataforge.datasets.umetrack import UmetrackConfig
from dataforge.datasets.umetrack_layers import hand_keypoints, write_geometry, write_hands, write_meshes, write_projections
from dataforge.datasets.umetrack_source import SequenceData, read_sequence
from dataforge.identity import SequenceIdentity


def test_discovery_and_selected_manifest_verification(tmp_path: Path) -> None:
    keys = ["synthetic/separate_hand/testing/user_19/recording_02", "real/hand_hand/training/user_03/recording_05"]
    root = tmp_path / "raw_data"
    for key in keys:
        path = root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".json").write_text("{}")
        path.with_suffix(".mp4").write_bytes(b"video")
    for domain in ("real", "synthetic"):
        entries = [
            f"https://github.com/facebookresearch/UmeTrack_data/raw/main/raw_data/{key}.{ext}"
            for key in keys
            if key.startswith(domain)
            for ext in ("json", "mp4")
        ]
        entries.append(f"https://github.com/facebookresearch/UmeTrack_data/raw/main/raw_data/{domain}/hand_hand/testing/user_00/recording_99.json")
        (tmp_path / f"raw_data_{domain}_manifest.txt").write_text("\n".join(entries))
    dataset = UmetrackConfig(root=root, sequences=tuple(keys)).setup()
    dataset.download()
    discovered = dataset.discover()
    assert [identity.recording_id for identity, _ in discovered] == ["umetrack__" + key.replace("/", "__") for key in sorted(keys)]
    with pytest.raises(FileNotFoundError):
        UmetrackConfig(root=root).setup().download()


@pytest.fixture
def tiny_umetrack(tmp_path: Path) -> Path:
    # This model is a small synthetic UmeTrack model, not a network/NAS fixture.

    transforms = np.tile(np.eye(4), (3, 4, 1, 1))
    transforms[:, :, 0, 3] = [100, 110, 120, 130]
    transforms[2] = 0
    wrists = np.tile(np.eye(4), (3, 2, 1, 1))
    wrists[1:, 0] = 0
    wrists[2, 1] = 0
    source = tmp_path / "recording.json"
    source.write_text(
        json.dumps(
            dict(
                cameras=[
                    dict(
                        ImageSizeX=16,
                        ImageSizeY=16,
                        fx=20.0 + k,
                        fy=20.0,
                        cx=8.0,
                        cy=8.0,
                        DistortionModel="FishEye62",
                        k1=0.1,
                        k2=0.0,
                        k3=0.0,
                        k4=0.0,
                        p1=0.0,
                        p2=0.0,
                        p3=0.3,
                        p4=0.4,
                    )
                    for k in range(4)
                ],
                camera_angles=[0.0, 90.0, 90.0, 180.0],
                hand_model=umetrack_model_document(),
                joint_angles=np.zeros((3, 2, 22)).tolist(),
                wrist_transforms=wrists.tolist(),
                hand_confidences=[[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
                camera_to_world_transforms=transforms.tolist(),
            )
        )
    )
    with av.open(str(source.with_suffix(".mp4")), "w") as container:
        stream = container.add_stream("libx264", rate=29, width=64, height=16, pix_fmt="yuv420p")
        for index in range(3):
            frame = av.VideoFrame.from_ndarray(np.full((16, 64), index * 30, dtype=np.uint8), format="gray")
            frame.pts = index
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return source


def test_container_clock_camera_order_and_dropouts(tiny_umetrack: Path) -> None:
    scene = read_sequence(tiny_umetrack)
    assert scene.fps == 29
    assert scene.times_ns.tolist() == [0, 34482759, 68965517]
    assert [camera.fx for camera in scene.labels.cameras] == [20.0, 21.0, 22.0, 23.0]
    assert [scene.crop(k) for k in range(4)] == [(16, 16, 0, 0), (16, 16, 16, 0), (16, 16, 32, 0), (16, 16, 48, 0)]
    np.testing.assert_allclose(scene.rig_T_cam[:, 0, 3], [0, 0.01, 0.02, 0.03], atol=1e-8)
    assert scene.tracked.tolist() == [True, True, False]
    assert np.isnan(scene.world_T_rig[2]).all()
    assert scene.labels.cameras[0].p3 == 0.3


def umetrack_model_document() -> dict:
    """A three-vertex model with all points bound to the wrist frame."""
    return dict(
        joint_rotation_axes=[[1.0, 0.0, 0.0]] * 22,
        joint_rest_positions=[[0.0, 0.0, 0.0]] * 22,
        joint_frame_index=[0] * 22,
        joint_parent=[-1] * 22,
        joint_first_child=[-1] * 22,
        joint_next_sibling=[-1] * 22,
        landmark_rest_positions=[[float(i), 0.0, 0.0] for i in range(21)],
        landmark_rest_bone_weights=[[1.0, 0.0, 0.0]] * 21,
        landmark_rest_bone_indices=[[0, 0, 0]] * 21,
        hand_scale=1.0,
        mesh_vertices=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
        mesh_triangles=[[0, 1, 2]],
        dense_bone_weights=[[1.0] + [0.0] * 16] * 3,
        joint_limits=[[-1.0, 1.0]] * 22,
    )


def test_hand_layers_clear_missing_rows_and_preserve_parameters(tiny_umetrack: Path, tmp_path: Path) -> None:
    scene = read_sequence(tiny_umetrack)
    with writing.atomic_recording(tmp_path / "hand_pose.rrd", recording_id="test", send_properties=False) as recording:
        write_hands(recording, scene, hand_keypoints(scene))
    with writing.atomic_recording(tmp_path / "hand_mesh.rrd", recording_id="test", send_properties=False) as recording:
        write_meshes(recording, scene)
    store = read_back(tmp_path / "hand_pose.rrd")
    xyz = column_rows(store, "/world/gt/coco133_xyz:Points3D:positions").column(1).to_pylist()
    assert np.isfinite(xyz[0][91:112]).all()
    np.testing.assert_allclose(xyz[0][91], [0.005, 0.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(xyz[0][112], [-0.005, 0.0, 0.0], atol=1e-8)
    profile = store.reader(index=None, contents="/world/gt/hands/profile").to_arrow_table()
    assert profile["/world/gt/hands/profile:TextDocument:text"][0].as_py() == [scene.profile_text]
    assert scene.profile_text in tiny_umetrack.read_text()
    assert "coco133_uv" not in " ".join(store.reader(index="video_time").to_arrow_table().column_names)
    assert np.isnan(xyz[1][91:112]).all()
    assert np.isnan(xyz[2]).all()
    confidence = column_rows(store, "/world/gt/hands/left/confidence:Scalars:scalars").column(1).to_pylist()
    assert confidence == [[1.0], [0.0], [0.0]]
    angles = column_rows(store, "/world/gt/hands/left/joint_angles:joint_angles")
    assert angles.num_rows == 1
    vertices = column_rows(read_back(tmp_path / "hand_mesh.rrd"), "/world/gt/hands/left/mesh:Mesh3D:vertex_positions").column(1).to_pylist()
    assert len(vertices[0]) == 3
    assert vertices[1:] == [[], []]


def test_geometry_roundtrip_and_raw_write_guard(tiny_umetrack: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    identity = SequenceIdentity("umetrack", ("real", "hand_hand", "training", "user_03", "recording_05"))
    scene = read_sequence(tiny_umetrack)
    target = tmp_path / "geometry.rrd"
    with writing.atomic_recording(target, recording_id=identity.recording_id) as recording:
        write_geometry(recording, scene, identity)
    store = read_back(target)
    translations = column_rows(store, "/world/rig_00:Transform3D:translation").column(1).to_pylist()
    np.testing.assert_allclose(translations[0], [[0.1, 0.0, 0.0]])
    assert np.isnan(translations[2]).all()
    capture = recording_properties(store, "capture")
    assert capture["clock_source"] == "mp4_container_pts"
    assert capture["world_up_axis"] == "+y"
    for camera in range(4):
        table = store.reader(index=None, contents=f"/world/rig_00/cam_{camera:02}/**").to_arrow_table()
        prefix = f"/world/rig_00/cam_{camera:02}"
        assert table[f"{prefix}:kind"][0].as_py() == ["grayscale"]
        assert table[f"{prefix}/pinhole:p3"][0].as_py() == [0.3]
        assert table[f"{prefix}/pinhole:p4"][0].as_py() == [0.4]
    assert recording_properties(store, "episode")["domain"] == "real"
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "forbidden"))
    with pytest.raises(ValueError, match="beneath raw root"):
        UmetrackConfig(root=tmp_path).setup().convert(identity, tiny_umetrack, force=True)


ITERATION_KEYS: tuple[str, str] = (
    "real/hand_hand/training/user_03/recording_05",
    "synthetic/separate_hand/testing/user_19/recording_02",
)


@pytest.mark.integration
@pytest.mark.parametrize("key", ITERATION_KEYS)
def test_real_and_synthetic_all_layers(key: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    root = paths.raw_root() / "umetrack-data/raw_data"
    assets: list[Path] = [
        *((root / key).with_suffix("." + extension) for extension in ("json", "mp4")),
        *(root.parent / f"raw_data_{domain}_manifest.txt" for domain in ("real", "synthetic")),
    ]
    for asset in assets:
        if not asset.is_file():
            pytest.skip(f"UmeTrack iteration asset absent: {asset}")
    request.getfixturevalue("nvenc_ffmpeg")
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path))
    dataset = UmetrackConfig(root=root, sequences=(key,), frame_limit=60).setup()
    dataset.download()
    identity, source = dataset.discover()[0]
    scene = read_sequence(source, 60)
    dataset.convert(identity, source, force=True)
    targets = dataset.targets(identity)
    assert all(target.is_file() for target in targets.values())
    projected = read_back(targets["projections"])
    assert recording_properties(projected, "projections") == {"derived_from": "coco133_xyz", "camera_model": "FishEye62"}
    projection_columns = component_rows(targets["projections"])
    assert {entity for entity, component, _ in projection_columns if component == "Points2D:positions"} == {
        schema.coco133_uv_projected_path(0, camera) for camera in range(4)
    }
    present = False
    for camera in range(4):
        table = projection_columns[(schema.coco133_uv_projected_path(0, camera), "Points2D:positions", ("frame_index", schema.TIMELINE))]
        assert table[schema.TIMELINE].cast("int64").to_pylist() == scene.times_ns.tolist()
        assert table["frame_index"].to_pylist() == scene.frame_indices.tolist()
        pixels = np.asarray(table["Points2D:positions"].to_pylist())
        valid = np.isfinite(pixels).all(axis=-1)
        present |= bool(valid.any())
        assert np.all(pixels[valid] >= 0)
        assert np.all(pixels[valid] < [scene.labels.cameras[camera].ImageSizeX, scene.labels.cameras[camera].ImageSizeY])
    if key.startswith("real"):
        assert present
    base = read_back(targets["base"])
    for camera in range(4):
        samples = column_rows(base, f"{schema.video_path(0, camera)}:VideoStream:sample")
        assert samples.num_rows == 60
        assert samples.column(0).cast("int64").to_pylist() == scene.times_ns.tolist()
    assert recording_properties(base, "capture")["clock_source"] == "mp4_container_pts"
    hand = read_back(targets["hand_pose"])
    xyz = column_rows(hand, "/world/gt/coco133_xyz:Points3D:positions")
    assert xyz.num_rows == 60
    if key.startswith("synthetic"):
        assert np.isnan(np.asarray(xyz.column(1).to_pylist())[:, 91:112]).all()
    # ChunkStore's table reader omits components whose cells are all empty.
    mesh: dict[ColumnKey, pa.Table] = component_rows(targets["hand_mesh"])
    for hand_index, side in enumerate(("left", "right")):
        vertices: pa.Table = mesh[(f"/world/gt/hands/{side}/mesh", "Mesh3D:vertex_positions", ("frame_index", schema.TIMELINE))]
        assert vertices.num_rows == 60
        assert vertices[schema.TIMELINE].cast("int64").to_pylist() == scene.times_ns.tolist()
        assert vertices["frame_index"].to_pylist() == scene.frame_indices.tolist()
        assert pc.call_function("list_value_length", [vertices["Mesh3D:vertex_positions"]]).to_pylist() == [
            len(scene.labels.hand_model.mesh_vertices) if confidence > 0 else 0
            for confidence in scene.labels.hand_confidences[:, hand_index]
        ]


@pytest.mark.golden
def test_reference_landmarks_rig_and_frame_430_dropout(tmp_path: Path) -> None:
    key = ITERATION_KEYS[0]
    source = paths.raw_root() / "umetrack-data/raw_data" / f"{key}.json"
    reference = paths.raw_root() / "exoego-forge-catalog-rig/umetrack" / f"{key}.rrd"
    for asset in (source, source.with_suffix(".mp4"), reference):
        if not asset.is_file():
            pytest.skip(f"UmeTrack golden asset absent: {asset}")
    scene = read_sequence(source)
    identity = SequenceIdentity("umetrack", tuple(key.split("/")))
    ours = tmp_path / "ours.rrd"
    with writing.atomic_recording(ours, recording_id=identity.recording_id) as recording:
        write_geometry(recording, scene, identity)
        write_hands(recording, scene, hand_keypoints(scene))
    store = read_back(ours)
    ref = read_back(reference)
    points = column_rows(store, "/world/gt/coco133_xyz:Points3D:positions")
    ref_points = column_rows(ref, "/world/gt/coco133_xyz:Points3D:positions")
    ours_times = points.column(0).cast("int64").to_pylist()
    ref_times = ref_points.column(0).cast("int64").to_pylist()
    assert ours_times == ref_times == scene.times_ns.tolist()
    xyz = np.asarray(points.column(1).to_pylist())
    ref_xyz = np.asarray(ref_points.column(1).to_pylist())
    for hand, slots in enumerate((slice(91, 112), slice(112, 133))):
        valid = scene.labels.hand_confidences[:, hand] > 0
        np.testing.assert_allclose(xyz[valid, slots], ref_xyz[valid, slots], atol=1e-4, rtol=0)
        index = int(np.flatnonzero(valid)[0])
        wrist = scene.labels.wrist_transforms[index, hand]
        angles = scene.labels.joint_angles[index, hand]
        expected = landmarks_from_hand_pose(scene.labels.hand_model, SingleHandPose(angles, wrist, 1.0), hand)
        actual = skin_landmarks(scene.labels.hand_model, angles[None], wrist_for_hand(wrist[None], hand))[0]
        np.testing.assert_allclose(actual, expected, atol=0.1, rtol=0)
    translation = column_rows(store, "/world/rig_00:Transform3D:translation")
    ref_translation = column_rows(ref, "/world/rig_00:Transform3D:translation")
    assert translation.column(0).cast("int64").to_pylist() == ref_translation.column(0).cast("int64").to_pylist()
    ours_rig = np.asarray(translation.column(1).to_pylist())
    reference_rig = np.asarray(ref_translation.column(1).to_pylist())
    np.testing.assert_allclose(ours_rig[scene.tracked], reference_rig[scene.tracked], atol=1e-4, rtol=0)
    assert not scene.tracked[430]
    assert np.isnan(ours_rig[430]).all()
    assert np.isnan(xyz[430]).all()


@pytest.mark.parametrize("synthetic", [False, True])
def test_typed_camera_distortion_roundtrip(tiny_umetrack: Path, tmp_path: Path, synthetic: bool) -> None:
    document = json.loads(tiny_umetrack.read_text())
    for camera in document["cameras"]:
        camera.update(k1=0.1, k2=0.2, k3=0.3, k4=0.4, p1=0.7, p2=0.8)
        camera.pop("p3")
        camera.pop("p4")
        camera.update({"p3" if synthetic else "k5": 0.5, "p4" if synthetic else "k6": 0.6})
    tiny_umetrack.write_text(json.dumps(document))
    scene: SequenceData = read_sequence(tiny_umetrack)
    target: Path = tmp_path / "calibration.rrd"
    identity: SequenceIdentity = SequenceIdentity("umetrack", ("synthetic" if synthetic else "real", "hand_hand", "training", "user_03", "recording_05"))
    with writing.atomic_recording(target, recording_id=identity.recording_id) as recording:
        write_geometry(recording, scene, identity)
    for index in range(4):
        prefix: str = schema.pinhole_path(0, index)
        table: pa.Table = read_back(target).reader(index=None, contents=prefix).to_arrow_table()
        assert table[f"{prefix}:simplecv.components.DistortionModel"][0].as_py() == ["kannala_brandt"]
        np.testing.assert_allclose(
            table[f"{prefix}:simplecv.components.DistortionCoefficients"][0].as_py(),
            [[0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.7, 0.8] if synthetic else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]],
        )
        radial_names: tuple[str, str] = ("p3", "p4") if synthetic else ("k5", "k6")
        absent_names: tuple[str, str] = ("k5", "k6") if synthetic else ("p3", "p4")
        for name, expected in zip(radial_names, (0.5, 0.6), strict=True):
            assert table[f"{prefix}:{name}"][0].as_py() == [expected]
        assert all(f"{prefix}:{name}" not in table.column_names for name in absent_names)



@pytest.mark.parametrize("pair", [{}, {"k5": 0.5}, {"p3": 0.5}, {"k5": 0.5, "p4": 0.6}, {"k5": 0.5, "k6": 0.6, "p3": 0.5, "p4": 0.6}])
def test_camera_rejects_missing_or_ambiguous_radial_pair(tiny_umetrack: Path, pair: dict[str, float]) -> None:
    document = json.loads(tiny_umetrack.read_text())
    camera = document["cameras"][0]
    camera.pop("p3")
    camera.pop("p4")
    camera.update(pair)
    tiny_umetrack.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="radial pair"):
        read_sequence(tiny_umetrack)


def test_preview_slices_labels_but_measures_full_record(tiny_umetrack: Path, tmp_path: Path) -> None:
    document = json.loads(tiny_umetrack.read_text())
    document["camera_angles"] = [0.0] * 4
    transforms: Float64[ndarray, "3 4 4 4"] = np.asarray(document["camera_to_world_transforms"], dtype=np.float64)
    transforms[0, :, :3, :3] = Rotation.from_euler("z", [[150], [150], [210], [210]], degrees=True).as_matrix()
    transforms[1, :, :3, :3] = Rotation.from_euler("z", [[110], [110], [250], [250]], degrees=True).as_matrix()
    document["camera_to_world_transforms"] = transforms.tolist()
    tiny_umetrack.write_text(json.dumps(document))
    scene: SequenceData = read_sequence(tiny_umetrack, frame_limit=1)
    assert scene.source_num_frames == 3
    assert scene.count == 1
    for rows in (scene.labels.joint_angles, scene.labels.wrist_transforms, scene.labels.hand_confidences, scene.labels.camera_to_world_transforms):
        assert len(rows) == 1
    np.testing.assert_allclose(scene.headset_up, [0, 1, 0], atol=1e-12)
    assert scene.headset_up_spread_deg == pytest.approx(50.0)
    identity: SequenceIdentity = SequenceIdentity("umetrack", ("real", "hand_hand", "training", "user_03", "recording_05"))
    target: Path = tmp_path / "preview.rrd"
    with writing.atomic_recording(target, recording_id=identity.recording_id) as recording:
        write_geometry(recording, scene, identity)
        write_hands(recording, scene, hand_keypoints(scene))
        write_meshes(recording, scene)
    capture = recording_properties(read_back(target), "capture")
    assert capture["source_num_frames"] == 3
    assert capture["num_frames"] == 1
    assert capture["headset_up_spread_deg"] == pytest.approx(50.0)
    assert "headset_up_y" not in capture


def test_projections_use_world_hands_camera_pose_and_both_clocks(tiny_umetrack: Path, tmp_path: Path) -> None:
    document = json.loads(tiny_umetrack.read_text())
    # Head moves 10 cm on frame 1; the world hand stays put. A rig dropout follows.
    for frame in range(2):
        for camera in range(4):
            document["camera_to_world_transforms"][frame][camera][0][3] += frame * 100
        for hand in range(2):
            if document["hand_confidences"][frame][hand] > 0:
                document["wrist_transforms"][frame][hand][2][3] = 1000
    tiny_umetrack.write_text(json.dumps(document))
    scene = read_sequence(tiny_umetrack)
    target = tmp_path / "projections.rrd"
    with writing.atomic_recording(target, recording_id="test", send_properties=False) as recording:
        write_projections(recording, scene, hand_keypoints(scene))
    store = read_back(target)
    assert recording_properties(store, "projections") == {"derived_from": "coco133_xyz", "camera_model": "FishEye62"}
    hand_target = tmp_path / "hands.rrd"
    with writing.atomic_recording(hand_target, recording_id="test", send_properties=False) as recording:
        write_hands(recording, scene, hand_keypoints(scene))
    xyz = np.array(column_rows(read_back(hand_target), f"{schema.coco133_xyz_path()}:Points3D:positions").column(1).to_pylist())
    columns = component_rows(target)
    assert {entity for entity, component, _ in columns if component == "Points2D:positions"} == {
        schema.coco133_uv_projected_path(0, camera) for camera in range(4)
    }
    for camera in range(4):
        path = schema.coco133_uv_projected_path(0, camera)
        table = columns[(path, "Points2D:positions", ("frame_index", schema.TIMELINE))]
        assert table[schema.TIMELINE].cast("int64").to_pylist() == scene.times_ns.tolist()
        assert table["frame_index"].to_pylist() == scene.frame_indices.tolist()
        pixels = np.asarray(table["Points2D:positions"].to_pylist())
        assert np.isnan(pixels[2]).all()
        # Independent scalar equidistant + radial polynomial, for the right wrist; the fixture's p3/p4 are not projected.
        for frame in range(2):
            x = xyz[frame, 112, 0] - (0.1 + frame * 0.1 + camera * 0.01)
            theta = math.atan2(abs(x), xyz[frame, 112, 2])
            theta_d = theta * (1 + 0.1 * theta**2)
            np.testing.assert_allclose(pixels[frame, 112], [8 + (20 + camera) * math.copysign(theta_d, x), 8], atol=1e-6, rtol=0)


def test_no_hand_gt_writes_empty_projections(tiny_umetrack: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    document = json.loads(tiny_umetrack.read_text())
    document["hand_confidences"] = np.zeros((3, 2)).tolist()
    document["wrist_transforms"] = np.zeros((3, 2, 4, 4)).tolist()
    tiny_umetrack.write_text(json.dumps(document))
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "output"))
    dataset = UmetrackConfig(root=tmp_path / "raw").setup()
    identity = SequenceIdentity("umetrack", ("real", "hand_hand", "training", "user_03", "recording_05"))
    targets = dataset.targets(identity)
    # Existing layers avoid video encoding; only the missing projection is pending.
    for layer, target in targets.items():
        if layer != "projections":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"existing")
    dataset.convert(identity, tiny_umetrack, force=False)
    assert targets["projections"].is_file()
    store = read_back(targets["projections"])
    for camera in range(4):
        path = schema.coco133_uv_projected_path(0, camera)
        pixels = column_rows(store, f"{path}:Points2D:positions").column(1).to_pylist()
        confidence = column_rows(store, f"{path}:simplecv.KeypointConfidence2D:confidences").column(1).to_pylist()
        assert np.asarray(pixels).shape == (3, 133, 2)
        assert np.isnan(pixels).all()
        np.testing.assert_array_equal(confidence, np.zeros((3, 133)))
