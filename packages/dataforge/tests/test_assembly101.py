"""Assembly101 public source and dataset contracts, using synthetic inputs."""

import json
from pathlib import Path

import numpy as np

from dataforge.datasets.assembly101_source import read_hand_rows


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
    assert scores[0, 91] == 0.0
    assert np.isnan(positions[:, 0]).all()


def test_calibration_scaling_and_projection() -> None:
    from dataforge.datasets.assembly101_calibration import Lens, project

    lens = Lens("OpenCV", 1920, 1080, "C10095_rgb", 900.0, 900.0, 960.0, 540.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(lens.matrix((1280, 720)), [[600, 0, 640], [0, 600, 360], [0, 0, 1]])
    np.testing.assert_allclose(project(np.array([[1.0, 0.0, 2.0]]), lens, (1280, 720)), [[940, 360]])
    # KB6 with only tangential terms: atan(1)=pi/4, then the swapped p1/p2 convention.
    fish = Lens("OVFishEye62", 636, 480, "84346135_mono10bit", 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2)
    np.testing.assert_allclose(project(np.array([[1.0, 0.0, 1.0]]), fish, (636, 480)), [[0.9704532459, 0.1233700550]])


def test_actions_union_overlap_and_30_hz_boundaries(tmp_path: Path) -> None:
    from dataforge.datasets.assembly101_actions import action_rows, read_actions

    fine = tmp_path / "fine-grained-annotations"
    fine.mkdir()
    (fine / "train.csv").write_text("video,start_frame,end_frame,action_cls\ns/cam1.mp4,30,90,hold\ns/cam2.mp4,30,90,hold\ns/cam1.mp4,60,120,turn\n")
    segments = read_actions(tmp_path, {"s"})["s"]["fine"]
    frames, texts = action_rows(segments)
    assert frames.tolist() == [60, 120, 180, 240]
    assert texts == ["hold", "hold\nturn", "turn", ""]


def test_discovery_sorted_serials_and_video_only(tmp_path: Path) -> None:
    from dataforge.datasets.assembly101 import Assembly101Config
    from dataforge.datasets.assembly101_layers import camera_sources

    for sequence in ("z", "a"):
        folder = tmp_path / "videos/av1-720-new" / sequence
        folder.mkdir(parents=True)
        for name in ("HMC_21179183_mono10bit_low", "C10404_rgb_low", "C10095_rgb_low", "HMC_21110305_mono10bit_low"):
            (folder / f"{name}.mp4").touch()
    dataset = Assembly101Config(root=tmp_path, annotations_root=tmp_path / "annotations").setup()
    assert [identity.recording_id for identity, _ in dataset.discover()] == ["assembly101__a", "assembly101__z"]
    cameras = camera_sources(tmp_path, "a")
    assert [cam.serial for cam in cameras] == ["C10095", "C10404", "21110305", "21179183"]
    assert [(cam.rig, cam.cam) for cam in cameras] == [(0, 0), (7, 0), (8, 0), (8, 1)]
    assert dataset.targets(dataset.discover()[0][0]).keys() == {"base", "hand_pose", "actions"}


def test_calibration_sessions_do_not_borrow_across_days(tmp_path: Path) -> None:
    from dataforge.datasets.assembly101_calibration import resolve_calibration
    from dataforge.datasets.assembly101_source import pose_path, read_transforms

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
    assert calibrated.sources["C10095:rgb"] == "nimble:donor"
    assert calibrated.lenses["C10095:rgb"].fy == 901.0
    transforms["C10095:rgb"][0, 3] += 1.0
    uncovered = resolve_calibration(tmp_path, transforms)
    assert "C10095:rgb" not in uncovered.lenses
    assert "84346135:mono10bit" in uncovered.lenses


def test_shipped_pixels_scaled_once_all_views(tmp_path: Path) -> None:
    from dataforge.datasets.assembly101_source import read_pixels

    path = tmp_path / "pixels.json"
    path.write_text(json.dumps({"12": {"C10095:rgb": {"0": [[900, 600]] * 21}, "84346135:mono10bit": {"1": [[300, 200]] * 21}}}))
    rows = dict(read_pixels(path, {12: np.array([0.3, 0.8], dtype=np.float32)}, ["C10095:rgb", "84346135:mono10bit"], None))
    assert rows["C10095:rgb"][0].tolist() == [12]
    np.testing.assert_allclose(rows["C10095:rgb"][1][0, 91], [600, 400])
    np.testing.assert_allclose(rows["84346135:mono10bit"][1][0, 112], [300, 200])


def test_transport_rejects_nas_before_network() -> None:
    import pytest

    from dataforge.datasets.assembly101_download import fetch_pose_members

    with pytest.raises(ValueError, match="never the NAS"):
        fetch_pose_members(Path("/mnt/nas/datasets/assembly101"), ("sequence",))


def test_video_only_conversion_remuxes_own_lengths_and_actions(tmp_path: Path, monkeypatch) -> None:
    import shutil

    import pyarrow as pa
    from conftest import read_chunks

    from dataforge import schema
    from dataforge.datasets.assembly101 import Assembly101Config
    from dataforge.datasets.assembly101_source import EXO_SERIALS

    root = tmp_path / "raw"
    folder = root / "videos/av1-720-new/s"
    folder.mkdir(parents=True)
    fixture = Path(__file__).parent / "fixtures/av1_48f_192x160.mp4"
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
    dataset.download()
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
    stamps = {layer: path.stat().st_mtime_ns for layer, path in targets.items() if path.exists()}
    dataset.convert(identity, source, force=False)
    assert stamps == {layer: path.stat().st_mtime_ns for layer, path in targets.items() if path.exists()}


def test_manifest_prevents_missing_poses_becoming_video_only(tmp_path: Path) -> None:
    import pytest

    from dataforge.datasets.assembly101 import Assembly101Config
    from dataforge.datasets.assembly101_source import EXO_SERIALS

    folder = tmp_path / "videos/av1-720-new/s"
    folder.mkdir(parents=True)
    for name in [*(f"{serial}_rgb_low" for serial in EXO_SERIALS), *(f"HMC_{serial}_mono10bit_low" for serial in (1, 2, 3, 4))]:
        (folder / f"{name}.mp4").touch()
    manifest = tmp_path / "manifests/sequences.csv"
    manifest.parent.mkdir()
    manifest.write_text("sequence_name,video_only\ns,False\n")
    with pytest.raises(FileNotFoundError, match="camera_extrinsics_fixed/s.json"):
        Assembly101Config(root=tmp_path).setup().download()


def test_moving_headset_uses_one_track_and_static_camera_offsets(tmp_path: Path) -> None:
    from dataforge.datasets.assembly101_layers import read_scene
    from dataforge.datasets.assembly101_source import pose_path

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
    scene = read_scene(tmp_path, "s", None)
    assert scene.frames.tolist() == [0, 2]
    assert scene.timestamp_t0 == 10.0
    np.testing.assert_allclose(scene.world_T_rig[:, :3, 3], [[0.1, 0.2, 0.3], [0.1, 0.3, 0.3]])
    np.testing.assert_allclose(scene.rig_T_cam["20:mono10bit"][:3, 3], [0.04, 0, 0])
    assert not scene.calibration.lenses
