"""Driver-run local integration and source projection goldens; no downloads."""

import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from conftest import read_chunks

from dataforge import schema
from dataforge.datasets.assembly101 import Assembly101Config
from dataforge.datasets.assembly101_calibration import project
from dataforge.datasets.assembly101_layers import camera_sources, read_scene
from dataforge.datasets.assembly101_source import EGO_RIG, POSE_MEMBERS, pose_path, read_confidence, read_hand_rows, read_pixels

KEY: str = "nusar-2021_action_both_9011-c03f_9011_user_id_2021-02-01_160239"
DONOR: str = "nusar-2021_action_both_9012-c07c_9012_user_id_2021-02-01_164345"


@pytest.fixture
def assembly101_root() -> Path:
    root = Path(os.environ.get("ASSEMBLY101_TEST_ROOT", "/home/pablo/exoego-data/assembly101/raw"))
    for member in POSE_MEMBERS:
        path = pose_path(root, member, KEY)
        if not path.is_file():
            pytest.skip(f"Assembly101 asset absent: {path}")
    donor = root / "assemblyhands-toolkit/calib/nimble_json_calib" / f"{DONOR}.json"
    for path in (donor, pose_path(root, "camera_extrinsics_fixed", DONOR)):
        if not path.is_file():
            pytest.skip(f"Assembly101 calibration asset absent: {path}")
    folder = root / "videos/av1-720-new" / KEY
    if len(camera_sources(root, KEY)) != 12:
        pytest.skip(f"Assembly101 asset absent: expected all 12 MP4s under {folder}")
    return root


@pytest.mark.integration
def test_real_first_60_frames_all_layers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, assembly101_root: Path) -> None:
    annotations = Path(os.environ.get("ASSEMBLY101_TEST_ANNOTATIONS", "/mnt/nas/datasets/assembly101/official/annotations"))
    for split in ("train", "validation", "test"):
        path = annotations / "fine-grained-annotations" / f"{split}.csv"
        if not path.is_file():
            pytest.skip(f"Assembly101 action asset absent: {path}")
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path))
    dataset = Assembly101Config(root=assembly101_root, annotations_root=annotations, sequences=(KEY,), frame_limit=60).setup()
    identity, source = dataset.discover()[0]
    dataset.convert(identity, source, force=True)
    targets = dataset.targets(identity)
    for layer, path in targets.items():
        assert path.is_file(), layer
        chunks = read_chunks(path)
        temporal = [chunk for chunk in chunks if not chunk.is_static]
        assert temporal, layer
        for chunk in temporal:
            batch = chunk.to_record_batch()
            assert set(chunk.timeline_names) == {"video_time", "frame_index"}
            frames = np.array(batch.column("frame_index").to_pylist())
            assert np.all((frames >= 0) & (frames < 60))
            np.testing.assert_array_equal(batch.column("video_time").cast(pa.int64()).to_numpy(), np.rint(frames / 60 * 1e9).astype(np.int64))
    base = read_chunks(targets["base"])
    hand_pose = read_chunks(targets["hand_pose"])
    for camera in camera_sources(assembly101_root, KEY):
        video = [
            chunk
            for chunk in base
            if chunk.entity_path == schema.video_path(camera.rig, camera.cam) and "VideoStream:sample" in chunk.to_record_batch().schema.names
        ]
        assert sum(chunk.num_rows for chunk in video) == 60
        pixels = [chunk for chunk in hand_pose if chunk.entity_path == schema.coco133_uv_path(camera.rig, camera.cam) and not chunk.is_static]
        assert sum(chunk.num_rows for chunk in pixels) == 60
    assert any(chunk.entity_path == schema.rig_path(EGO_RIG) and not chunk.is_static for chunk in base)
    assert not any(chunk.entity_path.startswith("/__properties") for chunk in hand_pose)
    stamps = {layer: path.stat().st_mtime_ns for layer, path in targets.items()}
    dataset.convert(identity, source, force=False)
    assert stamps == {layer: path.stat().st_mtime_ns for layer, path in targets.items()}


@pytest.mark.golden
def test_all_12_lenses_match_shipped_pixels(assembly101_root: Path) -> None:
    scene = read_scene(assembly101_root, KEY, None)
    confidence = read_confidence(pose_path(assembly101_root, "hand_confidences", KEY))
    # Every 97th real key across the whole capture, matching the raw-facts measurement.
    xyz = {}
    for frames, positions, _ in read_hand_rows(
        pose_path(assembly101_root, "landmarks3D", KEY), confidence, dimensions=3, scale=0.001, frame_limit=None
    ):
        for frame, points in zip(frames, positions, strict=True):
            if frame % 97 == 0:
                xyz[int(frame)] = points.astype(np.float64)
    cameras = {camera.key: camera for camera in scene.cameras}
    errors = {key: [] for key in cameras}
    for key, (frames, pixels, _) in read_pixels(pose_path(assembly101_root, "landmarks2D", KEY), confidence, list(cameras), None):
        camera = cameras[key]
        stored = (1280, 720) if camera.rig != EGO_RIG else (636, 480)
        for frame, measured in zip(frames, pixels, strict=True):
            if int(frame) not in xyz:
                continue
            if camera.rig == EGO_RIG:
                transform = scene.world_T_rig[np.searchsorted(scene.frames, frame)] @ scene.rig_T_cam[key]
            else:
                transform = scene.fixed[key].copy()
                transform[:3, 3] *= 0.001
            inverse = np.linalg.inv(transform)
            points = xyz[int(frame)] @ inverse[:3, :3].T + inverse[:3, 3]
            predicted = project(points, scene.calibration.lenses[key], stored)
            valid = np.isfinite(measured).all(axis=1) & (points[:, 2] > 0)
            valid &= (measured[:, 0] >= 0) & (measured[:, 0] < stored[0]) & (measured[:, 1] >= 0) & (measured[:, 1] < stored[1])
            # Thumb-base midpoints are derived separately in 2D and 3D; projection is nonlinear.
            valid[[92, 113]] = False
            errors[key].extend(np.linalg.norm(predicted[valid] - measured[valid], axis=1).tolist())
    assert len(errors) == 12
    for key, values in errors.items():
        assert len(values) > 100, key
        assert float(np.median(values)) <= 0.1, (key, float(np.median(values)))
