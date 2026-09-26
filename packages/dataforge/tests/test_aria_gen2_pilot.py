"""Native MPS timing and missing-data contracts, with no GPU or raw assets."""

from pathlib import Path

import numpy as np
import pytest

from dataforge.datasets.aria_gen2_pilot_source import Trajectory


def test_pose_lookup_interpolates_without_clamping_or_bridging_gaps() -> None:
    poses = np.tile(np.eye(4), (4, 1, 1))
    poses[:, 0, 3] = [0.0, 2.0, 4.0, 8.0]
    track = Trajectory(np.array([1000, 1001000, 2001000, 5001000]), poses, np.ones(4))
    result = track.at(np.array([0, 1000, 501000, 1501000, 3001000, 5001000, 5001001]))
    np.testing.assert_allclose(result[[1, 2, 3, 5], 0, 3], [0.0, 1.0, 3.0, 8.0])
    assert np.isnan(result[[0, 4, 6]]).all()


def test_invalid_quaternion_rows_are_missing_and_never_held(tmp_path: Path) -> None:
    import csv

    from dataforge.datasets.aria_gen2_pilot_source import read_trajectory

    path = tmp_path / "closed_loop_trajectory.csv"
    quaternions = [(0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0)]
    with path.open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["graph_uid", "tracking_timestamp_us", "tx_world_device", "ty_world_device", "tz_world_device"]
                        + [f"q{axis}_world_device" for axis in "xyzw"] + ["quality_score"])
        for index, quaternion in enumerate(quaternions):
            writer.writerow(["g", index * 1000, 1.0, 2.0, 3.0, *quaternion, 1.0])
    track = read_trajectory(path)
    assert track.times_ns.tolist() == [0, 1000000, 2000000, 3000000]
    assert np.isnan(track.poses[[1, 2]]).all()  # zero and non-unit quaternions are never renormalised
    result = track.at(np.array([0, 500000, 1000000, 2500000, 3000000]))
    assert np.isnan(result[1:4]).all()
    expected = np.eye(4)
    expected[:3, 3] = [1.0, 2.0, 3.0]
    np.testing.assert_array_equal(result[[0, 4]], np.stack([expected, expected]))


@pytest.mark.parametrize("invalid_column,invalid_value", [(None, None), ("tracking_timestamp_us", 0.5), ("left_tracking_confidence", np.nan), ("right_tracking_confidence", np.inf)])
def test_all_hand_rows_and_zero_confidence_survive(tmp_path: Path, invalid_column: str | None, invalid_value: float | None) -> None:
    import csv

    from dataforge import hands
    from dataforge.datasets.aria_gen2_pilot_source import hand_columns, read_hands

    path = tmp_path / "hands.csv"
    # Every column gets a value naming its side, group and index, so a swapped slice cannot pass.
    rows = []
    for time_us in (0, 33333, 66667, 100000):
        row: dict[str, int | float] = {"tracking_timestamp_us": time_us}
        for side_index, side in enumerate(("left", "right")):
            groups = hand_columns(side)
            row[groups["confidence"][0]] = 0.0 if side == "left" else -1.0
            for joint in range(21):
                for axis_index, name in enumerate(groups["landmarks"][3 * joint : 3 * joint + 3]):
                    row[name] = [float(joint), float(side_index + 1), 0.5 * axis_index][axis_index]
            for name, value in zip(groups["wrist"], (0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0), strict=True):
                row[name] = value
            for name, value in zip(groups["normals"], (1.0, 0.0, 0.0, 0.0, 1.0, 0.0), strict=True):
                row[name] = value
        rows.append(row)
    if invalid_column is not None:
        assert invalid_value is not None
        rows[0][invalid_column] = invalid_value
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    times = np.arange(1_000_000, 102_000_000, 1_000_000, dtype=np.int64)
    poses = np.tile(np.eye(4), (len(times), 1, 1))
    poses[:, 0, 3] = 5.0
    from dataforge.datasets.hot3d_vrs import nearest_framesets

    frame_clock = np.array([0, 33_333_333, 66_666_667, 100_000_000], dtype=np.int64)
    rgb_clock = np.array([0, 100_000_000], dtype=np.int64)
    assert nearest_framesets(frame_clock, rgb_clock).tolist() == [0, 3]
    assert nearest_framesets(frame_clock, frame_clock).tolist() == [0, 1, 2, 3]
    assert nearest_framesets(np.array([0, 10]), np.array([5])).tolist() == [0]
    if invalid_column is not None:
        with pytest.raises(ValueError, match=str(path)):
            read_hands(path, Trajectory(times, poses, np.ones(len(times))), frame_clock)
        return
    result = read_hands(path, Trajectory(times, poses, np.ones(len(times))), frame_clock)
    assert result.times_ns.tolist() == [0, 33333000, 66667000, 100000000]
    assert result.frame_indices.tolist() == [0, 1, 2, 3]
    assert np.isnan(result.positions[0]).all()  # head is missing, never clamped
    np.testing.assert_array_equal(result.scores[1:], [[0.0, -1.0]] * 3)
    # Left joint j sits at x = 5 + j (rig x offset 5), y = 1 (left), z = 1.0; the right hand is absent.
    landmarks = np.full((2, 21, 3), np.nan, dtype=np.float32)
    landmarks[0] = [[5.0 + joint, 1.0, 1.0] for joint in range(21)]
    expected, _ = hands.coco133_from_hands(landmarks, np.array([0.0, -1.0], dtype=np.float32))
    np.testing.assert_allclose(result.positions[1], expected)
    assert (result.confidence[:, 91] == 0.0).all()  # present zero stays present
    assert np.isnan(result.positions[:, 112:]).all()
    np.testing.assert_allclose(result.normals[1:, 0], [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]] * 3)
    np.testing.assert_allclose(result.wrists[1:, 0, :3, 3], [[5.1, 0.2, 0.3]] * 3)
    assert np.isnan(result.wrists[:, 1]).all()


def test_slerp_uses_rotation_not_matrix_lerp() -> None:
    from scipy.spatial.transform import Rotation

    poses = np.tile(np.eye(4), (2, 1, 1))
    poses[1, :3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    track = Trajectory(np.array([0, 2000000]), poses, np.ones(2))
    actual = track.at(np.array([1000000]))[0, :3, :3]
    np.testing.assert_allclose(actual, Rotation.from_euler("z", 45, degrees=True).as_matrix(), atol=1e-12)


def test_verify_only_discovery_and_default_root(tmp_path: Path, monkeypatch) -> None:
    import hashlib
    import json

    from dataforge.datasets.aria_gen2_pilot import AriaGen2PilotConfig

    monkeypatch.delenv('DATAFORGE_RAW_ROOT', raising=False)
    assert AriaGen2PilotConfig().root == Path('/mnt/nas/datasets/aria-gen2-pilot')
    monkeypatch.setenv('DATAFORGE_RAW_ROOT', str(tmp_path))
    manifest = {'sequences': {}}
    for sequence in ('clean_0', 'cook_0'):
        source = tmp_path / sequence
        (source / 'mps/slam').mkdir(parents=True)
        (source / 'mps/hand_tracking').mkdir()
        (source / 'video.vrs').write_bytes(b'vrs')
        (source / 'mps/slam/closed_loop_trajectory.csv').touch()
        (source / 'mps/hand_tracking/hand_tracking_results.csv').touch()
        manifest['sequences'][sequence] = {'main_vrs': {'file_size_bytes': 3, 'sha1sum': hashlib.sha1(b'vrs').hexdigest()}}
    (tmp_path / 'AriaGen2PilotDataset_download_urls.json').write_text(json.dumps(manifest))
    dataset = AriaGen2PilotConfig().setup()
    before = {path: path.stat().st_mtime_ns for path in tmp_path.rglob('*')}
    dataset.download()
    assert [identity.recording_id for identity, _ in dataset.discover()] == ['aria_gen2_pilot__clean_0', 'aria_gen2_pilot__cook_0']
    assert before == {path: path.stat().st_mtime_ns for path in tmp_path.rglob('*')}
    (tmp_path / 'cook_0/video.vrs').write_bytes(b'bad')
    import pytest

    with pytest.raises(ValueError, match='SHA-1'):
        dataset.download()


def test_conversion_rejects_raw_output_before_opening_inputs(tmp_path: Path, monkeypatch) -> None:
    import pytest

    from dataforge.datasets.aria_gen2_pilot import AriaGen2PilotConfig
    from dataforge.identity import SequenceIdentity

    source = tmp_path / 'raw/clean_0'
    source.mkdir(parents=True)
    output = tmp_path / 'output-link'
    output.symlink_to(source, target_is_directory=True)
    monkeypatch.setenv('DATAFORGE_OUTPUT_ROOT', str(output))
    dataset = AriaGen2PilotConfig(root=source.parent).setup()
    with pytest.raises(ValueError, match='refusing output under the raw data'):
        dataset.convert(SequenceIdentity('aria_gen2_pilot', ('clean_0',)), source, force=True)
    assert list(source.iterdir()) == []


def test_trajectory_rejects_fractional_timestamps(tmp_path: Path) -> None:
    import pytest

    from dataforge.datasets.aria_gen2_pilot_source import read_trajectory

    trajectory_path = tmp_path / "trajectory.csv"
    trajectory_path.write_text(
        "tracking_timestamp_us,tx_world_device,ty_world_device,tz_world_device,"
        "qx_world_device,qy_world_device,qz_world_device,qw_world_device,quality_score\n"
        "0.5,0,0,0,0,0,0,1,1\n"
    )
    with pytest.raises(ValueError, match=str(trajectory_path)):
        read_trajectory(trajectory_path)


def test_motion_hides_at_first_imu_stamp_and_after_each_run(tmp_path: Path, monkeypatch) -> None:
    from unittest.mock import Mock

    import pyarrow as pa
    from conftest import read_chunks

    from dataforge import aria, writing
    from dataforge.datasets.aria_gen2_pilot_layers import write_motion
    from dataforge.datasets.aria_gen2_pilot_source import Scene
    from dataforge.logging_toolkit import ImuChannel

    scene = Mock(spec=Scene)
    scene.source = tmp_path
    scene.stop_ns = None
    scene.cameras = [Mock(times_ns=np.array([10_000_000])), Mock(times_ns=np.array([9_000_000]))]
    scene.frame_clock = np.array([9_000_000, 10_000_000, 11_000_000, 12_000_000, 13_000_000, 14_000_000])
    scene.trajectory = Trajectory(np.array([10_000_000, 11_000_000, 14_000_000]), np.tile(np.eye(4), (3, 1, 1)), np.ones(3))
    from projectaria_tools.core.calibration import ImuCalibration

    scene.calibration.get_imu_calib.return_value = Mock(spec=ImuCalibration)
    scene.calibration.get_imu_calib.return_value.get_transform_device_imu.return_value.to_matrix.return_value = np.eye(4)
    imu_times = np.array([8_000_000, 12_000_000], dtype=np.int64)
    channel = ImuChannel(imu_times, np.zeros((2, 3)))
    monkeypatch.setattr(aria, "open_vrs", lambda path: None)
    monkeypatch.setattr(aria, "read_imu", lambda provider, stream: (channel, channel))
    target = tmp_path / "motion.rrd"
    with writing.atomic_recording(target, recording_id="motion", send_properties=False) as recording:
        write_motion(recording, scene)
    batches = [chunk.to_record_batch() for chunk in read_chunks(target) if chunk.entity_path == "/world/rig_00"]
    poses = next(batch for batch in batches if "Transform3D:translation" in batch.schema.names)
    assert poses.column("video_time").cast(pa.int64()).to_pylist() == [8_000_000, 10_000_000, 11_000_000, 11_000_001, 14_000_000, 14_000_001]
    assert np.isnan(np.asarray(poses.column("Transform3D:mat3x3").to_pylist())[[0, 3, 5]]).all()
    assert poses.column("frame_index").to_pylist() == [0, 1, 2, 2, 5, 5]


def test_named_hand_fields_survive_reordered_csv(tmp_path: Path) -> None:
    import csv

    from scipy.spatial.transform import Rotation

    from dataforge import hands
    from dataforge.datasets.aria_gen2_pilot_source import read_hands

    row = {"tracking_timestamp_us": 0.0}
    expected_landmarks = np.arange(126, dtype=np.float64).reshape(2, 21, 3) / 100.0
    expected_normals = np.arange(12, dtype=np.float64).reshape(2, 2, 3) / 10.0
    expected_wrists = np.tile(np.eye(4), (2, 1, 1))
    for index, side in enumerate(("left", "right")):
        row[f"{side}_tracking_confidence"] = 0.25 + index * 0.5
        for joint in range(21):
            for axis_index, axis in enumerate("xyz"):
                row[f"t{axis}_{side}_landmark_{joint}_device"] = expected_landmarks[index, joint, axis_index]
        quaternion = np.array([1.0, 2.0, 3.0, 4.0]) + index
        quaternion /= np.linalg.norm(quaternion)
        expected_wrists[index, :3, :3] = Rotation.from_quat(quaternion).as_matrix()
        expected_wrists[index, :3, 3] = np.array([0.13, 0.27, 0.41]) + index
        for axis_index, axis in enumerate("xyz"):
            row[f"t{axis}_{side}_device_wrist"] = expected_wrists[index, axis_index, 3]
        for axis_index, axis in enumerate("xyzw"):
            row[f"q{axis}_{side}_device_wrist"] = quaternion[axis_index]
        for part_index, part in enumerate(("palm", "wrist")):
            for axis_index, axis in enumerate("xyz"):
                row[f"n{axis}_{side}_{part}_device"] = expected_normals[index, part_index, axis_index]
    path = tmp_path / "reordered.csv"
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(row, reverse=True))
        writer.writeheader()
        writer.writerow(row)
    track = Trajectory(np.array([0]), np.eye(4)[None], np.ones(1))
    result = read_hands(path, track, np.array([0]))
    expected_positions, expected_confidence = hands.coco133_from_hands(expected_landmarks.astype(np.float32), np.array([0.25, 0.75], dtype=np.float32))
    np.testing.assert_allclose(result.positions[0], expected_positions)
    np.testing.assert_array_equal(result.confidence[0], expected_confidence)
    np.testing.assert_allclose(result.wrists[0], expected_wrists)
    np.testing.assert_array_equal(result.normals[0], expected_normals)
    np.testing.assert_array_equal(result.scores[0], [0.25, 0.75])
