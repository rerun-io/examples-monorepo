"""Native MPS timing and missing-data contracts, with no GPU or raw assets."""

from pathlib import Path

import numpy as np

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


def test_all_hand_rows_and_zero_confidence_survive(tmp_path: Path) -> None:
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
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    times = np.arange(1_000_000, 102_000_000, 1_000_000, dtype=np.int64)
    poses = np.tile(np.eye(4), (len(times), 1, 1))
    poses[:, 0, 3] = 5.0
    frame_clock = np.array([0, 50_000_000], dtype=np.int64)
    result = read_hands(path, Trajectory(times, poses, np.ones(len(times))), frame_clock)
    assert result.times_ns.tolist() == [0, 33333000, 66667000, 100000000]
    assert result.frame_indices.tolist() == [0, 1, 1, 1]
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
