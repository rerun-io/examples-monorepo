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


def test_singular_pose_is_missing_and_never_held() -> None:
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[1, :3, :3] = 0.0
    track = Trajectory(np.array([0, 1000000, 2000000]), poses, np.ones(3))
    result = track.at(np.array([0, 500000, 1000000, 1500000, 2000000]))
    assert np.isnan(result[1:4]).all()
    np.testing.assert_array_equal(result[[0, 4]], np.tile(np.eye(4), (2, 1, 1)))


def test_all_hand_rows_and_zero_confidence_survive(tmp_path: Path) -> None:
    import csv

    from dataforge.datasets.aria_gen2_pilot_source import read_hands

    path = tmp_path / "hands.csv"
    rows = []
    for time_us in (0, 33333, 66667, 100000):
        row: dict[str, int | float] = {"tracking_timestamp_us": time_us}
        for side in ("left", "right"):
            row[f"{side}_tracking_confidence"] = 0.0 if side == "left" else -1.0
            for joint in range(21):
                for axis, value in zip("xyz", (1.0, 2.0, 3.0), strict=True):
                    row[f"t{axis}_{side}_landmark_{joint}_device"] = value
            for kind, axes in (("t", "xyz"), ("q", "xyzw")):
                for axis in axes:
                    row[f"{kind}{axis}_{side}_device_wrist"] = 1.0 if kind == "q" and axis == "w" else 0.0
            for part in ("palm", "wrist"):
                for axis in "xyz":
                    row[f"n{axis}_{side}_{part}_device"] = 1.0 if axis == "z" else 0.0
        rows.append(row)
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    times = np.arange(1_000_000, 102_000_000, 1_000_000, dtype=np.int64)
    poses = np.tile(np.eye(4), (len(times), 1, 1))
    poses[:, 0, 3] = 5.0
    result = read_hands(path, Trajectory(times, poses, np.ones(len(times))))
    assert result.times_ns.tolist() == [0, 33333000, 66667000, 100000000]
    assert np.isnan(result.positions[0]).all()  # head is missing, never clamped
    np.testing.assert_allclose(result.positions[1:, 91], [[6.0, 2.0, 3.0]] * 3)
    assert (result.confidence[:, 91] == 0.0).all()  # present zero stays present
    assert np.isnan(result.positions[:, 112:]).all()
    np.testing.assert_allclose(result.normals[1:, 0, 0], [[0.0, 0.0, 1.0]] * 3)
    np.testing.assert_allclose(result.wrists[1:, 0, 0, 3], [5.0] * 3)


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
    with pytest.raises(ValueError, match='refusing raw-root or NAS output'):
        dataset.convert(SequenceIdentity('aria_gen2_pilot', ('clean_0',)), source, force=True)
    assert list(source.iterdir()) == []
