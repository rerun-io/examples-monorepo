"""Raw-data, NVENC and reference checks; absent assets skip by name."""

import os
from collections.abc import Iterator
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pytest
import rerun.chunk as rrc
from conftest import assert_readers_match_projectaria, raw_asset, read_chunks
from simplecv.sensors.camera import fisheye624

from dataforge import schema
from dataforge.aria import project_to_calibration
from dataforge.clocks import nearest_framesets
from dataforge.datasets.aria_gen2_pilot import AriaGen2PilotConfig
from dataforge.datasets.aria_gen2_pilot_source import Camera, Scene, read_scene
from dataforge.video_encoding import AV1_CQ, AV1_GOP, FrameSource, encode_frames_to_mp4
from dataforge.vrs import VrsFile, census_images
from dataforge.vrs_hevc import VrsHevcReader


@pytest.fixture(scope="module")
def pilot_source() -> Path:
    root = AriaGen2PilotConfig().root
    source = root / "clean_0"
    for relative in ("video.vrs", "mps/slam/closed_loop_trajectory.csv", "mps/hand_tracking/hand_tracking_results.csv"):
        raw_asset("Aria Gen2 asset", source / relative)
    return source


@pytest.mark.golden
@pytest.mark.parametrize("sequence", ["clean_0", "eat_1"])
def test_vrs_readers_match_projectaria(sequence: str) -> None:
    root = AriaGen2PilotConfig().root
    assert_readers_match_projectaria(VrsFile(raw_asset("Aria Gen2 asset", root / sequence / "video.vrs")), f"gen2-pilot-{sequence}")


@pytest.fixture(scope="module")
def pilot_scene(pilot_source: Path) -> Scene:
    return read_scene(pilot_source, 60)


def preview_access_units(scene: Scene, camera: Camera) -> Iterator[bytes]:
    return census_images(
        VrsHevcReader(scene.vrs, camera.stream_id).images(), camera.times_ns, camera.source_count, preview=True, where=camera.stream_id
    )


@pytest.mark.integration
def test_native_raw_counts_and_clocks(pilot_scene: Scene) -> None:
    scene = pilot_scene
    assert [camera.source_count for camera in scene.cameras] == [3302, 9907, 9906, 9906, 9907]
    for camera in scene.cameras:
        assert sum(1 for _ in preview_access_units(scene, camera)) == 60
    assert len(scene.trajectory.times_ns) == 329300
    assert (np.diff(scene.trajectory.times_ns) == 1000000).all()
    assert len(scene.hands.times_ns) > 140  # native 30 Hz, ~5.1 seconds after MPS starts
    assert np.median(np.diff(scene.hands.times_ns)) == 33333000
    assert np.isnan(scene.trajectory.at(scene.cameras[0].times_ns[:8])).all()
    assert scene.cameras[1].times_ns[0] != scene.cameras[2].times_ns[0]


@pytest.mark.integration
def test_sixty_frames_all_layers(pilot_source: Path, pilot_scene: Scene, nvenc_ffmpeg: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("DATAFORGE_FFMPEG", str(nvenc_ffmpeg))
    dataset = AriaGen2PilotConfig(root=pilot_source.parent, sequences=("clean_0",), frame_limit=60).setup()
    identity, source = dataset.discover()[0]
    dataset.convert(identity, source, force=True)
    targets = dataset.targets(identity)
    assert set(targets) == {"base", "hand_pose", "projections"}
    for target in targets.values():
        assert rrc.RrdReader(target).recordings()[0].recording_id == identity.recording_id
        assert target.stat().st_size > 0
    base = read_chunks(targets["base"])
    for index, camera in enumerate(pilot_scene.cameras):
        chunks = [chunk for chunk in base if chunk.entity_path == schema.video_path(0, index) and "VideoStream:sample" in chunk.to_record_batch().schema.names]
        times = np.concatenate([chunk.to_record_batch().column("video_time").cast(pa.int64()).to_numpy() for chunk in chunks])
        np.testing.assert_array_equal(np.sort(times), camera.times_ns)
    rig = [chunk for chunk in base if chunk.entity_path == schema.rig_path(0) and not chunk.is_static]
    rows = [row for chunk in rig for row in chunk.to_record_batch().to_pylist() if row.get("Transform3D:translation") is not None]
    rows.sort(key=lambda row: row["video_time"])
    assert np.isnan(rows[0]["Transform3D:translation"]).all()
    for imu in (0, 1):
        assert any(chunk.entity_path == schema.gyro_path(0, imu) for chunk in base)
        assert any(chunk.entity_path == schema.accel_path(0, imu) for chunk in base)
    hand_chunks = read_chunks(targets["hand_pose"])
    points = [chunk for chunk in hand_chunks if chunk.entity_path == schema.coco133_xyz_path() and not chunk.is_static]
    times = np.concatenate([chunk.to_record_batch().column("video_time").cast(pa.int64()).to_numpy() for chunk in points])
    np.testing.assert_array_equal(np.sort(times), pilot_scene.hands.times_ns)
    assert not any("coco133_uv" in chunk.entity_path for chunk in hand_chunks)
    projections = read_chunks(targets["projections"])
    for index in range(5):
        chunks = [chunk for chunk in projections if chunk.entity_path == schema.coco133_uv_projected_path(0, index) and not chunk.is_static]
        assert sum(chunk.num_rows for chunk in chunks) == len(pilot_scene.hands.times_ns)
    mtimes = {layer: path.stat().st_mtime_ns for layer, path in targets.items()}
    dataset.convert(identity, source, force=False)
    assert mtimes == {layer: path.stat().st_mtime_ns for layer, path in targets.items()}


@pytest.mark.integration
def test_hevc_encode_keeps_color_resolution_and_order(pilot_scene: Scene, nvenc_ffmpeg: Path, tmp_path: Path) -> None:
    from itertools import islice

    for camera in pilot_scene.cameras[:2]:
        clip = tmp_path / f"{camera.stream_id}.mp4"
        frames = islice(preview_access_units(pilot_scene, camera), 3)
        assert encode_frames_to_mp4(frames, clip, source=FrameSource("hevc"), fps=camera.fps, cq=AV1_CQ, gop=AV1_GOP, ffmpeg=nvenc_ffmpeg) == 3
        with av.open(str(clip)) as container:
            packets = [packet for packet in container.demux(video=0) if packet.pts is not None]
            assert len(packets) == 3
            assert all(packet.pts == packet.dts for packet in packets)
            assert container.streams.video[0].codec_context.codec.canonical_name == "av1"
        with av.open(str(clip)) as container:
            decoded = next(container.decode(video=0))
            assert (decoded.width, decoded.height) == (camera.calibration.width, camera.calibration.height)
            pixels = decoded.to_ndarray(format="rgb24").astype(np.float32)
            chroma = np.abs(pixels[:, :, 0] - pixels[:, :, 2])
            assert np.mean(chroma) > 1.0 if camera.stream_id == "214-1" else np.max(chroma) <= 2.0


@pytest.mark.golden
def test_full_lens_projection_matches_per_point_projection(pilot_scene: Scene) -> None:
    """The batched chain (pose, inverse, per-frame projection) against one point at a time.

    ``simplecv``'s tests hold the lens model itself to projectaria-tools' pixels.
    """
    batch = pilot_scene.hands
    selected = np.arange(0, len(batch.times_ns), 3)
    compared = 0
    for camera in pilot_scene.cameras:
        actual = project_to_calibration(camera.calibration, batch.device_poses[selected], batch.positions[selected])
        for row, index in enumerate(selected):
            pose = batch.device_poses[index]
            if not np.isfinite(pose).all():
                assert np.isnan(actual[row]).all()
                continue
            world_T_camera = pose @ camera.calibration.rig_T_cam.matrix()
            for joint in np.flatnonzero(np.isfinite(batch.positions[index]).all(axis=1)):
                point = np.linalg.solve(world_T_camera[:3, :3], batch.positions[index, joint].astype(np.float64) - world_T_camera[:3, 3])
                expected = fisheye624.project(camera.calibration, point[None])[0] if point[2] > 0.0 else np.full(2, np.nan)
                if np.isnan(expected).any():
                    assert np.isnan(actual[row, joint]).all()
                else:
                    np.testing.assert_allclose(actual[row, joint], expected, rtol=0.0, atol=1e-6)
                    compared += 1
    assert compared > 100


@pytest.mark.golden
@pytest.mark.parametrize("sequence", ["cook_0", "clean_0"])
def test_simplecv_parity_after_documented_pose_correction(sequence: str) -> None:
    """Compare retained rows; isolate simplecv's nearest-pose error in SE(3)."""
    root = AriaGen2PilotConfig().root
    reference_root = Path(os.environ.get("DATAFORGE_ARIA_GEN2_REFERENCE_ROOT", "/mnt/nas/datasets/exoego-forge-catalog-rig/aria-gen2"))
    source, reference = root / sequence, reference_root / f"{sequence}.rrd"
    required = [
        reference,
        source / "video.vrs",
        source / "mps/slam/closed_loop_trajectory.csv",
        source / "mps/hand_tracking/hand_tracking_results.csv",
    ]
    for path in required:
        raw_asset("Aria Gen2 parity asset", path)
    scene = read_scene(source)
    batch = scene.hands
    primary = scene.cameras[0].times_ns
    hand_indices = nearest_framesets(batch.times_ns, primary)
    compared = 0
    uncorrected_max = 0.0
    corrected_max = 0.0
    seen = 0
    for chunk in rrc.RrdReader(reference).stream():
        if chunk.entity_path != schema.coco133_xyz_path() or chunk.is_static:
            continue
        table = chunk.to_record_batch()
        if "Points3D:positions" not in table.schema.names:
            continue
        for stamp, values in zip(
            table.column("video_time").cast(pa.int64()).to_pylist(), table.column("Points3D:positions").to_pylist(), strict=True
        ):
            time = int(stamp) + int(primary[0])
            rgb = int(nearest_framesets(primary, np.array([time], dtype=np.int64))[0])
            assert abs(int(primary[rgb]) - time) <= 1000
            seen += 1
            if time < batch.times_ns[0] or time > batch.times_ns[-1]:
                continue  # simplecv clamps the clip head/tail
            index = int(hand_indices[rgb])
            previous = int(nearest_framesets(scene.trajectory.times_ns, batch.times_ns[index:index + 1])[0])
            assert previous >= 0
            old_pose = scene.trajectory.poses[previous]
            new_pose = batch.device_poses[index]
            expected = np.asarray(values, dtype=np.float64)
            ours = batch.positions[index].astype(np.float64)
            present = np.isfinite(ours).all(axis=1)
            for side, wrist, start in ((0, 9, 91), (1, 10, 112)):
                if batch.scores[index, side] <= 0.0:
                    present[wrist] = False
                    present[start : start + 21] = False
            np.testing.assert_array_equal(np.isfinite(expected).all(axis=1), present)
            if not present.any():
                continue
            uncorrected_max = max(uncorrected_max, float(np.linalg.norm(ours[present] - expected[present], axis=1).max()))
            correction = new_pose @ np.linalg.inv(old_pose)
            corrected = expected[present] @ correction[:3, :3].T + correction[:3, 3]
            error = float(np.linalg.norm(ours[present] - corrected, axis=1).max())
            corrected_max = max(corrected_max, error)
            assert error <= 1e-4
            compared += 1
    print(f"{sequence}: rows={seen}, compared={compared}, raw_max_m={uncorrected_max:.9f}, corrected_max_m={corrected_max:.9f}")
    assert seen == len(primary)
    assert compared > len(primary) * 0.5
