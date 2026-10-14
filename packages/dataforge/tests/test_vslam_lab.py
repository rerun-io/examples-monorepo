"""vslam-lab: one converter for every VSLAM-LAB benchmark, one catalog dataset per benchmark.

The unit tests write tiny VSLAM-LAB sequence trees (the real headers, synthetic frames) and read the layers back
the way a consumer does. Encoding needs NVENC, so the conversions are integration tests; so is the real-asset
preview, which skips naming the sequence when it is not on this machine.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pytest
import rerun.chunk as rrc
from conftest import column_rows, raw_asset, read_back, recording_properties  # pyrefly: ignore[missing-import]
from jaxtyping import UInt8, UInt16
from numpy import ndarray
from test_vslam_lab_source import EUROC_CALIBRATION, TUM_CALIBRATION  # pyrefly: ignore[missing-import]

from dataforge import paths, schema
from dataforge.datasets import dataset_defaults
from dataforge.datasets.vslam_lab import BENCHMARKS, VslamLabConfig, VslamLabDataset
from dataforge.identity import SequenceIdentity

TUM_STAMPS: tuple[int, ...] = (1305031102175303936, 1305031102211214080, 1305031102243211008)
EUROC_STAMPS: tuple[int, ...] = (1403636579763555584, 1403636579813555456, 1403636579863555584)


def tum_sequence(root: Path, name: str = "rgbd_dataset_freiburg1_xyz") -> Path:
    """A three-frame RGBDTUM sequence as VSLAM-LAB lays it out: RGB PNGs, 16-bit depth PNGs, world_T_cam gt."""
    sequence: Path = root / "RGBDTUM" / name
    (sequence / "rgb_0").mkdir(parents=True)
    (sequence / "depth_0").mkdir()
    (sequence / "calibration.yaml").write_text(TUM_CALIBRATION.replace("[640, 480]", "[160, 128]"))
    rows: list[str] = ["ts_rgb_0 (ns),path_rgb_0,ts_depth_0 (ns),path_depth_0"]
    for index, stamp in enumerate(TUM_STAMPS):
        rgb: UInt8[ndarray, "128 160 3"] = np.full((128, 160, 3), 40 * index, dtype=np.uint8)
        depth: UInt16[ndarray, "128 160"] = np.full((128, 160), 5000 * (index + 1), dtype=np.uint16)
        cv2.imwrite(str(sequence / f"rgb_0/{index}.png"), rgb)
        cv2.imwrite(str(sequence / f"depth_0/{index}.png"), depth)
        rows.append(f"{stamp},rgb_0/{index}.png,{stamp},depth_0/{index}.png")
    (sequence / "rgb.csv").write_text("\n".join(rows) + "\n")
    (sequence / "groundtruth.csv").write_text(
        "ts (ns),tx (m),ty (m),tz (m),qx,qy,qz,qw\n"
        "1305031102170000000,1.3563,0.6305,1.6380,0.6132,0.5962,-0.3311,-0.3986\n"
        "1305031102240000000,1.3543,0.6306,1.6360,0.6129,0.5966,-0.3316,-0.3980\n"
    )
    return sequence


def euroc_sequence(root: Path) -> Path:
    """A three-frame EUROC stereo-inertial sequence."""
    sequence: Path = root / "EUROC" / "MH_01_easy"
    for camera in ("rgb_0", "rgb_1"):
        (sequence / camera).mkdir(parents=True)
    (sequence / "calibration.yaml").write_text(EUROC_CALIBRATION.replace("[752, 480]", "[160, 128]"))
    rows: list[str] = ["ts_rgb_0 (ns),path_rgb_0,ts_rgb_1 (ns),path_rgb_1"]
    for index, stamp in enumerate(EUROC_STAMPS):
        for camera in ("rgb_0", "rgb_1"):
            cv2.imwrite(str(sequence / f"{camera}/{stamp}.png"), np.full((128, 160), 30 * index, dtype=np.uint8))
        rows.append(f"{stamp},rgb_0/{stamp}.png,{stamp},rgb_1/{stamp}.png")
    (sequence / "rgb.csv").write_text("\n".join(rows) + "\n")
    imu_rows: list[str] = [f"{EUROC_STAMPS[0] + 5_000_000 * step},0.1,0.2,0.3,9.8,0.0,0.1" for step in range(21)]
    (sequence / "imu_0.csv").write_text(
        "ts (ns),wx (rad s^-1),wy (rad s^-1),wz (rad s^-1),ax (m s^-2),ay (m s^-2),az (m s^-2)\n" + "\n".join(imu_rows) + "\n"
    )
    (sequence / "groundtruth.csv").write_text(
        "ts (ns),tx (m),ty (m),tz (m),qx,qy,qz,qw\n"
        "1403636579763555584,4.688319,-1.786938,0.783338,-0.153029,-0.827383,-0.082152,0.534108\n"
        "1403636579863555584,4.688177,-1.786770,0.787350,-0.152990,-0.826976,-0.082863,0.534640\n"
    )
    return sequence


def dataset(root: Path, benchmark: str, **overrides: int) -> VslamLabDataset:
    config: VslamLabConfig = replace(VslamLabConfig(), benchmark=benchmark, root=root, **overrides)  # pyrefly: ignore[bad-argument-type]
    built = config.setup()
    assert isinstance(built, VslamLabDataset)
    return built


def rebuild_derived_layers(built: VslamLabDataset, identity: SequenceIdentity, source: Path) -> None:
    """Convert with a base layer already published, the layer rule's rebuild path: gt and sensor_metadata come
    from their sidecars alone, so no frame is read and no GPU is needed."""
    base: Path = built.targets(identity)[paths.BASE_LAYER]
    base.parent.mkdir(parents=True)
    base.touch()
    built.convert(identity, source, force=False)


def test_every_benchmark_is_one_catalog_dataset_named_after_it() -> None:
    assert isinstance(dataset_defaults["vslam-lab"], VslamLabConfig)
    for benchmark in BENCHMARKS:
        assert replace(VslamLabConfig(), benchmark=benchmark).name == f"vslam-lab-{benchmark}"  # pyrefly: ignore[bad-argument-type]


def test_discover_finds_complete_sequences_of_the_chosen_benchmark_only(tmp_path: Path) -> None:
    tum_sequence(tmp_path)
    euroc_sequence(tmp_path)
    (tmp_path / "RGBDTUM" / "half_downloaded" / "rgb_0").mkdir(parents=True)
    found: list[tuple[SequenceIdentity, Path]] = dataset(tmp_path, "rgbdtum").discover()
    assert [identity.recording_id for identity, _ in found] == ["vslam-lab-rgbdtum__rgbd_dataset_freiburg1_xyz"]
    assert found[0][1] == tmp_path / "RGBDTUM" / "rgbd_dataset_freiburg1_xyz"


def test_download_names_what_a_sequence_lacks(tmp_path: Path) -> None:
    (tmp_path / "EUROC" / "MH_02_easy").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="EUROC"):
        dataset(tmp_path, "euroc").download()


def test_gt_is_world_t_rig_on_the_upstream_stamps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    source: Path = tum_sequence(tmp_path / "raw")
    built: VslamLabDataset = dataset(tmp_path / "raw", "rgbdtum")
    identity, _ = built.discover()[0]
    rebuild_derived_layers(built, identity, source)
    store: rrc.ChunkStore = read_back(paths.rrd_path(tmp_path / "out", layer=paths.GT_LAYER, identity=identity))
    poses: pa.Table = column_rows(store, f"{schema.rig_path(0)}:Transform3D:translation")
    assert poses.column(schema.TIMELINE).cast(pa.int64()).to_pylist() == [1305031102170000000, 1305031102240000000]
    np.testing.assert_allclose(poses.column(1).to_pylist()[1], [[1.3543, 0.6306, 1.6360]], rtol=1e-6)
    gt: dict[str, object] = recording_properties(store, "gt")
    assert (gt["num_poses"], gt["world_up"], gt["has_rotation"]) == (2, "+z", True)
    # TUM ships no IMU noise model, so there is no sensor_metadata layer to write.
    assert not paths.rrd_path(tmp_path / "out", layer=paths.SENSOR_METADATA_LAYER, identity=identity).exists()


def test_position_only_gt_is_flagged_and_announced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    source: Path = euroc_sequence(tmp_path / "raw")
    # VSLAM-LAB's own EuRoC groundtruth.csv: cam0 positions with a constant (1, 0, 0, 0) in the quaternion columns.
    (source / "groundtruth.csv").write_text(
        "ts (ns),tx (m),ty (m),tz (m),qx,qy,qz,qw\n1403636580863555669,4.66,-1.84,0.78,1,0,0,0\n1403636580913555383,4.66,-1.84,0.82,1,0,0,0\n"
    )
    built: VslamLabDataset = dataset(tmp_path / "raw", "euroc")
    identity, _ = built.discover()[0]
    rebuild_derived_layers(built, identity, source)
    store: rrc.ChunkStore = read_back(paths.rrd_path(tmp_path / "out", layer=paths.GT_LAYER, identity=identity))
    assert recording_properties(store, "gt")["has_rotation"] is False
    assert "position-only" in capsys.readouterr().out


def test_the_imu_noise_model_lands_on_the_imu_node_as_sensor_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    source: Path = euroc_sequence(tmp_path / "raw")
    built: VslamLabDataset = dataset(tmp_path / "raw", "euroc")
    identity, _ = built.discover()[0]
    rebuild_derived_layers(built, identity, source)
    store: rrc.ChunkStore = read_back(paths.rrd_path(tmp_path / "out", layer=paths.SENSOR_METADATA_LAYER, identity=identity))
    table: pa.Table = store.reader(index=None, contents=f"{schema.imu_path(0, 0)}/**").to_arrow_table()
    from simplecv.imu_calibration import ImuCalibration

    imu: ImuCalibration = ImuCalibration.from_catalog(table, schema.imu_path(0, 0))
    assert (imu.gyro_noise_density, imu.accel_noise_density, imu.gyro_bias_random_walk, imu.accel_bias_random_walk, imu.rate_hz) == (
        2e-3,
        2e-2,
        2e-4,
        2e-2,
        200.0,
    )


@pytest.mark.integration
def test_a_tum_sequence_converts_video_depth_and_gt_on_upstream_stamps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc_ffmpeg: Path) -> None:
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    built: VslamLabDataset = dataset(tmp_path / "raw", "rgbdtum")
    tum_sequence(tmp_path / "raw")
    identity, source = built.discover()[0]
    base_rrd: Path = built.convert(identity, source, force=False)
    store: rrc.ChunkStore = read_back(base_rrd)
    video: pa.Table = column_rows(store, f"{schema.video_path(0, 0)}:VideoStream:sample")
    assert video.column(schema.TIMELINE).cast(pa.int64()).to_pylist() == list(TUM_STAMPS)
    depth: pa.Table = column_rows(store, f"{schema.depth_path(0, 0)}:DepthImage:meter")
    assert depth.column(schema.TIMELINE).cast(pa.int64()).to_pylist() == list(TUM_STAMPS)
    assert depth.column(1).to_pylist()[0] == [5000.0]
    capture: dict[str, object] = recording_properties(store, "capture")
    assert (capture["num_cameras"], capture["num_frames"], capture["start_time_ns"], capture["benchmark"]) == (1, 3, TUM_STAMPS[0], "RGBDTUM")
    assert (paths.sidecar_path(tmp_path / "out", identity, "groundtruth.csv")).read_bytes() == (source / "groundtruth.csv").read_bytes()
    assert paths.rrd_path(tmp_path / "out", layer=paths.GT_LAYER, identity=identity).is_file()


@pytest.mark.integration
def test_an_euroc_sequence_converts_stereo_and_the_imu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc_ffmpeg: Path) -> None:
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    built: VslamLabDataset = dataset(tmp_path / "raw", "euroc")
    euroc_sequence(tmp_path / "raw")
    identity, source = built.discover()[0]
    store: rrc.ChunkStore = read_back(built.convert(identity, source, force=False))
    for camera in (0, 1):
        assert column_rows(store, f"{schema.video_path(0, camera)}:VideoStream:sample").num_rows == 3
    assert column_rows(store, f"{schema.accel_path(0, 0)}:Scalars:scalars").num_rows == 21
    rig: dict[str, object] = store.reader(index=None, contents=schema.rig_path(0)).to_arrow_table().to_pylist()[0]
    assert rig[f"{schema.rig_path(0)}:reference"] == ["imu_00"]


@pytest.mark.integration
def test_a_real_sequence_previews_through_every_layer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc_ffmpeg: Path) -> None:
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    config: VslamLabConfig = VslamLabConfig(benchmark="rgbdtum", frame_limit=10)
    raw_asset("VSLAM-LAB RGBDTUM freiburg1_xyz", config.root / "RGBDTUM/rgbd_dataset_freiburg1_xyz/rgb.csv")
    built: VslamLabDataset = dataset(config.root, "rgbdtum", frame_limit=10)
    identity, source = next(pair for pair in built.discover() if pair[0].parts == ("rgbd_dataset_freiburg1_xyz",))
    store: rrc.ChunkStore = read_back(built.convert(identity, source, force=False))
    assert column_rows(store, f"{schema.video_path(0, 0)}:VideoStream:sample").num_rows == 10
    assert column_rows(store, f"{schema.depth_path(0, 0)}:DepthImage:meter").num_rows == 10
