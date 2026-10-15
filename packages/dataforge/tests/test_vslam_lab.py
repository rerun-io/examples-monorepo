"""vslam-lab: one converter for every VSLAM-LAB benchmark, one catalog dataset per benchmark.

The unit tests write tiny VSLAM-LAB sequence trees (the real headers, synthetic frames) and read the layers back
the way a consumer does. Encoding needs NVENC, so the conversions are integration tests; so is the real-asset
preview, which skips naming the sequence when it is not on this machine.
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import pyarrow as pa
import pytest
import rerun.chunk as rrc
from conftest import column_rows, raw_asset, read_back, recording_properties  # pyrefly: ignore[missing-import]
from jaxtyping import Float64, UInt8, UInt16
from numpy import ndarray
from test_vslam_lab_source import EUROC_CALIBRATION, TUM_CALIBRATION  # pyrefly: ignore[missing-import]

from dataforge import paths, schema, writing
from dataforge.datasets import dataset_defaults
from dataforge.datasets.vslam_lab import BENCHMARKS, VslamLabConfig, VslamLabDataset, log_depth
from dataforge.datasets.vslam_lab_source import read_frame_index
from dataforge.identity import SequenceIdentity

TUM_STAMPS: tuple[int, ...] = (1305031102175303936, 1305031102211214080, 1305031102243211008)
EUROC_STAMPS: tuple[int, ...] = (1403636579763555584, 1403636579813555456, 1403636579863555584)
EUROC_CAM1_T_BS: tuple[float, ...] = (
    0.0125552670891, -0.9997550997230, 0.0182237714554, -0.0198435579556,
    0.9995987811510, 0.0130119051815, 0.0251588363115, 0.0453689425024,
    -0.0253898008918, 0.0179005838253, 0.9995173470780, 0.0078621244704,
    0.0, 0.0, 0.0, 1.0,
)  # fmt: skip
"""cam1's body_T_sensor, as EUROC_CALIBRATION writes it."""
EUROC_BODY_POSE: str = (
    "#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [], v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], "
    "v_RS_R_z [m s^-1], b_w_RS_S_x [rad s^-1], b_w_RS_S_y [rad s^-1], b_w_RS_S_z [rad s^-1], b_a_RS_S_x [m s^-2], b_a_RS_S_y [m s^-2], b_a_RS_S_z [m s^-2]\n"
    "1403636579763555584,4.688319,-1.786938,0.783338,0.534108,-0.153029,-0.827383,-0.082152,-0.027876,0.033207,0.800006,-0.003172,0.021267,0.078502,-0.025266,0.136696,0.075593\n"
    "1403636579863555584,4.688177,-1.786770,0.787350,0.534640,-0.152990,-0.826976,-0.082863,-0.029272,0.033992,0.804771,-0.003172,0.021267,0.078502,-0.025266,0.136696,0.075593\n"
)
"""EuRoC's own mav0/state_groundtruth_estimate0/data.csv (header and rows as upstream writes them): world_T_imu0, w first."""
VSLAM_LAB_EUROC_GROUNDTRUTH: str = (
    "ts (ns),tx (m),ty (m),tz (m),qx,qy,qz,qw\n1403636579763555584,4.66,-1.84,0.78,1,0,0,0\n1403636579863555584,4.66,-1.84,0.82,1,0,0,0\n"
)
"""VSLAM-LAB's own EuRoC groundtruth.csv: cam0 positions with a constant placeholder in the quaternion columns."""


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
    (sequence / "groundtruth.csv").write_text(VSLAM_LAB_EUROC_GROUNDTRUTH)
    (sequence / "state_groundtruth_estimate0.csv").write_text(EUROC_BODY_POSE)
    return sequence


def kept_sequence_zip(sequence: Path) -> Path:
    """The sequence zip VSLAM-LAB keeps under its standard retention, ``<EUROC>/<room>/<seq>/<seq>.zip``, holding EuRoC's
    own body pose; the extracted copy ``download`` would have written is removed."""
    archive: Path = sequence.parent / "machine_hall" / sequence.name / f"{sequence.name}.zip"
    archive.parent.mkdir(parents=True)
    with ZipFile(archive, "w") as kept:
        kept.writestr("mav0/state_groundtruth_estimate0/data.csv", EUROC_BODY_POSE)
        kept.writestr("mav0/cam0/data.csv", "#timestamp [ns],filename\n")
    (sequence / "state_groundtruth_estimate0.csv").unlink()
    return archive


def dataset(root: Path, benchmark: str) -> VslamLabDataset:
    config: VslamLabConfig = replace(VslamLabConfig(), benchmark=benchmark, root=root)  # pyrefly: ignore[bad-argument-type]
    built = config.setup()
    assert isinstance(built, VslamLabDataset)
    return built


def rebuild_derived_layers(built: VslamLabDataset, identity: SequenceIdentity, source: Path) -> None:
    """The rebuild path: base is published and the frames are gone, so gt and sensor_metadata come from the
    sequence's small files alone; no frame is read and no GPU is needed."""
    base: Path = built.targets(identity)[paths.BASE_LAYER]
    base.parent.mkdir(parents=True)
    base.touch()
    for frames in source.glob("*_[0-9]"):
        shutil.rmtree(frames)
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


def test_download_refuses_a_benchmark_with_no_sequence(tmp_path: Path) -> None:
    (tmp_path / "EUROC" / "MH_02_easy").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="EUROC holds no sequence"):
        dataset(tmp_path, "euroc").download()


def test_download_counts_sequences_beside_vslam_labs_support_directories(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    euroc_sequence(tmp_path)
    (tmp_path / "EUROC" / "supp_v2" / "gtFiles").mkdir(parents=True)
    dataset(tmp_path, "euroc").download()
    assert "vslam-lab-euroc: 1 sequence(s)" in capsys.readouterr().out


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
    quaternions: pa.Table = column_rows(store, f"{schema.rig_path(0)}:Transform3D:quaternion")
    np.testing.assert_allclose(quaternions.column(1).to_pylist()[1], [[0.6129, 0.5966, -0.3316, -0.3980]], rtol=1e-6)
    gt: dict[str, object] = recording_properties(store, "gt")
    assert (gt["num_poses"], gt["world_up"], gt["num_sanitized"]) == (2, "+z", 0)
    # TUM ships no IMU noise model, so there is no sensor_metadata layer to write.
    assert not paths.rrd_path(tmp_path / "out", layer=paths.SENSOR_METADATA_LAYER, identity=identity).exists()


def test_euroc_gt_is_euroc_s_own_body_pose_never_vslam_lab_s_groundtruth_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    source: Path = euroc_sequence(tmp_path / "raw")
    built: VslamLabDataset = dataset(tmp_path / "raw", "euroc")
    identity, _ = built.discover()[0]
    rebuild_derived_layers(built, identity, source)
    store: rrc.ChunkStore = read_back(paths.rrd_path(tmp_path / "out", layer=paths.GT_LAYER, identity=identity))
    poses: pa.Table = column_rows(store, f"{schema.rig_path(0)}:Transform3D:translation")
    assert poses.column(schema.TIMELINE).cast(pa.int64()).to_pylist() == [1403636579763555584, 1403636579863555584]
    np.testing.assert_allclose(poses.column(1).to_pylist()[1], [[4.688177, -1.786770, 0.787350]], rtol=1e-6)
    quaternions: pa.Table = column_rows(store, f"{schema.rig_path(0)}:Transform3D:quaternion")
    np.testing.assert_allclose(quaternions.column(1).to_pylist()[1], [[-0.152990, -0.826976, -0.082863, 0.534640]], rtol=1e-6)
    assert recording_properties(store, "gt")["source"] == BENCHMARKS["euroc"].gt.source


def test_download_extracts_euroc_s_body_pose_from_the_kept_sequence_zip(tmp_path: Path) -> None:
    source: Path = euroc_sequence(tmp_path)
    kept_sequence_zip(source)
    dataset(tmp_path, "euroc").download()
    assert (source / "state_groundtruth_estimate0.csv").read_text() == EUROC_BODY_POSE


def test_download_refuses_a_euroc_sequence_with_neither_its_body_pose_nor_the_kept_zip(tmp_path: Path) -> None:
    source: Path = euroc_sequence(tmp_path)
    (source / "state_groundtruth_estimate0.csv").unlink()
    with pytest.raises(FileNotFoundError, match="MH_01_easy.zip"):
        dataset(tmp_path, "euroc").download()


def test_convert_names_download_when_euroc_s_body_pose_was_never_extracted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    source: Path = euroc_sequence(tmp_path / "raw")
    kept_sequence_zip(source)
    built: VslamLabDataset = dataset(tmp_path / "raw", "euroc")
    identity, _ = built.discover()[0]
    with pytest.raises(FileNotFoundError, match="dataforge-download"):
        rebuild_derived_layers(built, identity, source)


def test_a_sequence_with_every_layer_published_is_skipped_without_reading_its_gt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    source: Path = euroc_sequence(tmp_path / "raw")
    built: VslamLabDataset = dataset(tmp_path / "raw", "euroc")
    identity, _ = built.discover()[0]
    rebuild_derived_layers(built, identity, source)
    (source / "state_groundtruth_estimate0.csv").unlink()
    assert built.convert(identity, source, force=False) == built.targets(identity)[paths.BASE_LAYER]


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


def test_derived_layers_read_the_sequence_files_in_place_and_keep_no_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    source: Path = euroc_sequence(tmp_path / "raw")
    built: VslamLabDataset = dataset(tmp_path / "raw", "euroc")
    identity, _ = built.discover()[0]
    rebuild_derived_layers(built, identity, source)
    for layer in (paths.GT_LAYER, paths.SENSOR_METADATA_LAYER):
        assert paths.rrd_path(tmp_path / "out", layer=layer, identity=identity).is_file()
    assert not (tmp_path / "out" / paths.SIDECAR_DIR).exists()


def test_a_forced_refresh_whose_gt_is_refused_replaces_no_layer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    source: Path = euroc_sequence(tmp_path / "raw")
    built: VslamLabDataset = dataset(tmp_path / "raw", "euroc")
    identity, _ = built.discover()[0]
    rebuild_derived_layers(built, identity, source)
    published: dict[str, bytes] = {layer: target.read_bytes() for layer, target in built.targets(identity).items()}
    (source / "state_groundtruth_estimate0.csv").write_text(EUROC_BODY_POSE.splitlines()[0] + "\n")
    monkeypatch.setattr(VslamLabDataset, "write_base", lambda *_: pytest.fail("base was rewritten before the gt was read"))
    with pytest.raises(ValueError, match="state_groundtruth_estimate0.csv"):
        built.convert(identity, source, force=True)
    assert {layer: target.read_bytes() for layer, target in built.targets(identity).items()} == published


def test_depth_is_sent_as_the_sensor_png_bytes_on_the_capture_clocks_only(tmp_path: Path) -> None:
    source: Path = tum_sequence(tmp_path / "raw")
    rrd: Path = tmp_path / "depth.rrd"
    with writing.recording_to(rrd, recording_id="depth") as recording:
        log_depth(recording, source, read_frame_index(source / "rgb.csv")["depth_0"], 0, meter=5000.0)
    store: rrc.ChunkStore = read_back(rrd)
    entity: str = schema.depth_path(0, 0)
    blobs: pa.Table = column_rows(store, f"{entity}:EncodedDepthImage:blob")
    assert blobs.column(schema.TIMELINE).cast(pa.int64()).to_pylist() == list(TUM_STAMPS)
    assert [bytes(cell[0]) for cell in blobs.column(1).to_pylist()] == [(source / f"depth_0/{index}.png").read_bytes() for index in range(3)]
    assert column_rows(store, f"{entity}:EncodedDepthImage:meter").column(1).to_pylist() == [[5000.0]] * 3
    # Only the capture clocks: no wall-clock log_time of whenever the conversion ran.
    assert sorted(column.name for column in store.schema().index_columns()) == sorted([schema.FRAME_INDEX, schema.TIMELINE])


def test_a_depth_frame_that_is_not_a_16_bit_gray_png_is_refused(tmp_path: Path) -> None:
    source: Path = tum_sequence(tmp_path / "raw")
    cv2.imwrite(str(source / "depth_0/1.png"), np.zeros((128, 160), dtype=np.uint8))
    with writing.recording_to(tmp_path / "depth.rrd", recording_id="depth") as recording, pytest.raises(ValueError, match="depth_0/1.png"):
        log_depth(recording, source, read_frame_index(source / "rgb.csv")["depth_0"], 0, meter=5000.0)


def test_the_layer_set_follows_the_benchmark(tmp_path: Path) -> None:
    identity: SequenceIdentity = SequenceIdentity("vslam-lab-rgbdtum", ("any",))
    assert list(dataset(tmp_path, "rgbdtum").targets(identity)) == [paths.BASE_LAYER, paths.GT_LAYER]
    assert list(dataset(tmp_path, "euroc").targets(identity)) == [paths.BASE_LAYER, paths.GT_LAYER, paths.SENSOR_METADATA_LAYER]


def test_an_imu_benchmark_whose_calibration_has_no_noise_model_fails_loudly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    source: Path = euroc_sequence(tmp_path / "raw")
    calibration: Path = source / "calibration.yaml"
    calibration.write_text("\n".join(line for line in calibration.read_text().splitlines() if "sigma_" not in line) + "\n")
    built: VslamLabDataset = dataset(tmp_path / "raw", "euroc")
    identity, _ = built.discover()[0]
    with pytest.raises(ValueError, match="sensor_metadata"):
        rebuild_derived_layers(built, identity, source)


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
    depth: pa.Table = column_rows(store, f"{schema.depth_path(0, 0)}:EncodedDepthImage:meter")
    assert depth.column(schema.TIMELINE).cast(pa.int64()).to_pylist() == list(TUM_STAMPS)
    assert depth.column(1).to_pylist()[0] == [5000.0]
    assert sorted(column.name for column in store.schema().index_columns()) == sorted([schema.FRAME_INDEX, schema.TIMELINE])
    capture: dict[str, object] = recording_properties(store, "capture")
    assert (capture["num_cameras"], capture["num_frames"], capture["start_time_ns"], capture["benchmark"]) == (1, 3, TUM_STAMPS[0], "RGBDTUM")
    assert "unshifted" in str(capture["clock_source"])
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
    # simplecv stores a camera as cam_T_rig with relation ChildFromParent (2), so cam_01's stored translation is
    # T_BS inverted; the IMU is the body, so its transform is the identity.
    static: dict[str, list[list[float]]] = store.reader(index=None, contents="/world/**").to_arrow_table().to_pylist()[0]
    cam1_T_body: Float64[ndarray, "4 4"] = np.linalg.inv(np.array(EUROC_CAM1_T_BS, dtype=np.float64).reshape(4, 4))
    assert static[f"{schema.cam_path(0, 1)}:Transform3D:relation"] == [2]
    np.testing.assert_allclose(static[f"{schema.cam_path(0, 1)}:Transform3D:translation"], [cam1_T_body[:3, 3]], rtol=1e-6)
    np.testing.assert_allclose(static[f"{schema.imu_path(0, 0)}:Transform3D:translation"], [[0.0, 0.0, 0.0]])


@pytest.mark.integration
def test_the_real_freiburg1_xyz_converts_every_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc_ffmpeg: Path) -> None:
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    root: Path = VslamLabConfig().root
    raw_asset("VSLAM-LAB RGBDTUM freiburg1_xyz frames", root / "RGBDTUM/rgbd_dataset_freiburg1_xyz/rgb_0/1305031102.175304.png")
    built: VslamLabDataset = dataset(root, "rgbdtum")
    identity, source = next(pair for pair in built.discover() if pair[0].parts == ("rgbd_dataset_freiburg1_xyz",))
    store: rrc.ChunkStore = read_back(built.convert(identity, source, force=False))
    assert column_rows(store, f"{schema.video_path(0, 0)}:VideoStream:sample").num_rows == 798
    assert column_rows(store, f"{schema.depth_path(0, 0)}:EncodedDepthImage:meter").num_rows == 798
