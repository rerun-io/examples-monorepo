"""tum-vi: the raw ASL tree of TUM-VI's 512x512 16-bit release → base, gt and sensor_metadata layers.

The unit tests write tiny sequence trees (the dataset's own yaml and csv headers, synthetic 16-bit frames) and read the
layers back the way a consumer does. Encoding needs NVENC, so a whole conversion is an integration test.
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pytest
import rerun.chunk as rrc
from conftest import column_rows, read_back, recording_properties  # pyrefly: ignore[missing-import]
from jaxtyping import Float64, UInt16
from numpy import ndarray

from dataforge import paths, schema
from dataforge.basalt import CalibratedCamera, Kb4Intrinsics
from dataforge.datasets import dataset_defaults
from dataforge.datasets.tumvi import TumViConfig, TumViDataset
from dataforge.datasets.tumvi_source import (
    MocapSpan,
    StereoFrames,
    mocap_spans,
    pair_stereo,
    read_camchain,
    read_imu_noise,
)
from dataforge.euroc import read_camera_index
from dataforge.identity import SequenceIdentity

CAMCHAIN: str = """cam0:
  T_cam_imu:
  - [-0.9995250378696743, 0.029615343885863205, -0.008522328211654736, 0.04727988224914392]
  - [0.0075019185074052044, -0.03439736061393144, -0.9993800792498829, -0.047443232143367084]
  - [-0.02989013031643309, -0.998969345370175, 0.03415885127385616, -0.0681999605066297]
  - [0.0, 0.0, 0.0, 1.0]
  cam_overlaps: [1]
  camera_model: pinhole
  distortion_coeffs: [0.0034823894022493434, 0.0007150348452162257, -0.0020532361418706202,
    0.00020293673591811182]
  distortion_model: equidistant
  intrinsics: [190.97847715128717, 190.9733070521226, 254.93170605935475, 256.8974428996504]
  resolution: [160, 128]
  rostopic: /cam0/image_raw
cam1:
  T_cam_imu:
  - [-0.9995110484978581, 0.030299116376600627, -0.0077218830287333565, -0.053697434688869734]
  - [0.008104079263822521, 0.012511643720192351, -0.9998888851620987, -0.046131737923635924]
  - [-0.030199136245891378, -0.9994625667418545, -0.012751072573940885, -0.07149261284195751]
  - [0.0, 0.0, 0.0, 1.0]
  T_cn_cnm1:
  - [0.9999994457734953, -0.0008233639921576076, -0.0006561436136444361, -0.10106110275180535]
  - [0.0007916877528168117, 0.9988994619156757, -0.04689603624058988, -0.0019764575873431013]
  - [0.0006940340102242531, 0.04689549078870055, 0.9988995601463054, -0.0011756424802046581]
  - [0.0, 0.0, 0.0, 1.0]
  cam_overlaps: [0]
  camera_model: pinhole
  distortion_coeffs: [0.0034003170790442797, 0.001766278153469831, -0.00266312569781606,
    0.0003299517423931039]
  distortion_model: equidistant
  intrinsics: [190.44236969414825, 190.4344384721956, 252.59949716835982, 254.91723064636983]
  resolution: [160, 128]
  rostopic: /cam1/image_raw
"""
"""TUM-VI's dso/camchain.yaml as every 512_16 sequence ships it, with the resolution cut to the test frames."""
IMU_CONFIG: str = """rostopic: /imu0
update_rate: 200.0  # Hz

#accelerometer_noise_density: 0.0014     # m/s^1.5
accelerometer_noise_density: 0.0028     # m/s^1.5
accelerometer_random_walk:   0.00086    # m/s^2.5
gyroscope_noise_density:     0.00016    # rad/s^0.5
gyroscope_random_walk:       0.000022   # rad/s^1.5
"""
"""TUM-VI's dso/imu_config.yaml: the inflated noise model, commented Allan values left in as upstream does."""
STAMPS: tuple[int, ...] = (1520530308199447626, 1520530308249448626, 1520530308299448626)
"""Three stereo stamps on the dataset's own (Unix epoch) clock."""
CAM0_ONLY: int = 1520530308349448626
"""A frame cam0 took and cam1 did not, as magistrale4's last cam0 frame."""
MOCAP_HEADER: str = "#timestamp [ns], p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []\n"
MOCAP_ROWS: tuple[str, ...] = (
    "1520530308189679351,0.8417820384,-0.2193354118,1.2499749141,0.9996498117,0.0037373093,0.0097169635,-0.0243283190",
    "1520530308198012351,0.8417344970,-0.2193450998,1.2500003423,0.9996655192,0.0038381891,0.0094471392,-0.0237669833",
    "1520530310198012351,0.9417344970,-0.2193450998,1.2500003423,0.9996655192,0.0038381891,0.0094471392,-0.0237669833",
)
"""mocap0 rows (w-first quaternion) with a 2 s pause before the last: the rig left the room and came back."""


def tumvi_sequence(root: Path, name: str = "room1") -> Path:
    """A three-pair TUM-VI sequence as the 512_16 tarball extracts it, plus one cam0 frame cam1 lacks."""
    sequence: Path = root / f"dataset-{name}_512_16"
    (sequence / "dso").mkdir(parents=True)
    (sequence / "dso/camchain.yaml").write_text(CAMCHAIN)
    (sequence / "dso/imu_config.yaml").write_text(IMU_CONFIG)
    for camera, stamps in (("cam0", (*STAMPS, CAM0_ONLY)), ("cam1", STAMPS)):
        (sequence / f"mav0/{camera}/data").mkdir(parents=True)
        for index, stamp in enumerate(stamps):
            frame: UInt16[ndarray, "128 160"] = np.full((128, 160), 4096 * (index + 1), dtype=np.uint16)
            cv2.imwrite(str(sequence / f"mav0/{camera}/data/{stamp}.png"), frame)
        (sequence / f"mav0/{camera}/data.csv").write_text("#timestamp [ns],filename\n" + "".join(f"{stamp},{stamp}.png\n" for stamp in stamps))
    (sequence / "mav0/imu0").mkdir()
    imu_rows: list[str] = [f"{STAMPS[0] - 20_000_000 + 5_000_000 * step},0.1,-0.08,0.02,-0.35,0.0,9.92" for step in range(30)]
    (sequence / "mav0/imu0/data.csv").write_text(
        "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n"
        + "\n".join(imu_rows)
        + "\n"
    )
    (sequence / "mav0/mocap0").mkdir()
    (sequence / "mav0/mocap0/data.csv").write_text(MOCAP_HEADER + "\n".join(MOCAP_ROWS) + "\n")
    return sequence


def dataset(root: Path) -> TumViDataset:
    built = replace(TumViConfig(), root=root).setup()
    assert isinstance(built, TumViDataset)
    return built


def rebuild_derived_layers(built: TumViDataset, identity: SequenceIdentity, source: Path) -> None:
    """The rebuild path: base is published and the frames are gone, so gt and sensor_metadata come from the small files."""
    base: Path = built.targets(identity)[paths.BASE_LAYER]
    base.parent.mkdir(parents=True)
    base.touch()
    for camera in ("cam0", "cam1"):
        shutil.rmtree(source / f"mav0/{camera}/data")
    built.convert(identity, source, force=False)


def test_tum_vi_is_one_catalog_dataset_named_tum_vi() -> None:
    assert isinstance(dataset_defaults["tum-vi"], TumViConfig)
    assert TumViConfig().name == "tum-vi"


def test_the_camchain_is_kb4_with_each_camera_posed_in_the_imu_frame(tmp_path: Path) -> None:
    (tmp_path / "camchain.yaml").write_text(CAMCHAIN)
    cameras: tuple[CalibratedCamera, ...] = read_camchain(tmp_path / "camchain.yaml")
    assert [camera.index for camera in cameras] == [0, 1]
    assert [camera.camera_model for camera in cameras] == ["kb4", "kb4"]
    assert cameras[0].model == Kb4Intrinsics(
        fx=190.97847715128717,
        fy=190.9733070521226,
        cx=254.93170605935475,
        cy=256.8974428996504,
        k1=0.0034823894022493434,
        k2=0.0007150348452162257,
        k3=-0.0020532361418706202,
        k4=0.00020293673591811182,
    )
    assert cameras[0].resolution == (160, 128)
    # Kalibr states T_cam_imu; the rig (IMU) frame wants its inverse, T_imu_cam.
    cam1_T_imu: Float64[ndarray, "4 4"] = np.array(
        [
            [-0.9995110484978581, 0.030299116376600627, -0.0077218830287333565, -0.053697434688869734],
            [0.008104079263822521, 0.012511643720192351, -0.9998888851620987, -0.046131737923635924],
            [-0.030199136245891378, -0.9994625667418545, -0.012751072573940885, -0.07149261284195751],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    imu_T_cam1: Float64[ndarray, "4 4"] = np.linalg.inv(cam1_T_imu)
    pose = cameras[1].rig_pose
    np.testing.assert_allclose([pose.px, pose.py, pose.pz], imu_T_cam1[:3, 3], atol=1e-12)
    # The stereo baseline (T_cn_cnm1's translation) is what the two poses imply: ~10.1 cm.
    pose0 = cameras[0].rig_pose
    assert np.linalg.norm(np.subtract([pose.px, pose.py, pose.pz], [pose0.px, pose0.py, pose0.pz])) == pytest.approx(0.10108, abs=1e-4)


def test_a_camchain_with_another_distortion_model_is_refused_naming_the_file(tmp_path: Path) -> None:
    (tmp_path / "camchain.yaml").write_text(CAMCHAIN.replace("equidistant", "radtan"))
    with pytest.raises(ValueError, match="camchain.yaml: not a TUM-VI camchain"):
        read_camchain(tmp_path / "camchain.yaml")


def test_the_imu_noise_model_is_the_files_inflated_values(tmp_path: Path) -> None:
    (tmp_path / "imu_config.yaml").write_text(IMU_CONFIG)
    noise = read_imu_noise(tmp_path / "imu_config.yaml")
    assert (noise.accelerometer_noise_density, noise.accelerometer_random_walk, noise.gyroscope_noise_density, noise.gyroscope_random_walk) == (
        0.0028,
        0.00086,
        0.00016,
        0.000022,
    )
    assert noise.update_rate == 200.0


def test_stereo_frames_pair_by_equal_stamp_and_count_the_frames_left_over() -> None:
    cam0 = read_camera_index(b"#timestamp [ns],filename\n10,10.png\n20,20.png\n30,30.png\n40,40.png\n")
    cam1 = read_camera_index(b"#timestamp [ns],filename\n10,10.png\n21,21.png\n30,30.png\n")
    stereo: StereoFrames = pair_stereo(cam0, cam1)
    assert stereo.times_ns.tolist() == [10, 30]
    assert stereo.filenames == (("10.png", "30.png"), ("10.png", "30.png"))
    assert stereo.unpaired == (2, 1)


def test_stamps_that_do_not_increase_are_refused() -> None:
    cam0 = read_camera_index(b"#timestamp [ns],filename\n20,20.png\n10,10.png\n")
    with pytest.raises(ValueError, match="cam0/data.csv stamps are not strictly increasing"):
        pair_stereo(cam0, cam0)


def test_mocap_splits_into_spans_where_the_rig_left_the_room() -> None:
    times_ns = np.array([0, 8_000_000, 16_000_000, 3_000_000_000, 3_008_000_000], dtype=np.int64)
    assert mocap_spans(times_ns) == (MocapSpan(0, 16_000_000, 3), MocapSpan(3_000_000_000, 3_008_000_000, 2))
    assert mocap_spans(np.array([], dtype=np.int64)) == ()


def test_discover_finds_complete_512_16_sequences_only(tmp_path: Path) -> None:
    tumvi_sequence(tmp_path, "room1")
    tumvi_sequence(tmp_path, "magistrale1")
    (tmp_path / "dataset-room2_512_16" / "mav0").mkdir(parents=True)
    (tmp_path / "dataset-room3_1024_16").mkdir()
    found: list[tuple[SequenceIdentity, Path]] = dataset(tmp_path).discover()
    assert [identity.recording_id for identity, _ in found] == ["tum-vi__magistrale1", "tum-vi__room1"]
    assert found[1][1] == tmp_path / "dataset-room1_512_16"


def test_gt_is_every_mocap_sample_as_world_t_imu_on_the_dataset_clock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    source: Path = tumvi_sequence(tmp_path / "raw")
    built: TumViDataset = dataset(tmp_path / "raw")
    identity, _ = built.discover()[0]
    rebuild_derived_layers(built, identity, source)
    store: rrc.ChunkStore = read_back(paths.rrd_path(tmp_path / "out", layer=paths.GT_LAYER, identity=identity))
    poses: pa.Table = column_rows(store, f"{schema.rig_path(0)}:Transform3D:translation")
    stamps: list[int] = [int(row.split(",")[0]) for row in MOCAP_ROWS]
    assert poses.column(schema.TIMELINE).cast(pa.int64()).to_pylist() == stamps
    np.testing.assert_allclose(poses.column(1).to_pylist()[2], [[0.9417344970, -0.2193450998, 1.2500003423]], rtol=1e-6)
    quaternions: pa.Table = column_rows(store, f"{schema.rig_path(0)}:Transform3D:quaternion")
    np.testing.assert_allclose(quaternions.column(1).to_pylist()[0], [[0.0037373093, 0.0097169635, -0.0243283190, 0.9996498117]], rtol=1e-6)
    gt: dict[str, object] = recording_properties(store, "gt")
    assert (gt["num_poses"], gt["world_up"], gt["num_sanitized"], gt["num_spans"]) == (3, "+z", 0, 2)
    table: pa.Table = store.reader(index=None, contents="/__properties/**").to_arrow_table()
    assert table.column("property:gt:span_start_ns").to_pylist()[0] == [stamps[0], stamps[2]]
    assert table.column("property:gt:span_num_poses").to_pylist()[0] == [2, 1]


def test_sensor_metadata_is_the_noise_model_and_a_zero_time_shift_on_the_imu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    source: Path = tumvi_sequence(tmp_path / "raw")
    built: TumViDataset = dataset(tmp_path / "raw")
    identity, _ = built.discover()[0]
    rebuild_derived_layers(built, identity, source)
    store: rrc.ChunkStore = read_back(paths.rrd_path(tmp_path / "out", layer=paths.SENSOR_METADATA_LAYER, identity=identity))
    table: pa.Table = store.reader(index=None, contents=f"{schema.imu_path(0, 0)}/**").to_arrow_table()
    from simplecv.imu_calibration import ImuCalibration

    imu: ImuCalibration = ImuCalibration.from_catalog(table, schema.imu_path(0, 0))
    assert (imu.gyro_noise_density, imu.accel_noise_density, imu.gyro_bias_random_walk, imu.accel_bias_random_walk, imu.rate_hz) == (
        0.00016,
        0.0028,
        0.000022,
        0.00086,
        200.0,
    )
    # The catalog feed refuses an IMU without the shift it must add; TUM-VI's is zero.
    assert table.column(f"{schema.imu_path(0, 0)}:applied_time_shift_ns").to_pylist()[0] == [0]


def test_a_forced_refresh_with_a_bad_camchain_replaces_no_layer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    source: Path = tumvi_sequence(tmp_path / "raw")
    built: TumViDataset = dataset(tmp_path / "raw")
    identity, _ = built.discover()[0]
    rebuild_derived_layers(built, identity, source)
    published: dict[str, bytes] = {layer: target.read_bytes() for layer, target in built.targets(identity).items()}
    (source / "dso/camchain.yaml").write_text(CAMCHAIN.replace("equidistant", "radtan"))
    monkeypatch.setattr(TumViDataset, "write_base", lambda *_: pytest.fail("base was rewritten before the camchain was read"))
    with pytest.raises(ValueError, match="camchain"):
        built.convert(identity, source, force=True)
    assert {layer: target.read_bytes() for layer, target in built.targets(identity).items()} == published


def test_output_beneath_the_raw_tree_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "raw" / "rrd"))
    tumvi_sequence(tmp_path / "raw")
    built: TumViDataset = dataset(tmp_path / "raw")
    identity, source = built.discover()[0]
    with pytest.raises(ValueError, match="refusing to write beneath protected input"):
        built.convert(identity, source, force=False)


@pytest.mark.integration
def test_a_sequence_converts_to_the_base_schema_on_its_stereo_pairs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc_ffmpeg: Path) -> None:
    """The cut fixture: a three-pair sequence written by this test, converted, and its base rrd read back."""
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    tumvi_sequence(tmp_path / "raw")
    built: TumViDataset = dataset(tmp_path / "raw")
    identity, source = built.discover()[0]
    store: rrc.ChunkStore = read_back(built.convert(identity, source, force=False))
    for camera in (0, 1):
        video: pa.Table = column_rows(store, f"{schema.video_path(0, camera)}:VideoStream:sample")
        assert video.column(schema.TIMELINE).cast(pa.int64()).to_pylist() == list(STAMPS), "only the stereo pairs, unshifted"
    assert column_rows(store, f"{schema.accel_path(0, 0)}:Scalars:scalars").num_rows == 30
    rig: dict[str, object] = store.reader(index=None, contents=schema.rig_path(0)).to_arrow_table().to_pylist()[0]
    assert rig[f"{schema.rig_path(0)}:reference"] == ["imu_00"]
    static: dict[str, list[list[float]]] = store.reader(index=None, contents="/world/**").to_arrow_table().to_pylist()[0]
    assert static[f"{schema.cam_path(0, 1)}/pinhole:simplecv.components.DistortionModel"] == ["kannala_brandt"]
    assert static[f"{schema.cam_path(0, 0)}:camera_model"] == ["kb4"]
    np.testing.assert_allclose(static[f"{schema.imu_path(0, 0)}:Transform3D:translation"], [[0.0, 0.0, 0.0]])
    capture: dict[str, object] = recording_properties(store, "capture")
    assert (capture["num_cameras"], capture["num_frames"], capture["start_time_ns"]) == (2, 3, STAMPS[0])
    assert (capture["unpaired_cam0"], capture["unpaired_cam1"]) == (1, 0)
    assert sorted(column.name for column in store.schema().index_columns()) == sorted([schema.FRAME_INDEX, schema.TIMELINE])
    for layer in (paths.GT_LAYER, paths.SENSOR_METADATA_LAYER):
        assert paths.rrd_path(tmp_path / "out", layer=layer, identity=identity).is_file()
