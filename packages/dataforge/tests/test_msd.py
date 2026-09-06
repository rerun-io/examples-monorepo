"""Reading one Monado SLAM sequence archive: calibration, csv rows, and PNG members.

Nothing here touches the network; every fixture is a synthetic sequence tree
zipped in ``tmp_path``, including a real Info-ZIP multi-volume set for the
split-archive reader.
"""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest
import serde.json
from jaxtyping import Float64, UInt8
from numpy import ndarray
from scipy.spatial.transform import Rotation
from simplecv.camera_parameters import (
    BrownConradyDistortion,
    Fisheye62Parameters,
    KannalaBrandtDistortion,
    PinholeParameters,
)

from dataforge.datasets.msd import (
    BasaltPose,
    CameraRow,
    MsdCalibration,
    TimestampedSamples,
    camera_parameters,
    first_timestamp_ns,
    open_member_reader,
    read_camera_index,
    read_numeric_csv,
)

# ── calibration ───────────────────────────────────────────────────────────

KB4_CALIBRATION: str = """
{"value0": {
  "T_imu_cam": [
    {"px": 0.069, "py": 0.121, "pz": 0.013, "qx": -0.0274, "qy": -0.0269, "qz": -0.7011, "qw": 0.7120},
    {"px": 0.069, "py": -0.013, "pz": 0.013, "qx": 0.0207, "qy": 0.0331, "qz": -0.6987, "qw": 0.7144}
  ],
  "intrinsics": [
    {"camera_type": "kb4", "intrinsics": {"fx": 420.5, "fy": 420.7, "cx": 469.5, "cy": 479.1,
      "k1": 0.193, "k2": 0.042, "k3": -0.233, "k4": 0.095}},
    {"camera_type": "kb4", "intrinsics": {"fx": 421.2, "fy": 421.5, "cx": 467.7, "cy": 484.5,
      "k1": 0.190, "k2": 0.055, "k3": -0.249, "k4": 0.103}}
  ],
  "resolution": [[960, 960], [960, 960]],
  "imu_update_rate": 1000.0,
  "cam_time_offset_ns": 0,
  "vignette": []
}}
"""
"""A two-camera kb4 calibration in basalt's format, trimmed from the Valve Index file."""

RADTAN8_CALIBRATION: str = """
{"value0": {
  "comment": "trimmed Reverb G2 calibration",
  "T_imu_cam": [
    {"px": -0.052, "py": 0.013, "pz": 0.006, "qx": -0.155, "qy": 0.034, "qz": 0.692, "qw": 0.704}
  ],
  "intrinsics": [
    {"camera_type": "pinhole-radtan8", "intrinsics": {"fx": 269.7, "fy": 269.8, "cx": 322.6, "cy": 228.9,
      "k1": 0.302, "k2": -0.021, "p1": -0.00025, "p2": 6.1e-05, "k3": 0.0158, "k4": 0.575, "k5": -0.063, "k6": 0.034,
      "rpmax": 2.7276}}
  ],
  "resolution": [[640, 480]]
}}
"""
"""A one-camera radtan8 calibration; ``rpmax`` and ``comment`` must be ignored, not rejected."""


def test_kb4_becomes_a_fisheye_with_its_four_radial_terms() -> None:
    calibration: MsdCalibration = serde.json.from_json(MsdCalibration, KB4_CALIBRATION)
    camera: PinholeParameters | Fisheye62Parameters = camera_parameters(calibration, 1, name="cam1")
    assert isinstance(camera, Fisheye62Parameters)
    assert isinstance(camera.distortion, KannalaBrandtDistortion)
    assert (camera.distortion.k1, camera.distortion.k2, camera.distortion.k3, camera.distortion.k4) == (0.190, 0.055, -0.249, 0.103)
    assert (camera.intrinsics.width, camera.intrinsics.height) == (960, 960)
    assert camera.intrinsics.fl_x == 421.2


def test_radtan8_becomes_a_pinhole_with_brown_conrady_and_no_rpmax() -> None:
    calibration: MsdCalibration = serde.json.from_json(MsdCalibration, RADTAN8_CALIBRATION)
    camera: PinholeParameters | Fisheye62Parameters = camera_parameters(calibration, 0, name="cam0")
    assert isinstance(camera, PinholeParameters)
    assert isinstance(camera.distortion, BrownConradyDistortion)
    assert (camera.distortion.k1, camera.distortion.k2, camera.distortion.p1, camera.distortion.p2) == (0.302, -0.021, -0.00025, 6.1e-05)
    assert (camera.distortion.k3, camera.distortion.k4, camera.distortion.k5, camera.distortion.k6) == (0.0158, 0.575, -0.063, 0.034)
    assert (camera.intrinsics.width, camera.intrinsics.height) == (640, 480)


def test_extrinsics_are_the_camera_pose_in_the_rig_frame() -> None:
    """``T_imu_cam`` is ``rig_T_cam``; the rig frame is the IMU frame."""
    calibration: MsdCalibration = serde.json.from_json(MsdCalibration, KB4_CALIBRATION)
    camera: PinholeParameters | Fisheye62Parameters = camera_parameters(calibration, 0, name="cam0")
    pose: BasaltPose = calibration.value0.T_imu_cam[0]
    expected_rig_R_cam: Float64[ndarray, "3 3"] = Rotation.from_quat([pose.qx, pose.qy, pose.qz, pose.qw]).as_matrix()
    rig_R_cam: Float64[ndarray, "3 3"] = np.asarray(camera.extrinsics.world_R_cam, dtype=np.float64)
    rig_t_cam: Float64[ndarray, "3"] = np.asarray(camera.extrinsics.world_t_cam, dtype=np.float64)
    np.testing.assert_allclose(rig_R_cam, expected_rig_R_cam, atol=1e-12)
    np.testing.assert_allclose(rig_t_cam, [pose.px, pose.py, pose.pz], atol=1e-12)


# ── csv readers ───────────────────────────────────────────────────────────

CAMERA_CSV: bytes = b"#timestamp [ns],filename\n13000000000000,13000000000000.png\n13000018518000,13000018518000.png\n"
IMU_CSV: bytes = (
    b"#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
    b"a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n"
    b"12999000000000,0.01,-0.02,0.03,0.1,-0.2,9.8\n"
    b"12999001000000,0.011,-0.021,0.031,0.11,-0.21,9.81\n"
)
GT_CSV: bytes = (
    b"#timestamp [ns], p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []\n"
    b"12998000000000,0.0,0.1,0.2,1.0,0.0,0.0,0.0\n"
    b"12998001000000,0.01,0.11,0.21,1.0,0.0,0.0,0.0\n"
)


def test_camera_index_rows_carry_a_typed_stamp_and_filename() -> None:
    rows: list[CameraRow] = read_camera_index(CAMERA_CSV)
    assert [row.timestamp_ns for row in rows] == [13000000000000, 13000018518000]
    assert rows[1].filename == "13000018518000.png"


def test_numeric_csv_splits_the_clock_from_the_values() -> None:
    samples: TimestampedSamples = read_numeric_csv(IMU_CSV, num_values=6)
    assert samples.times_ns.dtype == np.int64
    np.testing.assert_array_equal(samples.times_ns, [12999000000000, 12999001000000])
    np.testing.assert_allclose(samples.values[0], [0.01, -0.02, 0.03, 0.1, -0.2, 9.8])
    assert samples.values.shape == (2, 6)


def test_a_single_row_numeric_csv_still_reads_as_a_table() -> None:
    one_row: bytes = b"\n".join(IMU_CSV.splitlines()[:2]) + b"\n"
    samples: TimestampedSamples = read_numeric_csv(one_row, num_values=6)
    assert samples.times_ns.shape == (1,)
    assert samples.values.shape == (1, 6)


def test_gt_contributes_only_its_first_stamp_to_the_shared_origin() -> None:
    # The gt layer is a separate rrd, but both layers must share t0, so the base
    # converter reads gt's first row and nothing else.
    assert first_timestamp_ns(GT_CSV) == 12998000000000


def test_an_empty_csv_is_an_error_not_a_silent_zero() -> None:
    with pytest.raises(ValueError, match="no data rows"):
        first_timestamp_ns(b"#timestamp [ns], p_RS_R_x [m]\n")


# ── archive member reader ─────────────────────────────────────────────────

SEQUENCE: str = "MIO09_short_1_updown"
"""Sequence stem of every synthetic archive below; also its top directory inside the zip."""


FRAME_WIDTH: int = 192
"""Frame width; NVENC refuses anything much smaller, so the fixture is not tiny."""
FRAME_HEIGHT: int = 160
"""Frame height, likewise above NVENC's minimum."""


def png_frame(index: int) -> bytes:
    """One grayscale PNG with a square that moves with ``index``, as MSD ships them.

    Sensor-like noise is deliberate: a flat gradient compresses to a couple of
    kilobytes, and the split-archive fixture below needs an archive that really
    spans volumes.
    """
    noise: UInt8[ndarray, "160 192"] = np.random.default_rng(index).integers(0, 96, (FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
    frame: UInt8[ndarray, "160 192"] = np.tile(np.linspace(0, 159, FRAME_WIDTH, dtype=np.uint8), (FRAME_HEIGHT, 1)) + noise
    left: int = (index * 6) % (FRAME_WIDTH - 16)
    frame[8:24, left : left + 16] = np.uint8(255 - (index * 11) % 256)
    success, buffer = cv2.imencode(".png", frame)
    assert success
    return buffer.tobytes()


@dataclass(frozen=True, slots=True)
class StreamClocks:
    """What the synthetic tree actually wrote, so assertions read it back rather than recompute it."""

    firsts: dict[str, int]
    """First timestamp of every stream, by stream name (``cam0``, ``imu0``, ``mag0``, ``gt``)."""
    lasts: dict[str, int]
    """Last timestamp of every stream, same keys."""


def sequence_tree(root: Path, *, num_cameras: int, num_frames: int, with_magnetometer: bool) -> StreamClocks:
    """Write one synthetic ``<SEQ>/mav0/...`` tree and report the clock of every stream."""
    base_ns: int = 13_000_000_000_000
    firsts: dict[str, int] = {}
    lasts: dict[str, int] = {}
    for camera in range(num_cameras):
        data_dir: Path = root / SEQUENCE / "mav0" / f"cam{camera}" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        # Irregular steps around 18.5 ms (~54 fps), offset per camera: a real rig's clock.
        stamps: list[int] = [base_ns + camera * 300_000 + index * 18_518_000 + (index * 7919) % 400_000 for index in range(num_frames)]
        firsts[f"cam{camera}"] = stamps[0]
        lasts[f"cam{camera}"] = stamps[-1]
        rows: list[str] = ["#timestamp [ns],filename"]
        for index, stamp in enumerate(stamps):
            (data_dir / f"{stamp}.png").write_bytes(png_frame(index))
            rows.append(f"{stamp},{stamp}.png")
        (data_dir.parent / "data.csv").write_text("\n".join(rows) + "\n")
        # A decoy the converter must ignore, exactly as the real archives ship it.
        (data_dir.parent / "data.extra.csv").write_text("#timestamp [ns],exposure\n0,0\n")

    imu_dir: Path = root / SEQUENCE / "mav0" / "imu0"
    imu_dir.mkdir(parents=True, exist_ok=True)
    imu_first: int = base_ns - 5_000_000
    firsts["imu0"] = imu_first
    lasts["imu0"] = imu_first + 59 * 1_000_000
    imu_rows: list[str] = ["#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]"]
    for index in range(60):
        stamp: int = imu_first + index * 1_000_000
        imu_rows.append(f"{stamp},{0.01 * index},{-0.02 * index},{0.03 * index},{0.1},{-0.2},{9.81}")
    (imu_dir / "data.csv").write_text("\n".join(imu_rows) + "\n")

    if with_magnetometer:
        mag_dir: Path = root / SEQUENCE / "mav0" / "mag0"
        mag_dir.mkdir(parents=True, exist_ok=True)
        mag_first: int = base_ns - 2_000_000
        firsts["mag0"] = mag_first
        lasts["mag0"] = mag_first + 5 * 20_000_000
        mag_rows: list[str] = ["#timestamp [ns], x, y, z"]
        for index in range(6):
            mag_rows.append(f"{mag_first + index * 20_000_000},{300.0 + index},{-40.0},{12.0 * index}")
        (mag_dir / "data.csv").write_text("\n".join(mag_rows) + "\n")

    gt_dir: Path = root / SEQUENCE / "mav0" / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    # Earlier than every other stream on purpose: gt owns t0.
    gt_first: int = base_ns - 9_000_000
    firsts["gt"] = gt_first
    # gt is not logged by the base layer, so its last stamp never bounds the duration.
    gt_rows: list[str] = ["#timestamp [ns], p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []"]
    for index in range(40):
        gt_rows.append(f"{gt_first + index * 1_000_000},{0.001 * index},0.0,0.0,1.0,0.0,0.0,0.0")
    (gt_dir / "data.csv").write_text("\n".join(gt_rows) + "\n")
    return StreamClocks(firsts=firsts, lasts=lasts)


def test_plain_zip_reader_serves_csvs_and_frames_in_order(tmp_path: Path) -> None:
    tree: Path = tmp_path / "tree"
    sequence_tree(tree, num_cameras=2, num_frames=4, with_magnetometer=False)
    archive: Path = tmp_path / f"{SEQUENCE}.zip"
    shutil.make_archive(str(archive.with_suffix("")), "zip", root_dir=tree)

    with open_member_reader([archive], tmp_path / "work") as reader:
        rows: list[CameraRow] = read_camera_index(reader.csv_bytes(f"{SEQUENCE}/mav0/cam1/data.csv"))
        members: list[str] = [f"{SEQUENCE}/mav0/cam1/data/{row.filename}" for row in rows]
        frames: list[bytes] = list(reader.png_frames(members))
    assert len(frames) == 4
    assert all(frame.startswith(b"\x89PNG") for frame in frames)
    assert frames == [png_frame(index) for index in range(4)]


@pytest.mark.skipif(shutil.which("zip") is None, reason="needs Info-ZIP's zip to build a multi-volume fixture")
def test_split_archive_reader_extracts_one_camera_at_a_time(tmp_path: Path) -> None:
    """Python's zipfile cannot read a spanned archive at all, so 7-Zip does."""
    tree: Path = tmp_path / "tree"
    sequence_tree(tree, num_cameras=2, num_frames=12, with_magnetometer=False)
    subprocess.run(["zip", "-q", "-0", "-r", "-s", "64k", str(tmp_path / f"{SEQUENCE}.zip"), SEQUENCE], cwd=tree, check=True)
    volumes: list[Path] = sorted(tmp_path.glob(f"{SEQUENCE}.z*"))
    assert len(volumes) > 1, "the fixture must really be split for this test to mean anything"
    with pytest.raises(zipfile.BadZipFile):
        zipfile.ZipFile(tmp_path / f"{SEQUENCE}.zip").read(f"{SEQUENCE}/mav0/cam0/data/{sorted((tree / SEQUENCE / 'mav0' / 'cam0' / 'data').iterdir())[0].name}")

    work: Path = tmp_path / "work"
    with open_member_reader(volumes, work) as reader:
        rows: list[CameraRow] = read_camera_index(reader.csv_bytes(f"{SEQUENCE}/mav0/cam0/data.csv"))
        frames: list[bytes] = list(reader.png_frames([f"{SEQUENCE}/mav0/cam0/data/{row.filename}" for row in rows]))
    assert frames == [png_frame(index) for index in range(12)]
    # The camera's extracted PNGs are gone again; peak scratch is one camera, not the sequence.
    assert not list(work.rglob("*.png"))
