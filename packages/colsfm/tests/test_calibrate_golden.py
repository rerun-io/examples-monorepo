"""`colsfm calibrate` on two real RoboCap rigs, checked against their factory Kalibr calibrations (Pablo, 2026-11-27).

Per cap, the four SLAM fisheyes estimated from scratch on 300 rig frames: every image registers, and per camera the
reprojection error with the estimate is not worse than with the factory calibration held on the same tracks
(`colsfm.calibration_check`). The differences to factory (rotation, direction, focal, principal point) are printed as
information, not checked: the factory calibration is not a sub-degree answer key (Cap A's front pair is 0.18 deg off,
2026-08-30).

Assets: the RaCo and FlexGlue graphs; Cap A s00000021 on the Spark catalog (`COLSFM_CAP_A_CATALOG`, default
`rerun+http://dgx-spark:51235`); the Cap B session 28 base converted with its own 1.22 ms camera-IMU shift
(`COLSFM_CAP_B_BASE`). A missing asset skips with its name. About 10 minutes per cap on the 5090.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

import pytest
from rerun.catalog import CatalogClient

from colsfm.calibrate import CalibrateOptions, CalibrationResult, calibrate
from colsfm.calibration_check import CameraCheck, check_calibration, check_table
from colsfm.catalog import CatalogCameraModel, CatalogInputConfig, read_catalog_cameras, serve_catalog_input
from colsfm.matching import FLEXGLUE_MODEL_DIR, FLEXGLUE_MODEL_FILES
from colsfm.model_assets import RACO_DYNAMIC_ONNX_PATH, RACO_ONNX_PATH

pytestmark = pytest.mark.golden

CAMERAS: Final[tuple[str, ...]] = ("left_front", "right_front", "left", "right")
"""The RoboCap SLAM cameras."""

CAP_A_CATALOG: Final[str] = os.environ.get("COLSFM_CAP_A_CATALOG", "rerun+http://dgx-spark:51235")
CAP_B_BASE: Final[Path] = Path(
    os.environ.get("COLSFM_CAP_B_BASE", "/mnt/nas/ssd-8tb/datasets/robocap/rrd-s28-capb-shift1p22ms/base/robocap__fe6fede545c972fa__s00000028.rrd")
)


def _require_graphs() -> None:
    missing: list[str] = [
        str(path)
        for path in (RACO_ONNX_PATH, RACO_DYNAMIC_ONNX_PATH, *(FLEXGLUE_MODEL_DIR / name for name in FLEXGLUE_MODEL_FILES.values()))
        if not path.exists()
    ]
    if missing:
        pytest.skip(f"missing asset(s): {', '.join(missing)}")


def _check(source: CatalogInputConfig, work_dir: Path) -> None:
    with serve_catalog_input(source) as served:
        result: CalibrationResult = calibrate(served, CalibrateOptions(work_dir=work_dir))
        factory: tuple[CatalogCameraModel, ...] = read_catalog_cameras(
            CatalogClient(served.catalog_url).get_dataset(name=served.dataset), served.segment, CAMERAS
        )
    checks: tuple[CameraCheck, ...] = check_calibration(result, factory)
    print(
        f"[golden] {source.segment or source.rrd}: {sum(result.registered_images.values())} images in {result.seconds:.0f} s\n{check_table(checks)}"
    )
    assert all(count == 300 for count in result.registered_images.values()), result.registered_images
    assert all(check.not_worse for check in checks), check_table(checks)


def test_cap_a_calibrates_no_worse_than_its_factory_calibration(tmp_path: Path) -> None:
    _require_graphs()
    url = urlparse(CAP_A_CATALOG.replace("rerun+", ""))
    try:
        socket.create_connection((url.hostname, url.port), timeout=3).close()
    except OSError:
        pytest.skip(f"missing asset: the Cap A catalog {CAP_A_CATALOG} does not answer")
    source = CatalogInputConfig(
        dataset="robocap",
        segment="robocap__f408193e6447b3b0__s00000021",
        catalog_url=CAP_A_CATALOG,
        cameras=CAMERAS,
        trim_start_seconds=2.0,
        trim_end_seconds=2.0,
    )
    _check(source, tmp_path)


def test_cap_b_calibrates_no_worse_than_its_factory_calibration(tmp_path: Path) -> None:
    _require_graphs()
    if not CAP_B_BASE.is_file():
        pytest.skip(f"missing asset: {CAP_B_BASE}")
    # A 152 s window 5 minutes in, inside the first 30 minutes of the 4 h session.
    source = CatalogInputConfig(dataset="robocap", rrd=CAP_B_BASE, cameras=CAMERAS, trim_start_seconds=300.0, trim_end_seconds=14031.0)
    _check(source, tmp_path)
