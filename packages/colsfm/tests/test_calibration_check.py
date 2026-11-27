"""`colsfm.calibration_check` on the synthetic fisheye rig (`calibrate_helpers`): the truth held reprojects as well as ours,
a wrong focal held reprojects worse, and the differences come out in the right units."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pycolmap
import pytest
from calibrate_helpers import HEIGHT, NAMES, TRUE_PARAMS, TRUE_RIG_T_CAM, WIDTH, synthetic_database

from colsfm.calibrate import CalibrationResult, cameras_from_rig, solve_rig
from colsfm.calibration_check import CameraCheck, check_calibration, check_table
from colsfm.catalog import CatalogCameraModel

ENTITIES: dict[str, str] = {name: f"/world/rig_00/cam_{index:02d}" for index, name in enumerate(NAMES)}


@pytest.fixture(scope="module")
def solved(tmp_path_factory: pytest.TempPathFactory) -> CalibrationResult:
    """The synthetic rig, solved once for the module (about 10 s)."""
    work: Path = tmp_path_factory.mktemp("calibrate")
    synthetic_database(work / "database.db")
    model: pycolmap.Reconstruction = solve_rig(work / "database.db", NAMES, work)
    cameras = cameras_from_rig(model, {name: index + 1 for index, name in enumerate(NAMES)}, ENTITIES, "OPENCV_FISHEYE")
    registered: dict[str, int] = dict.fromkeys(NAMES, 24)
    return CalibrationResult(
        cameras=cameras,
        layer_path=work / "unused.rrd",
        registered_images=registered,
        mean_reprojection_error_px=0.0,
        seconds=0.0,
        reconstruction=model,
    )


def _truth() -> tuple[CatalogCameraModel, ...]:
    """The true calibration, in metres and in a rig frame that is not the first camera's, as a factory's would be."""
    rig_T_first = np.eye(4)
    rig_T_first[:3, 3] = [0.3, -0.1, 0.2]
    return tuple(
        CatalogCameraModel(
            entity=ENTITIES[name],
            name=name,
            rig_T_cam=rig_T_first @ pose,
            k_matrix=np.array([[params[0], 0.0, params[2]], [0.0, params[1], params[3]], [0.0, 0.0, 1.0]]),
            width=WIDTH,
            height=HEIGHT,
            projection="OPENCV_FISHEYE",
            distortion=tuple(params[4:]),
        )
        for name, pose, params in zip(NAMES, TRUE_RIG_T_CAM, TRUE_PARAMS, strict=True)
    )


def test_the_truth_held_reprojects_as_well_as_the_estimate(solved: CalibrationResult) -> None:
    checks: tuple[CameraCheck, ...] = check_calibration(solved, _truth())
    for check in checks:
        assert check.registered_images == 24 and check.observations > 1000 and check.dropped_observations == 0
        assert check.reference_held_px == pytest.approx(check.reprojection_px, abs=0.05), check.name
        assert check.rotation_deg < 0.1 and check.direction_deg < 1.0, check.name
        assert abs(check.focal_change_percent) < 0.5 and check.principal_point_offset_px < 1.5, check.name
    assert "| front | 24 |" in check_table(checks)


def test_a_wrong_focal_held_reprojects_worse(solved: CalibrationResult) -> None:
    wrong: list[CatalogCameraModel] = list(_truth())
    wrong[1] = dataclasses.replace(wrong[1], k_matrix=wrong[1].k_matrix @ np.diag([1.03, 1.03, 1.0]))
    checks: tuple[CameraCheck, ...] = check_calibration(solved, wrong)
    assert all(check.not_worse for check in checks)
    assert checks[1].reference_held_px > checks[1].reprojection_px + 0.5
    assert checks[1].focal_change_percent == pytest.approx(-100.0 * (1.0 - 1.0 / 1.03), abs=0.5)


def test_a_reference_keeps_its_own_lens_model(solved: CalibrationResult) -> None:
    """A Brown-Conrady reference installs as COLMAP FULL_OPENCV (12 parameters) over a fisheye camera of the model."""
    from colsfm.calibration_check import held

    pinhole: list[CatalogCameraModel] = list(_truth())
    pinhole[2] = dataclasses.replace(pinhole[2], projection="DISTORTED_PINHOLE", distortion=(0.0,) * 8)
    installed: pycolmap.Reconstruction = held(solved.reconstruction, pinhole, scale=1.0)
    assert installed.cameras[3].model == pycolmap.CameraModelId.FULL_OPENCV and len(installed.cameras[3].params) == 12
    assert installed.cameras[1].model == pycolmap.CameraModelId.OPENCV_FISHEYE
    checks: tuple[CameraCheck, ...] = check_calibration(solved, pinhole)
    assert checks[2].reference_held_px > checks[2].reprojection_px  # a pinhole cannot hold a fisheye's rays
    assert {check.observations for check in checks} and all(check.dropped_observations >= 0 for check in checks)
