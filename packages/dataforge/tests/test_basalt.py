"""basalt's calibration.json: the two camera models, the extrinsics, and the follow frame.

Every positive case reads one of the three **real** device files checked in under
``fixtures/msd/`` (verbatim copies — see ``fixtures/README.md``), so what is
asserted is what upstream actually ships rather than what a fixture happens to
make convenient. The negative cases take a real file and break one thing in it,
which is the only way to write down what "incomplete" means for a format whose
own writer never produces one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import calibration_fixture
from jaxtyping import Float64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from simplecv.camera_parameters import (
    BrownConradyDistortion,
    Fisheye62Parameters,
    KannalaBrandtDistortion,
    PinholeParameters,
)

from dataforge.basalt import (
    BasaltPose,
    CalibratedCamera,
    FollowFrame,
    Kb4Intrinsics,
    Radtan8Intrinsics,
    camera_parameters,
    follow_frame,
    load_calibration,
)

# ── the real files ────────────────────────────────────────────────────────


def test_the_index_ships_two_kb4_cameras_with_no_validity_radius() -> None:
    """kb4 is valid over the whole fisheye, so it declares no radius at all."""
    cameras: tuple[CalibratedCamera, ...] = load_calibration(calibration_fixture("index"))

    assert len(cameras) == 2
    assert [camera.index for camera in cameras] == [0, 1]
    assert {camera.camera_model for camera in cameras} == {"kb4"}
    assert all(isinstance(camera.model, Kb4Intrinsics) for camera in cameras)
    assert all(camera.distortion_valid_radius is None for camera in cameras)
    assert {camera.resolution for camera in cameras} == {(960, 960)}


@pytest.mark.parametrize(("device", "num_cameras"), [("g2", 4), ("odyssey", 2)])
def test_the_mocap_headsets_ship_radtan8_cameras_that_all_carry_an_rpmax(device: str, num_cameras: int) -> None:
    """The rational model stops holding past a radius, so every radtan8 camera states one."""
    cameras: tuple[CalibratedCamera, ...] = load_calibration(calibration_fixture(device))

    assert len(cameras) == num_cameras
    assert {camera.camera_model for camera in cameras} == {"pinhole-radtan8"}
    for camera in cameras:
        assert isinstance(camera.model, Radtan8Intrinsics)
        assert camera.distortion_valid_radius is not None, f"cam{camera.index} carries no rpmax"
    assert {camera.resolution for camera in cameras} == {(640, 480)}


def test_the_odyssey_ships_a_zero_rpmax_and_that_is_not_a_missing_one() -> None:
    """Upstream really writes ``"rpmax": 0.0`` on both Odyssey+ cameras.

    This is why presence and not truthiness is the invariant: a zero radius is a
    value the file states, so treating it as absent would mean rejecting a real
    device's calibration, and defaulting a *missing* one to zero would mean
    inventing that same claim for a truncated file. The G2's are ~2.8.
    """
    odyssey: tuple[CalibratedCamera, ...] = load_calibration(calibration_fixture("odyssey"))
    g2: tuple[CalibratedCamera, ...] = load_calibration(calibration_fixture("g2"))

    assert [camera.distortion_valid_radius for camera in odyssey] == [0.0, 0.0]
    assert all((camera.distortion_valid_radius or 0.0) > 2.7 for camera in g2)


def test_a_kb4_camera_becomes_a_fisheye_with_its_four_radial_terms() -> None:
    cameras: tuple[CalibratedCamera, ...] = load_calibration(calibration_fixture("index"))
    model: Kb4Intrinsics | Radtan8Intrinsics = cameras[1].model
    assert isinstance(model, Kb4Intrinsics)

    camera: PinholeParameters | Fisheye62Parameters = camera_parameters(cameras[1], name="cam1")

    assert isinstance(camera, Fisheye62Parameters)
    assert isinstance(camera.distortion, KannalaBrandtDistortion)
    terms: tuple[float, float, float, float] = (camera.distortion.k1, camera.distortion.k2, camera.distortion.k3, camera.distortion.k4)
    assert terms == (model.k1, model.k2, model.k3, model.k4)
    assert (camera.intrinsics.width, camera.intrinsics.height) == (960, 960)
    assert camera.intrinsics.fl_x == model.fx


def test_a_radtan8_camera_becomes_a_pinhole_with_all_eight_brown_conrady_terms() -> None:
    cameras: tuple[CalibratedCamera, ...] = load_calibration(calibration_fixture("g2"))
    model: Kb4Intrinsics | Radtan8Intrinsics = cameras[0].model
    assert isinstance(model, Radtan8Intrinsics)

    camera: PinholeParameters | Fisheye62Parameters = camera_parameters(cameras[0], name="cam0")

    assert isinstance(camera, PinholeParameters)
    distortion: object = camera.distortion
    assert isinstance(distortion, BrownConradyDistortion)
    assert (distortion.k1, distortion.k2, distortion.p1, distortion.p2) == (model.k1, model.k2, model.p1, model.p2)
    assert (distortion.k3, distortion.k4, distortion.k5, distortion.k6) == (model.k3, model.k4, model.k5, model.k6)
    assert (camera.intrinsics.width, camera.intrinsics.height) == (640, 480)


def test_extrinsics_are_the_camera_pose_in_the_rig_frame() -> None:
    """``T_imu_cam`` is ``rig_T_cam``; the rig frame is the IMU frame."""
    cameras: tuple[CalibratedCamera, ...] = load_calibration(calibration_fixture("index"))
    camera: PinholeParameters | Fisheye62Parameters = camera_parameters(cameras[0], name="cam0")

    pose: BasaltPose = cameras[0].rig_pose
    expected_rig_R_cam: Float64[ndarray, "3 3"] = Rotation.from_quat([pose.qx, pose.qy, pose.qz, pose.qw]).as_matrix()
    rig_R_cam: Float64[ndarray, "3 3"] = np.asarray(camera.extrinsics.world_R_cam, dtype=np.float64)
    rig_t_cam: Float64[ndarray, "3"] = np.asarray(camera.extrinsics.world_t_cam, dtype=np.float64)
    np.testing.assert_allclose(rig_R_cam, expected_rig_R_cam, atol=1e-12)
    np.testing.assert_allclose(rig_t_cam, [pose.px, pose.py, pose.pz], atol=1e-12)


# ── validation ────────────────────────────────────────────────────────────


def broken_calibration(tmp_path: Path, device: str, damage) -> Path:
    """Write one real calibration back out with ``damage`` applied to its ``value0``.

    Args:
        tmp_path: Where the damaged copy goes.
        device: Which real fixture to start from.
        damage: Mutates the parsed ``value0`` dict in place.

    Returns:
        The damaged file.
    """
    document: dict[str, Any] = json.loads(calibration_fixture(device).read_text())
    damage(document["value0"])
    target: Path = tmp_path / f"{device}-broken.json"
    target.write_text(json.dumps(document))
    return target


def test_a_kb4_camera_missing_a_radial_term_is_refused_by_camera_and_key(tmp_path: Path) -> None:
    """Zero-defaulting k4 would turn a truncated file into a differently distorted camera."""
    broken: Path = broken_calibration(tmp_path, "index", lambda value: value["intrinsics"][1]["intrinsics"].pop("k4"))

    with pytest.raises(ValueError, match="cam1.*'k4'"):
        load_calibration(broken)


def test_a_radtan8_camera_missing_its_rpmax_is_refused(tmp_path: Path) -> None:
    """The validity radius is part of the model, not an optional extra."""
    broken: Path = broken_calibration(tmp_path, "odyssey", lambda value: value["intrinsics"][0]["intrinsics"].pop("rpmax"))

    with pytest.raises(ValueError, match="cam0.*'rpmax'"):
        load_calibration(broken)


def test_lists_of_different_lengths_are_refused_rather_than_zipped(tmp_path: Path) -> None:
    """The three lists are per-camera; joining them short would silently drop a camera."""
    broken: Path = broken_calibration(tmp_path, "g2", lambda value: value["resolution"].pop())

    with pytest.raises(ValueError, match="must agree"):
        load_calibration(broken)


def test_a_camera_count_the_device_disagrees_with_is_refused() -> None:
    """The wrong device's file parses perfectly; only the expected count catches it."""
    with pytest.raises(ValueError, match="4 camera"):
        load_calibration(calibration_fixture("g2"), expected_cameras=2)


def test_an_unknown_camera_type_is_refused_by_name(tmp_path: Path) -> None:
    def retag(value: dict[str, Any]) -> None:
        value["intrinsics"][0]["camera_type"] = "eucm"

    broken: Path = broken_calibration(tmp_path, "index", retag)

    with pytest.raises(ValueError, match="'eucm'"):
        load_calibration(broken)


def test_a_resolution_that_is_not_two_positive_ints_is_refused(tmp_path: Path) -> None:
    broken: Path = broken_calibration(tmp_path, "index", lambda value: value["resolution"].__setitem__(0, [960, 0]))

    with pytest.raises(ValueError, match="positive"):
        load_calibration(broken)


# ── follow frame ──────────────────────────────────────────────────────────


def test_the_index_pair_looks_along_rig_z_and_calls_rig_minus_x_up() -> None:
    """Both answers are known outside this file, from the headset itself.

    The Index's cameras face where the wearer faces, and its 13.4 cm stereo
    baseline runs along rig y — so y is the lateral axis and the optical axis is
    rig +z. Up is then ±x, and MIO09's raw accelerometer mean (an accelerometer
    at rest reads *up*) picks -x, which is the measurement ``MSD_DEVICES``
    already records for this device.
    """
    frame: FollowFrame = follow_frame(load_calibration(calibration_fixture("index")))

    np.testing.assert_allclose(frame.forward, [0.0, 0.0, 1.0], atol=0.05)
    np.testing.assert_allclose(frame.up, [-1.0, 0.0, 0.0], atol=0.05)


UPRIGHT_PAIR: tuple[CalibratedCamera, ...] = (
    CalibratedCamera(
        index=0,
        rig_pose=BasaltPose(px=0.0, py=0.0, pz=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0),
        resolution=(1, 1),
        model=Kb4Intrinsics(fx=1.0, fy=1.0, cx=0.5, cy=0.5, k1=0.0, k2=0.0, k3=0.0, k4=0.0),
    ),
    CalibratedCamera(
        index=1,
        rig_pose=BasaltPose(px=1.0, py=0.0, pz=0.5, qx=0.0, qy=0.0, qz=0.0, qw=1.0),
        resolution=(1, 1),
        model=Kb4Intrinsics(fx=1.0, fy=1.0, cx=0.5, cy=0.5, k1=0.0, k2=0.0, k3=0.0, k4=0.0),
    ),
)
"""Two unrotated cameras whose baseline is deliberately *not* perpendicular to them.

Unrotated means RDF as it comes: the optical axis is rig +z and the sensor's
right is rig +x. The 0.5 m of ``pz`` between the two is the point — a baseline
with a component along the optical axis, which only an orthogonalized frame
survives. No real device is mounted like this, hence a built record rather than
a file.
"""


def test_a_baseline_tilted_out_of_the_image_plane_still_yields_an_orthonormal_frame() -> None:
    """Gram-Schmidt: the baseline is a hint about the lateral axis, not the axis itself."""
    frame: FollowFrame = follow_frame(UPRIGHT_PAIR)

    np.testing.assert_allclose(frame.forward, [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(frame.up, [0.0, -1.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("device", ["index", "g2", "odyssey"])
def test_a_follow_frame_is_two_orthogonal_unit_vectors(device: str) -> None:
    frame: FollowFrame = follow_frame(load_calibration(calibration_fixture(device)))
    assert np.linalg.norm(frame.forward) == pytest.approx(1.0, abs=1e-9)
    assert np.linalg.norm(frame.up) == pytest.approx(1.0, abs=1e-9)
    assert float(np.dot(frame.forward, frame.up)) == pytest.approx(0.0, abs=1e-9)


def test_a_single_camera_cannot_place_an_up_axis() -> None:
    with pytest.raises(ValueError, match="stereo pair"):
        follow_frame(UPRIGHT_PAIR, camera_indices=(0,))
