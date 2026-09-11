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
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import cv2
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
    rotate_camera_cw,
    upright_quarter_turns,
)


def rig_R_cam(camera: CalibratedCamera) -> Float64[ndarray, "3 3"]:
    """One record's ``T_imu_cam`` rotation as a matrix; the rig frame is the IMU frame."""
    pose: BasaltPose = camera.rig_pose
    return Rotation.from_quat([pose.qx, pose.qy, pose.qz, pose.qw]).as_matrix()

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
def test_the_mocap_headsets_ship_radtan8_cameras_that_all_state_an_rpmax(device: str, num_cameras: int) -> None:
    """Every radtan8 block holds the key; whether its value is a *limit* is the next test."""
    cameras: tuple[CalibratedCamera, ...] = load_calibration(calibration_fixture(device))

    assert len(cameras) == num_cameras
    assert {camera.camera_model for camera in cameras} == {"pinhole-radtan8"}
    for camera in cameras:
        assert isinstance(camera.model, Radtan8Intrinsics)
        assert camera.model.rpmax is not None, f"cam{camera.index} states no rpmax"
    assert {camera.resolution for camera in cameras} == {(640, 480)}


def test_the_odyssey_zero_rpmax_is_a_stated_value_but_not_a_validity_limit() -> None:
    """Upstream really writes ``"rpmax": 0.0`` on both Odyssey+ cameras, and basalt reads
    a non-positive rpmax as *the validity check is off*.

    So the file's value is kept as parsed — presence is what load-time validation
    checks, because a *missing* key is a truncated block and defaulting it to zero
    would invent this very claim — while the radius a camera node reports is
    ``None``: emitting ``0.0`` would tell a consumer the rational model holds
    nowhere on that camera. The G2's four cameras state a real ~2.8 and keep it.
    """
    odyssey: tuple[CalibratedCamera, ...] = load_calibration(calibration_fixture("odyssey"))
    g2: tuple[CalibratedCamera, ...] = load_calibration(calibration_fixture("g2"))

    odyssey_models: list[Radtan8Intrinsics] = [camera.model for camera in odyssey if isinstance(camera.model, Radtan8Intrinsics)]
    assert [model.rpmax for model in odyssey_models] == [0.0, 0.0], "the parsed value is the file's"
    assert [camera.distortion_valid_radius for camera in odyssey] == [None, None], "but it states no limit"
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


# ── the upright roll ──────────────────────────────────────────────────────

UPRIGHT_TURNS: dict[str, tuple[int, ...]] = {"index": (0, 0), "odyssey": (0, 0), "g2": (1, 1, 3, 3)}
"""Clockwise quarter turns each real camera needs to be encoded upright.

The Index's and the Odyssey+'s cameras are mounted upright, so they answer zero
and their converted output is unchanged by the whole upright rule. All four G2
cameras are rolled a quarter turn — the front pair (``cam0``/``cam1``) one way
and the sideways-looking pair (``cam2``/``cam3``) the other, which is why the
rule is per camera and not per device.
"""
UPRIGHT_ALIGNMENT_FLOOR: float = 0.99
"""How closely a chosen turn must point image-up at the headset's up, as a dot product.

0.99 is 8.1 degrees; the worst real camera manages 0.9985 (3.1 degrees), so this
asserts the turn is the right *quarter* without pinning the residual mounting roll.
"""
PROJECTION_TOLERANCE_PX: float = 1e-6
"""How far a rotated model's projection may sit from the pixel rotation of the original's.

The remap is exact algebra, so the honest bound is float noise: the real fixtures
measure 1.1e-13 px. A millionth of a pixel is far below anything a consumer could
observe and far above the noise, so a failure here is a wrong formula.
"""
PROJECTION_POINTS: int = 400
"""Random points per projection check; enough to sweep the frame, cheap enough to run per camera."""


def opencv_pixels(camera: CalibratedCamera, points_xyz: Float64[ndarray, "n_points 3"]) -> Float64[ndarray, "n_points 2"]:
    """Project camera-frame points through this record's model, using OpenCV as the reference.

    OpenCV is the independent source of truth here: it implements both the
    equidistant fisheye (``kb4``) and the eight-term Brown-Conrady rational model
    (``pinhole-radtan8``) that basalt's coefficients belong to, so the rotation
    remap can be checked against a projection nothing in this repo wrote.

    Args:
        camera: The record whose model and principal point to project with.
        points_xyz: Points in that camera's own frame, metres, all in front of it.

    Returns:
        Pixel coordinates, one row per point.
    """
    model: Kb4Intrinsics | Radtan8Intrinsics = camera.model
    camera_matrix: Float64[ndarray, "3 3"] = np.array(
        [[model.fx, 0.0, model.cx], [0.0, model.fy, model.cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    no_pose: Float64[ndarray, "3"] = np.zeros(3, dtype=np.float64)
    if isinstance(model, Kb4Intrinsics):
        fisheye_terms: Float64[ndarray, "4"] = np.array([model.k1, model.k2, model.k3, model.k4], dtype=np.float64)
        projected, _ = cv2.fisheye.projectPoints(points_xyz.reshape(-1, 1, 3), no_pose, no_pose, camera_matrix, fisheye_terms)
    else:
        # OpenCV's eight-coefficient order interleaves the tangential pair after k2.
        rational_terms: Float64[ndarray, "8"] = np.array(
            [model.k1, model.k2, model.p1, model.p2, model.k3, model.k4, model.k5, model.k6], dtype=np.float64
        )
        projected, _ = cv2.projectPoints(points_xyz.reshape(-1, 1, 3), no_pose, no_pose, camera_matrix, rational_terms)
    return np.asarray(projected, dtype=np.float64).reshape(-1, 2)


def rotate_pixels_cw(
    pixels_uv: Float64[ndarray, "n_points 2"], *, resolution: tuple[int, int], quarter_turns: int
) -> Float64[ndarray, "n_points 2"]:
    """Move pixel coordinates the way a clockwise image rotation moves them.

    This is the reference the intrinsics remap is checked against, and it is the
    same mapping ``test_encoding`` pins to ``np.rot90(frame, k=-1)``: one
    clockwise quarter turn of a ``width x height`` image sends ``(u, v)`` to
    ``((height - 1) - v, u)``, pixel centres at integer coordinates.

    Args:
        pixels_uv: Pixel coordinates in the original image.
        resolution: That image's ``(width, height)``.
        quarter_turns: Clockwise quarter turns to apply.

    Returns:
        The same points' coordinates in the rotated image.
    """
    columns_u: Float64[ndarray, "n_points"] = pixels_uv[:, 0]
    rows_v: Float64[ndarray, "n_points"] = pixels_uv[:, 1]
    width, height = resolution
    for _ in range(quarter_turns):
        columns_u, rows_v = (height - 1) - rows_v, columns_u
        width, height = height, width
    return np.column_stack([columns_u, rows_v])


def points_in_front_of(camera: CalibratedCamera) -> Float64[ndarray, "n_points 3"]:
    """Random camera-frame points inside the cone both models hold over, seeded per camera."""
    generator: np.random.Generator = np.random.default_rng(camera.index)
    return np.column_stack(
        [
            generator.uniform(-0.4, 0.4, PROJECTION_POINTS),
            generator.uniform(-0.4, 0.4, PROJECTION_POINTS),
            generator.uniform(0.8, 3.0, PROJECTION_POINTS),
        ]
    )


@pytest.mark.parametrize("device", ["index", "g2", "odyssey"])
@pytest.mark.parametrize("quarter_turns", [1, 2, 3])
def test_a_rolled_camera_projects_where_the_pixel_rotation_puts_the_original(device: str, quarter_turns: int) -> None:
    """The whole point of the roll: the rotated model describes the rotated pixels.

    Two paths to the same pixels must agree. One projects a point with the
    **original** model and then moves the resulting pixel the way the image
    rotation moves it. The other expresses the same point in the **rotated**
    camera's frame and projects it with the rotated model. Both go through
    OpenCV, and the frame change comes from the rotated record's *own pose*
    rather than from the constant the implementation uses — so a pose rolled
    inconsistently with the intrinsics fails this too.

    Covers both camera models (kb4 on the Index, radtan8 on the other two) and
    every turn, which is what pins ``fx``/``fy``, the principal point and the
    tangential swap all at once.
    """
    for original in load_calibration(calibration_fixture(device)):
        rolled: CalibratedCamera = rotate_camera_cw(original, quarter_turns)
        width, height = original.resolution
        assert rolled.resolution == ((width, height) if quarter_turns % 2 == 0 else (height, width))
        assert rolled.index == original.index

        points_xyz: Float64[ndarray, "n_points 3"] = points_in_front_of(original)
        cam_R_rolled: Float64[ndarray, "3 3"] = rig_R_cam(original).T @ rig_R_cam(rolled)
        # Right-multiplying the row-stacked points is cam_R_rolled.T applied to each.
        rolled_points_xyz: Float64[ndarray, "n_points 3"] = points_xyz @ cam_R_rolled

        expected_uv: Float64[ndarray, "n_points 2"] = rotate_pixels_cw(
            opencv_pixels(original, points_xyz), resolution=original.resolution, quarter_turns=quarter_turns
        )
        np.testing.assert_allclose(opencv_pixels(rolled, rolled_points_xyz), expected_uv, atol=PROJECTION_TOLERANCE_PX)


@pytest.mark.parametrize("device", ["index", "g2", "odyssey"])
def test_four_quarter_turns_give_back_the_camera_they_started_from(device: str) -> None:
    """A full revolution is the identity, so nothing in the remap is one-way lossy.

    Not exactly equal: ``(height - 1) - cy`` twice is a pair of subtractions that
    each round, so the principal point comes back within float noise rather than
    bit for bit.
    """
    for original in load_calibration(calibration_fixture(device)):
        turned: CalibratedCamera = rotate_camera_cw(original, 4)

        assert turned.index == original.index
        assert turned.resolution == original.resolution
        assert type(turned.model) is type(original.model)
        started: dict[str, float] = asdict(original.model)
        for key, value in asdict(turned.model).items():
            assert value == pytest.approx(started[key], abs=1e-12), f"cam{original.index} {key} did not come back"
        # A quaternion and its negation are the same rotation, so compare the matrices.
        np.testing.assert_allclose(rig_R_cam(turned), rig_R_cam(original), atol=1e-12)
        np.testing.assert_allclose(
            [turned.rig_pose.px, turned.rig_pose.py, turned.rig_pose.pz],
            [original.rig_pose.px, original.rig_pose.py, original.rig_pose.pz],
            atol=1e-12,
        )


def test_a_negative_turn_count_is_refused_rather_than_read_as_counter_clockwise() -> None:
    """The name says clockwise; a negative count would be an unstated second convention."""
    with pytest.raises(ValueError, match="clockwise"):
        rotate_camera_cw(UPRIGHT_PAIR[0], -1)


@pytest.mark.parametrize("device", ["index", "g2", "odyssey"])
def test_the_upright_turn_of_every_real_camera(device: str) -> None:
    """The Index and the Odyssey+ answer zero turns, so their converted output is unchanged.

    That is the load-bearing half: the upright rule is uniform, applied to every
    camera of every device, and it must be a no-op wherever the mounting is
    already upright. The G2's four cameras all answer a non-zero turn, which is
    the sideways picture this rule exists to fix.
    """
    cameras: tuple[CalibratedCamera, ...] = load_calibration(calibration_fixture(device))
    up_rig: tuple[float, float, float] = follow_frame(cameras).up

    turns: tuple[int, ...] = tuple(upright_quarter_turns(camera, up_rig) for camera in cameras)

    assert turns == UPRIGHT_TURNS[device]
    assert (device == "g2") == all(turn != 0 for turn in turns)


@pytest.mark.parametrize("device", ["index", "g2", "odyssey"])
def test_the_chosen_turn_is_the_one_that_points_image_up_at_the_headset_up(device: str) -> None:
    """The selection rule's actual claim, checked rather than the integer it returns.

    Image-up is camera ``-y`` (RDF puts ``+y`` down) carried into the rig frame,
    and after the chosen turn it must sit within a few degrees of the headset's
    own up. This holds for the cameras that need no turn as well, which is what
    makes ``0`` an answer rather than an absence of one.
    """
    cameras: tuple[CalibratedCamera, ...] = load_calibration(calibration_fixture(device))
    up_xyz: Float64[ndarray, "3"] = np.asarray(follow_frame(cameras).up, dtype=np.float64)

    for camera in cameras:
        upright: CalibratedCamera = rotate_camera_cw(camera, upright_quarter_turns(camera, follow_frame(cameras).up))
        image_up_xyz: Float64[ndarray, "3"] = rig_R_cam(upright) @ np.array([0.0, -1.0, 0.0], dtype=np.float64)
        alignment: float = float(image_up_xyz @ up_xyz)
        assert alignment > UPRIGHT_ALIGNMENT_FLOOR, f"cam{camera.index}'s upright image-up is {alignment:.4f} onto the headset up"


@pytest.mark.parametrize("device", ["index", "g2", "odyssey"])
def test_rolling_every_camera_upright_leaves_the_follow_frame_alone(device: str) -> None:
    """A roll turns the sensor about its own optical axis, so it moves neither axis of the frame.

    ``follow_frame`` reads the mean optical axis and the baseline, and a roll
    changes camera ``+z`` not at all and the translations not at all — so the
    device's forward and up are the same before and after, and the follow camera
    and world axes of an upright conversion are unchanged.
    """
    cameras: tuple[CalibratedCamera, ...] = load_calibration(calibration_fixture(device))
    frame: FollowFrame = follow_frame(cameras)

    upright: tuple[CalibratedCamera, ...] = tuple(rotate_camera_cw(camera, upright_quarter_turns(camera, frame.up)) for camera in cameras)

    rolled_frame: FollowFrame = follow_frame(upright)
    np.testing.assert_allclose(rolled_frame.forward, frame.forward, atol=1e-9)
    np.testing.assert_allclose(rolled_frame.up, frame.up, atol=1e-9)


def test_an_up_axis_along_the_optical_axis_names_no_turn() -> None:
    """A camera aimed straight up has no image direction that points up; four turns tie at zero."""
    with pytest.raises(ValueError, match="no image direction that points up"):
        upright_quarter_turns(UPRIGHT_PAIR[0], (0.0, 0.0, 1.0))


def test_a_camera_rolled_half_a_quarter_turn_is_refused_as_ambiguous() -> None:
    """At 45 degrees two turns are equally good, and picking either would be a coin toss."""
    diagonal: CalibratedCamera = replace(
        UPRIGHT_PAIR[0], rig_pose=replace(UPRIGHT_PAIR[0].rig_pose, qz=float(np.sin(np.pi / 8)), qw=float(np.cos(np.pi / 8)))
    )
    with pytest.raises(ValueError, match="ambiguous|apart"):
        upright_quarter_turns(diagonal, (0.0, -1.0, 0.0))


def test_an_up_axis_that_is_not_a_direction_is_refused() -> None:
    with pytest.raises(ValueError, match="direction"):
        upright_quarter_turns(UPRIGHT_PAIR[0], (0.0, 0.0, 0.0))
