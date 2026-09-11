"""The RoboCap rig's own calibration, and the probe's cross-check against it.

The recording stores 1920x1080 frames and the C++ ran 640x360, so the feed's
downscale has to reproduce basalt's own converted calibration digit for digit,
and the probe refuses a calibration file that describes another rig. The matcher,
the inertial pairing and the camera subset are in ``test_frameset_matching``;
the feed's MSD path is in ``test_catalog_feed``.
"""

import json
from dataclasses import replace

import numpy as np
import pytest
from jaxtyping import Float64
from numpy import ndarray
from scipy.spatial.transform import Rotation

from slam_rs import _core
from slam_rs.catalog_feed import (
    CHILD_FROM_PARENT,
    CameraCalib,
    CameraStatics,
    CatalogSegment,
    RigProfile,
    camera_calib,
    open_segment,
    scale_principal_point,
)
from slam_rs.reference import ImuParameters, ReferenceManifest, RobocapSession
from slam_rs.tracking import check_calibration_matches_recording

# The four fed cameras' native intrinsics exactly as the recording carries them,
# in the C++'s order. float32 statics, so the digits stop where float32 does.
ROBOCAP_INTRINSICS: tuple[tuple[float, float, float, float], ...] = (
    (625.53564453125, 626.1504516601562, 999.440185546875, 539.0486450195312),
    (636.4360961914062, 634.7124633789062, 956.2034301757812, 525.4381103515625),
    (630.6917724609375, 628.777587890625, 946.6721801757812, 539.53125),
    (612.0897827148438, 608.9950561523438, 967.4310913085938, 551.28759765625),
)
ROBOCAP_KB4: tuple[tuple[float, float, float, float], ...] = (
    (0.05310669541358948, 0.017954021692276, -0.00536160776391625, 0.00056962034432217),
    (0.06166616827249527, -0.0210909266024828, 0.0371633879840374, -0.013518619351089),
    (0.07725944370031357, -0.06258341670036316, 0.08006518334150314, -0.02879575826227665),
    (0.07467304170131683, 0.0116525636985898, -0.00467674527317286, 0.00134803401306272),
)
"""Each fed camera's KB4 coefficients as the recording carries them; no downscale touches them."""
ROBOCAP_IMU_T_CAM: tuple[tuple[float, ...], ...] = (
    (0.11399778805101912, -0.08575139322056138, 0.01508625718898565)
    + (0.02437430433928967, -0.10791313648223877, 0.9938614964485168)
    + (0.9846004247665405, 0.17474322021007538, -0.00517362076789141)
    + (-0.17311225831508636, 0.9786825776100159, 0.11051056534051895),
    (0.03477445441432928, 0.00997360635818159, -0.0064599738573613)
    + (-0.9994690418243408, 0.00405994756147265, 0.03232910111546516)
    + (0.03183514997363091, -0.08968588709831238, 0.9954611659049988)
    + (0.0069409841671586, 0.9959618449211121, 0.089509017765522),
    (-0.05141922995087802, 0.0107788254374653, -0.00630683517106816)
    + (-0.999871015548706, -0.01601020060479641, -0.0012666096445173)
    + (-0.00030042274738662, -0.06020709872245789, 0.9981858730316162)
    + (-0.01605741493403912, 0.998057484626770, 0.06019452586770058),
    (-0.12514879125479297, -0.09818661029737424, 0.00357639240882723)
    + (0.02767287567257881, 0.09491033107042313, -0.9951010942459106)
    + (-0.9824655652046204, 0.1861988753080368, -0.00956229493021965)
    + (0.18437914550304413, 0.9779171943664551, 0.09839878976345062),
)
"""Each fed camera's ``imu_T_cam`` as the recording gives it: translation, then the rotation row by row.

Frozen from session 15 the way the intrinsics above are, so the cross-check is
read against the rig rather than against the file it is checking.
"""


def imu_T_cam(camera: int) -> Float64[ndarray, "4 4"]:
    """One fed camera's pose in the IMU frame, from :data:`ROBOCAP_IMU_T_CAM`."""
    pose: Float64[ndarray, "4 4"] = np.eye(4)
    pose[:3, 3] = ROBOCAP_IMU_T_CAM[camera][:3]
    pose[:3, :3] = np.array(ROBOCAP_IMU_T_CAM[camera][3:]).reshape(3, 3)
    return pose


def robocap_statics(
    fx: float = 625.53564453125,
    fy: float = 626.1504516601562,
    cx: float = 999.440185546875,
    cy: float = 539.0486450195312,
    camera: int = 0,
    distortion: tuple[float, float, float, float] | None = None,
    turn_deg: float = 0.0,
    shift_m: float = 0.0,
) -> CameraStatics:
    """One RoboCap camera's statics as the recording carries them: native 1920x1080, KB4.

    Args:
        fx: Focal length along image x, native pixels.
        fy: Focal length along image y, native pixels.
        cx: Principal point x, native pixels.
        cy: Principal point y, native pixels.
        camera: Which fed camera's frozen distortion and extrinsics to carry.
        distortion: KB4 coefficients to store instead of that camera's own.
        turn_deg: Rotate the camera this far about x, for the refusal cases.
        shift_m: Move the camera this far along x, for the refusal cases.

    Returns:
        The statics as :func:`slam_rs.catalog_feed.read_camera_statics` reads them:
        column-major matrices and the ``ChildFromParent`` relation, so the stored
        transform is ``cam_T_imu`` and the feed inverts it.
    """
    drifted: Float64[ndarray, "4 4"] = imu_T_cam(camera)
    drifted[:3, :3] = Rotation.from_euler("x", turn_deg, degrees=True).as_matrix() @ drifted[:3, :3]
    drifted[0, 3] += shift_m
    cam_T_imu: Float64[ndarray, "4 4"] = np.linalg.inv(drifted)
    return CameraStatics(
        distortion_model="kannala_brandt",
        distortion_coefficients=np.array([*(distortion if distortion is not None else ROBOCAP_KB4[camera]), 0.0, 0.0, 0.0, 0.0]),
        image_from_camera=np.array([fx, 0.0, 0.0, 0.0, fy, 0.0, cx, cy, 1.0]),
        resolution_wh=np.array([1920.0, 1080.0]),
        transform_mat3x3=cam_T_imu[:3, :3].reshape(-1, order="F"),
        transform_translation=cam_T_imu[:3, 3],
        transform_relation=CHILD_FROM_PARENT,
        distortion_valid_radius=None,
    )
# --- the downscale, on both the frames and the intrinsics --------------------


def test_downscaling_the_recording_reproduces_basalts_own_calibration(manifest: ReferenceManifest) -> None:
    """The recording's native statics at downscale 3 are the C++'s own 640x360 file.

    The two come from one Kalibr tree by different routes — the fork's converter
    wrote the file, the ``dataforge`` conversion wrote the statics — so this is
    the check that the port is fed the rig the C++ was fed.
    """
    basalt: _core.Calibration = _core.Calibration.from_json((manifest.package_root / manifest.robocap.calibration).read_text())
    assert list(basalt.resolution) == [(640, 360)] * 4

    calib = camera_calib(0, robocap_statics(), manifest.robocap.downscale)
    assert (calib.width, calib.height) == (640, 360)
    written = json.loads(basalt.to_json())["value0"]["intrinsics"][0]["intrinsics"]
    assert calib.fx == pytest.approx(written["fx"], abs=1e-3)
    assert calib.cx == pytest.approx(written["cx"], abs=1e-3)
    assert calib.cy == pytest.approx(written["cy"], abs=1e-3)
    # KB4's coefficients are resolution-invariant, so they are not scaled.
    assert calib.distortion.tolist() == pytest.approx([written[f"k{index}"] for index in (1, 2, 3, 4)], abs=1e-6)


def test_the_principal_point_scales_by_the_pixel_centre() -> None:
    """`(c + 0.5) / d - 0.5`, not `c / d`: the converter's convention, and half a pixel apart."""
    assert scale_principal_point(999.4402, 2) == pytest.approx(499.4701)
    assert scale_principal_point(999.4402, 3) == pytest.approx(332.8134, abs=1e-4)
    assert scale_principal_point(999.4402, 1) == 999.4402


def test_a_downscale_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="downscale must be at least 1"):
        camera_calib(0, robocap_statics(), 0)

def test_a_downscale_that_leaves_no_frame_is_refused() -> None:
    """A downscale past the frame's own width gives a camera zero pixels wide.

    ``decode_gray`` clamps its target to one pixel, so the calibration and the
    frames would disagree about the size of the image the estimator is given.
    """
    with pytest.raises(ValueError, match=r"downscale 2000 leaves nothing of the 1920x1080 frame"):
        camera_calib(0, robocap_statics(), 2000)

def robocap_rig(downscale: int = 3) -> tuple[CameraCalib, ...]:
    """The four fed cameras as the recording gives them, scaled by ``downscale``."""
    return tuple(camera_calib(number, robocap_statics(*values, camera=number), downscale) for number, values in enumerate(ROBOCAP_INTRINSICS))


def test_the_probe_refuses_a_calibration_that_is_not_the_recordings_rig(manifest: ReferenceManifest) -> None:
    """A calibration for another resolution, another lens or another rig geometry stops the run."""
    basalt: _core.Calibration = _core.Calibration.from_json((manifest.package_root / manifest.robocap.calibration).read_text())
    at_three: tuple[CameraCalib, ...] = robocap_rig()
    check_calibration_matches_recording(basalt, at_three, manifest.robocap.imu, 3)

    with pytest.raises(ValueError, match=r"basalt's calibration is \[\(640, 360\).*the feed decodes \[\(960, 540\)"):
        check_calibration_matches_recording(basalt, robocap_rig(downscale=2), manifest.robocap.imu, 2)

    with pytest.raises(ValueError, match="the feed selected 3"):
        check_calibration_matches_recording(basalt, at_three[:3], manifest.robocap.imu, 3)

    moved = ROBOCAP_INTRINSICS[0][:2] + (1200.0, ROBOCAP_INTRINSICS[0][3])
    shifted = (camera_calib(0, robocap_statics(*moved), 3), *at_three[1:])
    with pytest.raises(ValueError, match="cam 0: basalt's cx is 332.81.*the recording gives 399.66"):
        check_calibration_matches_recording(basalt, shifted, manifest.robocap.imu, 3)


def test_the_probe_refuses_a_lens_or_a_rig_geometry_that_drifted(manifest: ReferenceManifest) -> None:
    """The distortion and the extrinsics are compared too: a drifted route is a refusal, not a bias.

    A conversion that moved a coefficient or a camera would otherwise show up
    only as a few centimetres of trajectory error nobody could attribute to it.
    """
    basalt: _core.Calibration = _core.Calibration.from_json((manifest.package_root / manifest.robocap.calibration).read_text())
    at_three: tuple[CameraCalib, ...] = robocap_rig()

    bent = ROBOCAP_KB4[0][:1] + (ROBOCAP_KB4[0][1] + 1e-4,) + ROBOCAP_KB4[0][2:]
    with pytest.raises(ValueError, match="cam 0: basalt's k2 is"):
        lens = (camera_calib(0, robocap_statics(*ROBOCAP_INTRINSICS[0], distortion=bent), 3), *at_three[1:])
        check_calibration_matches_recording(basalt, lens, manifest.robocap.imu, 3)

    with pytest.raises(ValueError, match="cam 0: basalt places it 1.000 mm from where the recording does"):
        shifted = (camera_calib(0, robocap_statics(*ROBOCAP_INTRINSICS[0], shift_m=1e-3), 3), *at_three[1:])
        check_calibration_matches_recording(basalt, shifted, manifest.robocap.imu, 3)

    with pytest.raises(ValueError, match="cam 0: basalt turns it 0.1000 deg from where the recording does"):
        turned = (camera_calib(0, robocap_statics(*ROBOCAP_INTRINSICS[0], turn_deg=0.1), 3), *at_three[1:])
        check_calibration_matches_recording(basalt, turned, manifest.robocap.imu, 3)


def test_the_probe_refuses_a_calibration_whose_imu_is_not_the_manifests(manifest: ReferenceManifest) -> None:
    """The estimator reads the file's noise model and the feed reads the manifest's: they must be one model."""
    basalt: _core.Calibration = _core.Calibration.from_json((manifest.package_root / manifest.robocap.calibration).read_text())
    at_three: tuple[CameraCalib, ...] = robocap_rig()

    louder: ImuParameters = replace(manifest.robocap.imu, gyro_noise_std=2.0 * manifest.robocap.imu.gyro_noise_std)
    with pytest.raises(ValueError, match="basalt's gyro_noise_std is"):
        check_calibration_matches_recording(basalt, at_three, louder, 3)

    # The offset belongs to the feed, which adds it to the frames; a file that
    # carried it too would move every frameset twice.
    document: dict = json.loads((manifest.package_root / manifest.robocap.calibration).read_text())
    document["value0"]["cam_time_offset_ns"] = manifest.robocap.imu.cam_time_offset_ns
    with pytest.raises(ValueError, match="carries cam_time_offset_ns 14902432"):
        check_calibration_matches_recording(_core.Calibration.from_json(json.dumps(document)), at_three, manifest.robocap.imu, 3)
# --- the real rig, behind `slow` ---------------------------------------------






@pytest.mark.slow
def test_the_feed_opens_the_real_robocap_rig(manifest: ReferenceManifest) -> None:
    """Four of six cameras, 640x360, framesets on the C++'s own clock, both channels paired.

    The one test that proves the whole read rather than its pieces: it needs the
    NAS, so it sits behind ``slow`` like every other reference read.
    """
    session: RobocapSession = manifest.robocap.session("s00000015")
    with open_segment(CatalogSegment(manifest.catalog_url, "robocap", session.segment_id), manifest.robocap.imu, profile=RigProfile.from_robocap(manifest.robocap)) as feed:
        assert feed.camera_positions == (4, 0, 1, 5)
        assert feed.rig_cameras == 6
        # The feed reads its rig knobs off the profile it was given, so what the
        # manifest says and what the feed does are one statement.
        assert feed.profile == RigProfile.from_robocap(manifest.robocap)
        assert [(camera.width, camera.height) for camera in feed.cameras] == [(640, 360)] * 4
        assert all(camera.model == "kb4" for camera in feed.cameras)
        assert all(len(camera.distortion) == 4 for camera in feed.cameras)
        # The frameset count and clock are basalt's own, to the nanosecond.
        assert len(feed.frame_t_ns) == 1588

        frameset = next(feed.framesets())
        assert int(frameset.t_ns) == 70_258_640_500 + 14_902_432
        assert not feed.has_ground_truth
        assert [image.shape for image in frameset.images] == [(360, 640)] * 4
        assert len(frameset.imu) > 0
        assert frameset.ground_truth is None
        # video_time is the device clock here, so the export adds nothing to it.
        assert feed.export_offset_ns == 0
