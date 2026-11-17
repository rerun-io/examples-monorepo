"""Convert an Insta360 dual-fisheye ``.insv`` clip into one exoego:v2 ``.rrd`` the slam-rs feed reads.

The two HEVC lens tracks are remuxed, not re-encoded: each sample becomes an Annex B access unit with the
parameter sets in front of every IRAP (``slam_rs._core.InsvClip.read_samples``). ``cam_00`` and ``cam_01`` are lens 0
and lens 1, whichever track holds them. Frame times come from the exposure record, IMU times from the trailer, both
on the camera's own clock; ``video_time`` counts from the earliest of them.

The cameras carry a four-coefficient Kannala-Brandt fit of the factory (V6) radial curve. The factory model's
tangential and prism terms do not fit KB4; the raw factory strings and the window crop are kept in the ``insta360``
recording property, and slam-rs rebuilds the full model from them in Rust for the panorama stitch.

The X4 Air device profile (:data:`X4_AIR`) applies the calibration a gyro-visual epipolar fit measured on the clip
VID_20261116_125200_00_003 (2026-11-16, no board):

- Rig frame = IMU frame (``reference = 'imu_00'``). Each lens camera is ``T_imu_lens = T_imu_P * T_P_lens``, with
  ``T_P_lens`` the factory rotation and centre in the panorama frame ``P`` (``kornia_staging_io::insta360``:
  ``pano_from_lens_rotation``, ``pano_centre_m``) and ``T_imu_P = (R_imu_P, R_imu_P * lever_arm)``.
- IMU times are the trailer's, without the vendor's 1.6 ms ``gyro_timestamp`` subtraction (the probe measured on
  that clock). Each frame is stamped at its effective time on the IMU clock,
  ``t_i = t_exposure_record_i - 0.44 * shutter_i - 0.1 ms + extra``: the mid-exposure time of the lens image's
  MIDDLE row. A rolling-shutter row ``v`` of a 1920-row lens image is read at ``t + readout * (v - 959.5) / 1920``.

The IMU noise model is a placeholder, marked as such in its ``source``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
from dataforge import schema
from dataforge.logging_toolkit import ImuChannel, log_camera_node, log_camera_source, log_imu, log_rig_node, time_column
from jaxtyping import Float64, Int64
from numpy import ndarray
from simplecv.camera_parameters import Extrinsics, Fisheye62Parameters, Intrinsics, KannalaBrandtDistortion
from simplecv.imu_calibration import ImuCalibration

from slam_rs import _core

PLACEHOLDER_IMU: ImuCalibration = ImuCalibration(
    gyro_noise_density=0.0028,
    accel_noise_density=0.016,
    gyro_bias_random_walk=0.0001,
    accel_bias_random_walk=0.001,
    rate_hz=1000.0,
    source="placeholder: consumer MEMS (TDK ICM-40609 class, values not measured on this unit)",
)
"""Noise model for the X4 Air IMU until one is measured; the slam-rs feed refuses an IMU without all five numbers."""


@dataclass(frozen=True, slots=True)
class DeviceProfile:
    """Calibration of one Insta360 body that the file does not carry, with where each number comes from."""

    name: str
    """Camera type the profile is for (``InsvClip.camera_type``)."""
    pano_from_imu_measured: Float64[ndarray, "3 3"]
    """``R_P_imu`` as fitted (rows: P axes in IMU coordinates); orthonormalised before use."""
    frame_time_shutter_slope: float
    """``a`` of the per-frame offset ``t_d_i = a * shutter_i + b`` (shutter = exposure record 4 duration, s)."""
    frame_time_constant_s: float
    """``b`` of the same model, seconds."""
    readout_s: float
    """Rolling-shutter readout of one lens image, seconds; rows read top to bottom in both lenses."""
    provenance: str
    """Where the numbers were measured."""


X4_AIR: DeviceProfile = DeviceProfile(
    name="Insta360 X4 Air",
    # R_rig_imu (P <- imu), epipolar joint fit of segments 40-60 s and 190-210 s; 0.13 deg from the
    # permutation x_P = -z_imu, y_P = +x_imu, z_P = -y_imu; the two segments agree within 0.2 deg.
    pano_from_imu_measured=np.array(
        [[-0.00092, 0.00149, -1.00000], [1.00000, -0.00138, -0.00092], [-0.00138, -1.00000, -0.00149]], dtype=np.float64
    ),
    # t_d = -0.435 * shutter - 0.13 ms (joint epipolar fit with the shutter term, rms 0.035 rad/s), rounded here
    # to -0.44 * shutter - 0.1 ms. gyro(t_frame + t_d) matches the frame.
    frame_time_shutter_slope=-0.44,
    frame_time_constant_s=-0.1e-3,
    # Metadata rolling_shutter 15.883 ms; free fits 13-21 ms (lens 1), 11-31 ms (lens 0); rows top to bottom.
    readout_s=15.883e-3,
    provenance="gyro-visual epipolar fit on the clip VID_20261116_125200_00_003",
)
"""The X4 Air profile measured on the clip VID_20261116_125200_00_003."""

DEVICE_PROFILES: dict[str, DeviceProfile] = {X4_AIR.name: X4_AIR}
"""Device profiles by ``InsvClip.camera_type``; a clip from any other body is refused rather than guessed."""


def orthonormal(matrix: Float64[ndarray, "3 3"]) -> Float64[ndarray, "3 3"]:
    """The rotation nearest to ``matrix`` (SVD polar factor, determinant +1)."""
    u, _, vt = np.linalg.svd(matrix)
    rotation: Float64[ndarray, "3 3"] = u @ vt
    if np.linalg.det(rotation) < 0:
        rotation = u @ np.diag([1.0, 1.0, -1.0]) @ vt
    return rotation


def nearest_permutation(rotation: Float64[ndarray, "3 3"]) -> Float64[ndarray, "3 3"]:
    """The signed axis permutation nearest to ``rotation``, row by row."""
    permutation: Float64[ndarray, "3 3"] = np.zeros((3, 3))
    for row in range(3):
        column: int = int(np.argmax(np.abs(rotation[row])))
        permutation[row, column] = np.sign(rotation[row, column])
    return permutation


def angle_to_nearest_permutation_deg(rotation: Float64[ndarray, "3 3"]) -> float:
    """Angle in degrees between ``rotation`` and the signed axis permutation nearest to it."""
    permutation: Float64[ndarray, "3 3"] = nearest_permutation(rotation)
    cosine: float = (float(np.trace(permutation.T @ rotation)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


@dataclass
class Config:
    """Convert one ``.insv`` clip to an exoego:v2 recording."""

    insv: Path
    """The Insta360 clip."""
    output: Path | None = None
    """Destination ``.rrd``; defaults to the clip's path with the ``.rrd`` suffix."""
    max_frames: int | None = None
    """Convert only the first frames (IMU trimmed to match); None converts the whole clip."""
    fit_max_deg: float = 97.0
    """Field angle up to which the KB4 fit follows the factory radial curve; the frame edges reach 96-99 degrees."""
    batch_frames: int = 300
    """Samples read and sent per call, so the clip is never held in memory whole."""
    lever_arm_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Vector from the IMU to the lens midpoint, metres, in the panorama frame P (not measured; +-3 cm ablations are still to run)."""
    extra_frame_offset_ms: float = 0.0
    """Constant added to every frame time on top of the profile's per-frame model, ms (set by a later calibration scan)."""


@dataclass(frozen=True, slots=True)
class Kb4Fit:
    """Kannala-Brandt ``r = f (t + k1 t^3 + k2 t^5 + k3 t^7 + k4 t^9)`` fitted to a factory radial curve."""

    focal_px: float
    """Focal length, pixels per radian."""
    k: Float64[ndarray, " 4"]
    """``k1..k4``."""
    max_residual_px: float
    """Largest radius error over the fitted range, pixels."""


def kb4_fit(theta: Float64[ndarray, " n"], radius_px: Float64[ndarray, " n"]) -> Kb4Fit:
    """Least-squares KB4 fit of a radial curve sampled at field angles ``theta`` (radians).

    The model is linear in ``(f, f k1, .., f k4)``, so one linear solve gives the fit.
    """
    basis: Float64[ndarray, "n 5"] = np.stack([theta ** (2 * i + 1) for i in range(5)], axis=1)
    coefficients: Float64[ndarray, " 5"] = np.linalg.lstsq(basis, radius_px, rcond=None)[0]
    residual: float = float(np.abs(basis @ coefficients - radius_px).max())
    return Kb4Fit(focal_px=float(coefficients[0]), k=coefficients[1:] / coefficients[0], max_residual_px=residual)


def imu_from_lens(lens: _core.InsvLens, imu_from_pano: Float64[ndarray, "3 3"], lever_arm_m: Float64[ndarray, " 3"]) -> Float64[ndarray, "4 4"]:
    """``T_imu_lens = T_imu_P * T_P_lens``, with ``T_imu_P = (R_imu_P, R_imu_P * lever_arm)``."""
    pose: Float64[ndarray, "4 4"] = np.eye(4)
    pose[:3, :3] = imu_from_pano @ np.asarray(lens.pano_from_lens_rotation, dtype=np.float64)
    pose[:3, 3] = imu_from_pano @ (np.asarray(lens.pano_centre_m, dtype=np.float64) + lever_arm_m)
    return pose


def lens_camera(lens: _core.InsvLens, fit: Kb4Fit, name: str, width: int, height: int, imu_T_lens: Float64[ndarray, "4 4"]) -> Fisheye62Parameters:
    """The lens as a KB4 camera posed in the rig (= IMU) frame; the stored transform is ``lens_T_imu``."""
    lens_R_imu: Float64[ndarray, "3 3"] = imu_T_lens[:3, :3].T
    return Fisheye62Parameters(
        name=name,
        extrinsics=Extrinsics(cam_R_world=lens_R_imu, cam_t_world=-lens_R_imu @ imu_T_lens[:3, 3]),
        intrinsics=Intrinsics.from_focal_principal_point(
            camera_conventions="RDF", fl_x=fit.focal_px, fl_y=fit.focal_px, cx=lens.cx, cy=lens.cy, width=width, height=height
        ),
        distortion=KannalaBrandtDistortion(k1=float(fit.k[0]), k2=float(fit.k[1]), k3=float(fit.k[2]), k4=float(fit.k[3])),
    )


def main(config: Config) -> None:
    """Write the recording and print the KB4 fit residuals."""
    if not config.insv.is_file():
        raise FileNotFoundError(f"{config.insv}: no such clip")
    output: Path = config.output or config.insv.with_suffix(".rrd")
    clip: _core.InsvClip = _core.InsvClip(str(config.insv))
    if clip.camera_type not in DEVICE_PROFILES:
        raise ValueError(f"{config.insv}: no device profile for a {clip.camera_type!r} (profiles: {sorted(DEVICE_PROFILES)})")
    profile: DeviceProfile = DEVICE_PROFILES[clip.camera_type]
    lenses: list[_core.InsvLens] = clip.lenses_v6
    factory: dict[str, str] = dict(clip.factory_strings)
    if len(lenses) != 2 or "v6" not in factory:
        raise ValueError(f"{config.insv}: no V6 factory lens string; the panorama stitch is built from it")
    frames: int = clip.num_frames if config.max_frames is None else min(config.max_frames, clip.num_frames)
    frame_t_us, exposure_vin0 = clip.frame_exposures(False)
    vin1_t_us, exposure_vin1 = clip.frame_exposures(True)
    imu_t_us, accel, gyro = clip.imu()
    # IMU on the trailer clock as stored: the gyro-visual fit measured the time offset WITHOUT the vendor's gyro_timestamp shift.
    imu_ns: Int64[ndarray, " n_imu"] = imu_t_us * 1000
    # Effective frame time on the IMU clock (the middle row's mid-exposure): the exposure-record stamp plus the
    # profile's per-frame offset, from the record-4 shutter time, plus the constant a calibration scan may add.
    shift_s: Float64[ndarray, " n_frames"] = (
        profile.frame_time_shutter_slope * exposure_vin0[:frames] + profile.frame_time_constant_s + config.extra_frame_offset_ms * 1e-3
    )
    frame_ns: Int64[ndarray, " n_frames"] = frame_t_us[:frames] * 1000 + np.round(shift_s * 1e9).astype(np.int64)
    if np.any(np.diff(frame_ns) <= 0):
        raise ValueError(f"{config.insv}: the per-frame time model makes frame times non-increasing")
    pano_from_imu: Float64[ndarray, "3 3"] = orthonormal(profile.pano_from_imu_measured)
    imu_from_pano: Float64[ndarray, "3 3"] = pano_from_imu.T
    lever_arm: Float64[ndarray, " 3"] = np.asarray(config.lever_arm_m, dtype=np.float64)
    if config.max_frames is not None:
        keep = imu_ns <= frame_ns[-1] + 1_000_000_000
        imu_ns, accel, gyro = imu_ns[keep], accel[keep], gyro[keep]
    start_ns: int = int(min(imu_ns[0], frame_ns[0]))
    width, height = clip.resolution
    theta: Float64[ndarray, " n"] = np.linspace(0.0, np.deg2rad(config.fit_max_deg), 2000)
    fits: list[Kb4Fit] = [kb4_fit(theta, lens.radius_px(theta.tolist())) for lens in lenses]
    crop: tuple[int, int, int, int, int, int] | None = clip.window_crop

    with rr.RecordingStream("slam-rs-insv", recording_id=config.insv.stem) as rec:
        rec.save(output)
        rec.send_recording_name(config.insv.stem)
        rec.send_property("capture", rr.AnyValues(schema=schema.DATAFORGE_SCHEMA_VERSION, start_time_ns=start_ns, num_frames=frames, num_cameras=2))
        # Any: AnyValues takes its keys as **kwargs beside its own typed flags.
        insta360: dict[str, Any] = dict(
            camera_type=clip.camera_type,
            firmware=clip.firmware,
            first_frame_timestamp_us=clip.first_frame_timestamp_us,
            rolling_shutter_ms=clip.rolling_shutter_ms,
            gyro_timestamp_ms=clip.gyro_timestamp_ms,
            lens_track=list(clip.lens_track),
            kb4_fit_max_deg=config.fit_max_deg,
            kb4_fit_max_residual_px=[fit.max_residual_px for fit in fits],
        )
        insta360.update({f"offset_{version}": text for version, text in clip.factory_strings})
        insta360.update(
            lens_string_version="v6",
            window_crop=list(crop) if crop is not None else [],
            window_crop_order="src_w src_h dst_w dst_h offset_x offset_y",
            profile=profile.name,
            profile_provenance=profile.provenance,
            pano_from_imu_measured=profile.pano_from_imu_measured.ravel().tolist(),
            pano_from_imu=pano_from_imu.ravel().tolist(),
            pano_from_imu_orthonormalisation_deg=float(np.degrees(np.linalg.norm(pano_from_imu - profile.pano_from_imu_measured, 2))),
            pano_from_imu_to_nearest_permutation_deg=angle_to_nearest_permutation_deg(pano_from_imu),
            lever_arm_m=list(config.lever_arm_m),
            rolling_shutter_readout_s=profile.readout_s,
            rolling_shutter_rows="top to bottom, both lenses",
            rolling_shutter_row_time="t + readout_s * (v - 959.5) / 1920 for row v of the 1920-row lens image",
            frame_time_convention="mid-exposure time of the middle row of the lens image, on the IMU clock",
            frame_time_model="t_exposure_record4 + slope * shutter_record4 + constant + extra",
            frame_time_shutter_slope=profile.frame_time_shutter_slope,
            frame_time_constant_s=profile.frame_time_constant_s,
            frame_time_extra_offset_s=config.extra_frame_offset_ms * 1e-3,
            imu_time="trailer clock, gyro_timestamp NOT subtracted",
        )
        rec.send_property("insta360", rr.AnyValues(**insta360))
        log_rig_node(rec, 0, reference="imu_00", num_cameras=2, name=clip.camera_type)
        log_imu(
            rec, 0, 0,
            gyro=ImuChannel(times_ns=imu_ns - start_ns, values_xyz=gyro),
            accel=ImuChannel(times_ns=imu_ns - start_ns, values_xyz=accel),
            name="imu",
        )
        rec.log(
            schema.imu_path(0, 0),
            PLACEHOLDER_IMU,
            rr.AnyValues(
                applied_time_shift_ns=0,
                time_shift_source="none on the IMU; frames carry the per-frame model in property insta360:frame_time_* (exposure: frame_time_shift_s)",
            ),
            static=True,
        )
        # Per-frame exposure time of each sensor record, and how far the second sensor's stamp is from the first's.
        rr.send_columns(
            f"{schema.rig_path(0)}/exposure",
            indexes=[time_column(frame_ns - start_ns)],
            columns=rr.Scalars.columns(
                scalars=np.stack([exposure_vin0[:frames], exposure_vin1[:frames], (vin1_t_us[:frames] - frame_t_us[:frames]) * 1e-6, shift_s], axis=1)
            ),
            recording=rec,
        )
        rec.log(
            f"{schema.rig_path(0)}/exposure",
            rr.SeriesLines(names=["exposure_vin0_s", "exposure_vin1_s", "vin1_minus_vin0_s", "frame_time_shift_s"]),
            static=True,
        )
        for cam, (lens, fit) in enumerate(zip(lenses, fits, strict=True)):
            track: int = clip.lens_track[cam]
            name: str = f"lens{cam}"
            camera: Fisheye62Parameters = lens_camera(lens, fit, name, width, height, imu_from_lens(lens, imu_from_pano, lever_arm))
            log_camera_node(rec, 0, cam, camera, name=name, kind="rgb", image_plane_distance=0.05, camera_model="kb4")
            log_camera_source(rec, 0, cam, name=name, source_width=width, source_height=height, video_codec="hevc", stream_id=f"track{track}")
            video: str = schema.video_path(0, cam)
            rec.log(video, rr.VideoStream(codec=rr.VideoCodec.H265), static=True)
            for first in range(0, frames, config.batch_frames):
                last: int = min(first + config.batch_frames, frames)
                samples, keyframes = clip.read_samples(track, first, last)
                rr.send_columns(
                    video,
                    indexes=[time_column(frame_ns[first:last] - start_ns)],
                    columns=rr.VideoStream.columns(sample=samples, is_keyframe=keyframes.tolist()),
                    recording=rec,
                )
                # Bound memory to one batch: without a flush the sink holds most of the clip.
                rec.flush()
            print(f"cam_{cam:02d} = lens {cam} (track {track}): KB4 f {fit.focal_px:.3f} px, k {np.round(fit.k, 6)}, max residual {fit.max_residual_px:.4f} px")
    print(
        f"R_P_imu: orthonormalised (moved {np.degrees(np.linalg.norm(pano_from_imu - profile.pano_from_imu_measured, 2)):.4f} deg), "
        f"{angle_to_nearest_permutation_deg(pano_from_imu):.3f} deg from the nearest axis permutation; "
        f"frame shift {shift_s.min() * 1e3:.2f}..{shift_s.max() * 1e3:.2f} ms"
    )
    print(f"Wrote {frames} frames per lens and {imu_ns.size} IMU samples to {output}")
