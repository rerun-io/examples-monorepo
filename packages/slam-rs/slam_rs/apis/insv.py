"""Convert an Insta360 dual-fisheye ``.insv`` clip into one exoego:v2 ``.rrd`` the slam-rs feed reads.

The two HEVC lens tracks are remuxed, not re-encoded: each sample becomes an Annex B access unit with the
parameter sets in front of every IRAP (``slam_rs._core.InsvClip.read_samples``). ``cam_00`` and ``cam_01`` are lens 0
and lens 1, whichever track holds them. Frame times come from the exposure record, IMU times from the trailer, both
on the camera's own clock; ``video_time`` counts from the earliest of them.

The cameras carry a four-coefficient Kannala-Brandt fit of the factory radial curve. The factory model's tangential
and prism terms do not fit KB4; the raw factory strings are kept in the ``insta360`` recording property for the
consumers that need the full model. The IMU frame is taken as the rig frame (identity) until a calibration exists,
and the IMU noise model is a placeholder, marked as such in its ``source``.
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


def lens_camera(lens: _core.InsvLens, fit: Kb4Fit, name: str, width: int, height: int) -> Fisheye62Parameters:
    """The lens as a KB4 camera posed in the rig frame (``cam_T_rig`` from the factory rig)."""
    return Fisheye62Parameters(
        name=name,
        extrinsics=Extrinsics(
            cam_R_world=np.asarray(lens.lens_from_rig_rotation, dtype=np.float64),
            cam_t_world=np.asarray(lens.lens_from_rig_translation_m, dtype=np.float64),
        ),
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
    frames: int = clip.num_frames if config.max_frames is None else min(config.max_frames, clip.num_frames)
    frame_t_us, exposure_vin0 = clip.frame_exposures(False)
    vin1_t_us, exposure_vin1 = clip.frame_exposures(True)
    imu_t_us, accel, gyro = clip.imu()
    # The vendor's tools subtract gyro_timestamp from IMU times only; the feed undoes a shift stated here.
    shift_ns: int = -round((clip.gyro_timestamp_ms or 0.0) * 1e6)
    frame_ns: Int64[ndarray, " n_frames"] = frame_t_us[:frames] * 1000
    imu_ns: Int64[ndarray, " n_imu"] = imu_t_us * 1000 + shift_ns
    if config.max_frames is not None:
        keep = imu_ns <= frame_ns[-1] + 1_000_000_000
        imu_ns, accel, gyro = imu_ns[keep], accel[keep], gyro[keep]
    start_ns: int = int(min(imu_ns[0], frame_ns[0]))
    width, height = clip.resolution
    theta: Float64[ndarray, " n"] = np.linspace(0.0, np.deg2rad(config.fit_max_deg), 2000)
    fits: list[Kb4Fit] = [kb4_fit(theta, lens.radius_px(theta.tolist())) for lens in clip.lenses]

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
            rr.AnyValues(applied_time_shift_ns=shift_ns, time_shift_source="insta360 metadata gyro_timestamp (field 28), subtracted from IMU times"),
            static=True,
        )
        # Per-frame exposure time of each sensor record, and how far the second sensor's stamp is from the first's.
        rr.send_columns(
            f"{schema.rig_path(0)}/exposure",
            indexes=[time_column(frame_ns - start_ns)],
            columns=rr.Scalars.columns(scalars=np.stack([exposure_vin0[:frames], exposure_vin1[:frames], (vin1_t_us[:frames] - frame_t_us[:frames]) * 1e-6], axis=1)),
            recording=rec,
        )
        rec.log(f"{schema.rig_path(0)}/exposure", rr.SeriesLines(names=["exposure_vin0_s", "exposure_vin1_s", "vin1_minus_vin0_s"]), static=True)
        for cam, (lens, fit) in enumerate(zip(clip.lenses, fits, strict=True)):
            track: int = clip.lens_track[cam]
            name: str = f"lens{cam}"
            log_camera_node(rec, 0, cam, lens_camera(lens, fit, name, width, height), name=name, kind="rgb", image_plane_distance=0.05, camera_model="kb4")
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
    print(f"Wrote {frames} frames per lens and {imu_ns.size} IMU samples to {output}")
