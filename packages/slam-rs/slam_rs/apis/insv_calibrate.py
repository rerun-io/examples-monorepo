"""Check an Insta360 recording's IMU calibration from its own pixels: there is no board in this project, so this is the procedure.

It reports, and changes nothing:

1. ``R_P_imu``, the rotation from the IMU frame to the panorama frame P. The lane's own frontend tracks the two
   hemisphere tiles over chosen spans; each consecutive frame pair gets a relative pose (rotation and translation
   direction, robust epipolar fit on the sphere, seeded from the gyro under the recording's current calibration).
   Rotation-only fits are biased indoors (recon/09 section 2). The visual rotation vectors are then aligned with
   the gyro-integrated ones in closed form, with the gyro bias as a free constant.
2. The frame time offset, two ways: (a) the residual of that alignment over a grid of offsets of the gyro window,
   and (b) the inertial bootstrap's own residual over the same grid on several 2 s windows, with the frame times
   handed to ``Vio`` shifted (no re-conversion). Both are "extra offset to add to the converter's per-frame model"
   (``slam-rs-insv --extra-frame-offset-ms``), relative to the frame times of the recording at hand.
3. The rolling-shutter readout as a check: the bootstrap residual with the recording's readout, none, and its
   sign flipped (the readout handed to ``Vio``; ``config.panorama_rolling_shutter`` stays as the dataset sets it).
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from jaxtyping import Float64, Int64
from numpy import ndarray

from slam_rs import BackendRequest, _core
from slam_rs.apis.insv import orthonormal
from slam_rs.catalog_feed import Frameset, LocalSegment, RigProfile, open_segment
from slam_rs.catalog_timing import ImuStream
from slam_rs.config import load_slam_config

PANORAMA_WIDTH: int = 1920
"""Columns of the stitched panorama (``kornia_staging_imgproc::panorama::TileGeometry`` default: 360 degrees)."""
PANORAMA_HEIGHT: int = 960
"""Rows of the stitched panorama (180 degrees)."""
CAUCHY_RAD: float = 2.0 * np.pi / 960.0
"""Smallest Cauchy scale of the epipolar fit: 2 panorama pixels."""
MIN_MATCHES: int = 30
"""A frame pair with fewer common tracks gives no relative pose."""


@dataclass
class Config:
    """Measure R_P_imu, the frame time offset and the rolling-shutter readout of a converted Insta360 recording."""

    rrd: Path
    """Recording written by ``slam-rs-insv``."""
    dataset: str = "insta360"
    """Dataset whose VIO config (``slam.toml``) the frontend and the bootstrap run with."""
    rotation_spans_s: tuple[float, ...] = (40.0, 190.0)
    """Start of each span the rotation is fitted on, seconds after the first frame; two disjoint spans give the agreement."""
    span_s: float = 20.0
    """Length of each rotation span, seconds."""
    scan_windows_s: tuple[float, ...] = (20.0, 60.0, 100.0, 140.0, 180.0, 220.0)
    """Start of each bootstrap window, seconds after the first frame; empty skips the bootstrap scan and the shutter check."""
    window_s: float = 2.3
    """Frames decoded per bootstrap window; the bootstrap's first attempt needs 2 s of them."""
    offsets_ms: tuple[float, ...] = tuple(float(value) for value in range(-8, 9))
    """Offsets added to the frame times, ms, for both offset estimators."""
    backend: BackendRequest = "cpu"
    """Frontend lane."""
    decode_threads: int = 4
    """PyAV threads per lens stream."""


@dataclass(frozen=True, slots=True)
class TrackedFrame:
    """The frontend's tracks on one frameset, as unit bearings in P."""

    t_ns: int
    """The frameset's stamp, unshifted."""
    ids: Int64[ndarray, " n"]
    """Keypoint ids."""
    bearings: Float64[ndarray, "n 3"]
    """Unit bearings in the panorama frame P."""


@dataclass(frozen=True, slots=True)
class PairPose:
    """Relative pose of one frame pair: ``rays1 ~ rotation @ rays0 + direction * depth``."""

    rotation: Float64[ndarray, "3 3"]
    """Maps frame-0 bearings into frame 1."""
    direction: Float64[ndarray, " 3"]
    """Unit translation direction in frame 1, sign free."""


@dataclass(frozen=True, slots=True)
class VisualPairs:
    """Relative rotations of consecutive frame pairs."""

    t0_ns: Int64[ndarray, " n"]
    """Stamp of each pair's first frame."""
    t1_ns: Int64[ndarray, " n"]
    """Stamp of each pair's second frame."""
    pano_rotvecs: Float64[ndarray, "n 3"]
    """Rotation vector of rig_0 <- rig_1 in P, the visual counterpart of the integrated gyro."""


@dataclass(frozen=True, slots=True)
class HandEyeFit:
    """The rotation that aligns the visual rotations with the gyro-integrated ones."""

    pano_R_imu: Float64[ndarray, "3 3"]
    """Fitted ``R_P_imu``."""
    bias_rad_s: Float64[ndarray, " 3"]
    """Gyro bias, IMU frame."""
    residual_rad_s: float
    """Weighted RMS of the rate residual."""
    weights: Float64[ndarray, " n"]
    """Robust weight of each pair."""


@dataclass(frozen=True, slots=True)
class WindowScan:
    """The bootstrap residual over the offset grid on one window."""

    start_s: float
    """Window start, seconds after the first frame."""
    offsets_ms: tuple[float, ...]
    """Offsets added to the frame times."""
    residual_rad: tuple[float | None, ...]
    """First attempt's residual at each offset; None when it reported none."""
    verdicts: tuple[str, ...]
    """First attempt's verdict at each offset."""

    def minimum_ms(self) -> float | None:
        """Parabola minimum over the offsets with a residual; None with fewer than three."""
        keep: list[int] = [index for index, value in enumerate(self.residual_rad) if value is not None]
        if len(keep) < 3:
            return None
        return parabola_minimum(np.asarray(self.offsets_ms)[keep], np.asarray([self.residual_rad[index] for index in keep], dtype=np.float64))


@dataclass(frozen=True, slots=True)
class SpanFit:
    """The rotation fit on one span."""

    start_s: float
    """Span start, seconds after the first frame."""
    pairs: int
    """Frame pairs with a relative pose."""
    rate_curve: tuple[float, ...]
    """Estimator (a): rate residual at each offset of the grid, rad/s."""
    rate_offset_ms: float
    """Its parabola minimum."""
    fit: HandEyeFit
    """Rotation fit at that offset."""
    angle_to_current_deg: float
    """Angle between the fit and the recording's ``R_P_imu``."""


@dataclass(frozen=True, slots=True)
class RotationReport:
    """R_P_imu from every span, alone and together."""

    spans: list[SpanFit]
    """One fit per span."""
    joint: HandEyeFit
    """All pairs together, at the mean rate offset."""
    joint_angle_to_current_deg: float
    """Joint fit against the recording's ``R_P_imu``."""
    span_agreement_deg: float | None
    """Angle between the first two spans' fits; None with one span."""
    nearest_permutation: Float64[ndarray, "3 3"]
    """Signed axis permutation nearest to the joint fit."""
    angle_to_permutation_deg: float
    """Joint fit against that permutation."""


@dataclass(frozen=True, slots=True)
class ShutterCheck:
    """Bootstrap residual at offset 0 with the recording's readout, none, and its sign flipped (rad)."""

    start_s: float
    """Window start, seconds after the first frame."""
    on: float | None
    """Recording's readout."""
    off: float | None
    """Readout 0."""
    flipped: float | None
    """Readout negated."""


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Everything the tool measured on one recording."""

    current_pano_R_imu: Float64[ndarray, "3 3"]
    """``R_P_imu`` the converter posed the lenses with."""
    readout_s: float
    """The recording's rolling-shutter readout."""
    rotation: RotationReport
    """Item 1, and estimator (a) per span."""
    scans: list[WindowScan] = field(default_factory=list)
    """Estimator (b) per window."""
    shutter: list[ShutterCheck] = field(default_factory=list)
    """Item 3 per window."""


def rotation_exp(rotvec: Float64[ndarray, " 3"]) -> Float64[ndarray, "3 3"]:
    """Rodrigues: the rotation of angle ``|rotvec|`` about ``rotvec``."""
    angle: float = float(np.linalg.norm(rotvec))
    cross: Float64[ndarray, "3 3"] = np.array([[0.0, -rotvec[2], rotvec[1]], [rotvec[2], 0.0, -rotvec[0]], [-rotvec[1], rotvec[0], 0.0]])
    if angle < 1e-12:
        return np.eye(3) + cross
    return np.eye(3) + np.sin(angle) / angle * cross + (1.0 - np.cos(angle)) / angle**2 * cross @ cross


def rotation_log(rotation: Float64[ndarray, "3 3"]) -> Float64[ndarray, " 3"]:
    """Rotation vector of ``rotation`` (angles below pi)."""
    angle: float = float(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
    axis: Float64[ndarray, " 3"] = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]])
    return 0.5 * axis if angle < 1e-12 else angle / (2.0 * np.sin(angle)) * axis


def rotation_angle_deg(rotation: Float64[ndarray, "3 3"]) -> float:
    """Angle of ``rotation`` in degrees."""
    return float(np.degrees(np.linalg.norm(rotation_log(rotation))))


def nearest_permutation(rotation: Float64[ndarray, "3 3"]) -> Float64[ndarray, "3 3"]:
    """The signed axis permutation nearest to ``rotation``, row by row."""
    permutation: Float64[ndarray, "3 3"] = np.zeros((3, 3))
    for row in range(3):
        column: int = int(np.argmax(np.abs(rotation[row])))
        permutation[row, column] = np.sign(rotation[row, column])
    return permutation


def tile_bearings(pixels_xy: Float64[ndarray, "n 2"], origin: int) -> Float64[ndarray, "n 3"]:
    """Unit bearings in P of hemisphere-tile pixels; the tile's column 0 is panorama column ``origin``.

    Panorama column ``c`` is longitude ``(c + 0.5) 2 pi / W - pi / 2`` and row ``r`` latitude ``(H / 2 - 0.5 - r) 2 pi / W``;
    the bearing is ``(cos lat sin lon, -sin lat, cos lat cos lon)`` (``TileGeometry::bearing``).
    """
    step: float = 2.0 * np.pi / PANORAMA_WIDTH
    lon: Float64[ndarray, " n"] = (pixels_xy[:, 0] + origin + 0.5) * step - np.pi / 2
    lat: Float64[ndarray, " n"] = (PANORAMA_HEIGHT / 2 - 0.5 - pixels_xy[:, 1]) * step
    return np.stack([np.cos(lat) * np.sin(lon), -np.sin(lat), np.cos(lat) * np.cos(lon)], axis=1)


def epipolar_residuals(
    rotation: Float64[ndarray, "3 3"], direction: Float64[ndarray, " 3"], rays0: Float64[ndarray, "n 3"], rays1: Float64[ndarray, "n 3"]
) -> Float64[ndarray, " n"]:
    """Sine of each frame-1 ray's angle off its epipolar plane, the plane through ``direction`` and ``rotation @ ray0``."""
    normals: Float64[ndarray, "n 3"] = np.cross(direction, rays0 @ rotation.T)
    return np.einsum("ij,ij->i", rays1, normals) / np.maximum(np.linalg.norm(normals, axis=1), 1e-12)


def relative_pose(rays0: Float64[ndarray, "n 3"], rays1: Float64[ndarray, "n 3"], seed: Float64[ndarray, "3 3"], iterations: int = 30) -> PairPose:
    """Robust 5-DoF relative pose (rotation and translation direction) from matched unit bearings.

    Gauss-Newton on the epipolar residual with a Cauchy loss whose scale shrinks from twice the median residual
    down to :data:`CAUCHY_RAD`. The direction starts as the null vector of the epipolar constraints under ``seed``.
    """
    rotation: Float64[ndarray, "3 3"] = seed.copy()
    direction: Float64[ndarray, " 3"] = np.linalg.svd(np.cross(rays0 @ rotation.T, rays1))[2][-1]
    for _ in range(iterations):
        # r = b1 . n / |n| with n = t x q, q = R b0; dr/dn = g = (b1 - r n/|n|) / |n|. A rotation step d moves q by d x q,
        # so dr = d . (q x (g x t)); a direction step s moves t by s, so dr = s . (q x g).
        rotated: Float64[ndarray, "n 3"] = rays0 @ rotation.T
        normals: Float64[ndarray, "n 3"] = np.cross(direction, rotated)
        lengths: Float64[ndarray, " n"] = np.maximum(np.linalg.norm(normals, axis=1), 1e-12)
        residual: Float64[ndarray, " n"] = np.einsum("ij,ij->i", rays1, normals) / lengths
        gradient: Float64[ndarray, "n 3"] = (rays1 - residual[:, None] * normals / lengths[:, None]) / lengths[:, None]
        scale: float = max(CAUCHY_RAD, 2.0 * float(np.median(np.abs(residual))))
        weights: Float64[ndarray, " n"] = 1.0 / (1.0 + (residual / scale) ** 2)
        basis: Float64[ndarray, "2 3"] = np.linalg.svd(direction[None, :])[2][1:]
        jacobian: Float64[ndarray, "n 5"] = np.concatenate([np.cross(rotated, np.cross(gradient, direction)), np.cross(rotated, gradient) @ basis.T], axis=1)
        normal: Float64[ndarray, "5 5"] = jacobian.T @ (weights[:, None] * jacobian)
        normal += 1e-9 * np.trace(normal) * np.eye(5)
        delta: Float64[ndarray, " 5"] = -np.linalg.solve(normal, jacobian.T @ (weights * residual))
        rotation = rotation_exp(delta[:3]) @ rotation
        direction = direction + basis.T @ delta[3:]
        direction /= np.linalg.norm(direction)
        if np.linalg.norm(delta) < 1e-9:
            break
    return PairPose(rotation=orthonormal(rotation), direction=direction)


def integrated_gyro(imu: ImuStream, t0_ns: Int64[ndarray, " n"], t1_ns: Int64[ndarray, " n"], offset_ms: float) -> Float64[ndarray, "n 3"]:
    """Trapezoid integral of the gyro over ``[t0 + offset, t1 + offset]``, rad (IMU frame, bias included)."""
    times_s: Float64[ndarray, " n_imu"] = (imu.t_ns - imu.t_ns[0]) * 1e-9
    steps: Float64[ndarray, "n_steps 3"] = 0.5 * (imu.gyro_rad_s[1:] + imu.gyro_rad_s[:-1]) * np.diff(times_s)[:, None]
    cumulative: Float64[ndarray, "n_imu 3"] = np.concatenate([np.zeros((1, 3)), np.cumsum(steps, axis=0)])
    start_s: Float64[ndarray, " n"] = (t0_ns - imu.t_ns[0]) * 1e-9 + offset_ms * 1e-3
    end_s: Float64[ndarray, " n"] = (t1_ns - imu.t_ns[0]) * 1e-9 + offset_ms * 1e-3
    return np.stack([np.interp(end_s, times_s, cumulative[:, axis]) - np.interp(start_s, times_s, cumulative[:, axis]) for axis in range(3)], axis=1)


def visual_pairs(frames: list[TrackedFrame], imu: ImuStream, current_pano_R_imu: Float64[ndarray, "3 3"]) -> VisualPairs:
    """Relative rotation of every consecutive frame pair with :data:`MIN_MATCHES` common tracks, seeded from the gyro."""
    t0_ns: Int64[ndarray, " n"] = np.array([frame.t_ns for frame in frames[:-1]], dtype=np.int64)
    t1_ns: Int64[ndarray, " n"] = np.array([frame.t_ns for frame in frames[1:]], dtype=np.int64)
    seeds: Float64[ndarray, "n 3"] = integrated_gyro(imu, t0_ns, t1_ns, 0.0) @ current_pano_R_imu.T
    kept: list[int] = []
    rotvecs: list[Float64[ndarray, " 3"]] = []
    for index, (first, second) in enumerate(zip(frames[:-1], frames[1:], strict=True)):
        common: tuple[Int64[ndarray, " m"], Int64[ndarray, " m"], Int64[ndarray, " m"]] = np.intersect1d(first.ids, second.ids, return_indices=True)
        if common[0].size < MIN_MATCHES:
            continue
        # rays1 = R rays0 with R = (rig_0 <- rig_1)^T, so the seed is exp(-gyro) and the visual rotation vector is -log(R).
        pose: PairPose = relative_pose(first.bearings[common[1]], second.bearings[common[2]], rotation_exp(-seeds[index]))
        kept.append(index)
        rotvecs.append(-rotation_log(pose.rotation))
    return VisualPairs(t0_ns=t0_ns[kept], t1_ns=t1_ns[kept], pano_rotvecs=np.array(rotvecs).reshape(-1, 3))


def fit_hand_eye(pairs: VisualPairs, imu: ImuStream, offset_ms: float, iterations: int = 30) -> HandEyeFit:
    """``R_P_imu`` and the gyro bias with ``v = R (g - b dt)``: Kabsch for R, least squares for b, Huber weights, alternated."""
    gyro: Float64[ndarray, "n 3"] = integrated_gyro(imu, pairs.t0_ns, pairs.t1_ns, offset_ms)
    dt_s: Float64[ndarray, " n"] = (pairs.t1_ns - pairs.t0_ns) * 1e-9
    weights: Float64[ndarray, " n"] = np.ones(dt_s.size)
    bias: Float64[ndarray, " 3"] = np.zeros(3)
    rotation: Float64[ndarray, "3 3"] = np.eye(3)
    errors: Float64[ndarray, " n"] = np.zeros(dt_s.size)
    for _ in range(iterations):
        u, _, vt = np.linalg.svd((weights[:, None] * pairs.pano_rotvecs).T @ (gyro - np.outer(dt_s, bias)))
        rotation = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
        bias = (weights * dt_s) @ (gyro - pairs.pano_rotvecs @ rotation) / float(weights @ dt_s**2)
        errors = np.linalg.norm(pairs.pano_rotvecs - (gyro - np.outer(dt_s, bias)) @ rotation.T, axis=1) / dt_s
        weights = np.minimum(1.0, 3.0 * float(np.median(errors)) / np.maximum(errors, 1e-12))
    return HandEyeFit(pano_R_imu=rotation, bias_rad_s=bias, residual_rad_s=float(np.sqrt(weights @ errors**2 / weights.sum())), weights=weights)


def rate_offset_curve(pairs: VisualPairs, imu: ImuStream, fit: HandEyeFit, offsets_ms: tuple[float, ...]) -> Float64[ndarray, " n_offsets"]:
    """Estimator (a): weighted RMS rate residual at each offset, with the fit's rotation and weights and the bias re-solved."""
    dt_s: Float64[ndarray, " n"] = (pairs.t1_ns - pairs.t0_ns) * 1e-9
    gyro_in_imu: Float64[ndarray, "n 3"] = pairs.pano_rotvecs @ fit.pano_R_imu
    curve: list[float] = []
    for offset_ms in offsets_ms:
        gyro: Float64[ndarray, "n 3"] = integrated_gyro(imu, pairs.t0_ns, pairs.t1_ns, offset_ms)
        bias: Float64[ndarray, " 3"] = (fit.weights * dt_s) @ (gyro - gyro_in_imu) / float(fit.weights @ dt_s**2)
        errors: Float64[ndarray, " n"] = np.linalg.norm(gyro_in_imu - gyro + np.outer(dt_s, bias), axis=1) / dt_s
        curve.append(float(np.sqrt(fit.weights @ errors**2 / fit.weights.sum())))
    return np.asarray(curve)


def parabola_minimum(offsets: Float64[ndarray, " n"], values: Float64[ndarray, " n"]) -> float:
    """Vertex of the parabola through the lowest grid point and two neighbours on each side; the grid point at an edge."""
    lowest: int = int(np.argmin(values))
    if lowest in (0, values.size - 1):
        return float(offsets[lowest])
    first, last = max(lowest - 2, 0), min(lowest + 3, values.size)
    curvature, slope, _ = np.polyfit(offsets[first:last], values[first:last], 2)
    if curvature <= 0:
        return float(offsets[lowest])
    return float(np.clip(-slope / (2 * curvature), offsets[first], offsets[last - 1]))


def offset_summary(scans: list[WindowScan]) -> tuple[float, float]:
    """Estimator (b): mean and standard deviation of the windows' minima (NaN when no window has one)."""
    minima: list[float] = [value for value in (scan.minimum_ms() for scan in scans) if value is not None]
    if not minima:
        return float("nan"), float("nan")
    return float(np.mean(minima)), float(np.std(minima, ddof=1)) if len(minima) > 1 else 0.0


def first_attempt(vio: _core.Vio) -> _core.BootstrapAttempt | None:
    """The inertial bootstrap's first attempt, or None before the first solve (and on the stereo start)."""
    attempts: list[_core.BootstrapAttempt] = vio.bootstrap_attempts
    return attempts[0] if attempts else None


def drive(vio: _core.Vio, framesets: Iterable[Frameset], imu: ImuStream, offset_ns: int, stop_at_attempt: bool) -> list[TrackedFrame]:
    """Track framesets with their times shifted by ``offset_ns``, pushing the IMU one sample ahead; the tracks in P."""
    origins: tuple[int, int] | None = vio.panorama_tile_origins
    if origins is None:
        raise ValueError("the dataset config does not stitch a panorama: run with an Insta360 dataset (config.panorama_dual_fisheye)")
    frames: list[TrackedFrame] = []
    pushed: int = 0
    for frameset in framesets:
        t_ns: int = frameset.t_ns + offset_ns
        stop: int = min(int(np.searchsorted(imu.t_ns, t_ns, side="right")) + 1, len(imu))
        if stop > pushed:
            vio.push_imu_batch(imu.t_ns[pushed:stop], np.ascontiguousarray(imu.gyro_rad_s[pushed:stop]), np.ascontiguousarray(imu.accel_m_s2[pushed:stop]))
            pushed = stop
        if vio.track(t_ns, frameset.images).status == _core.VioStatus.NeedMoreImu:
            raise ValueError(f"frameset {frameset.t_ns} + {offset_ns} ns runs past the IMU read for it")
        flow: _core.FlowFrame | None = vio.flow_frame()
        if flow is not None:
            frames.append(
                TrackedFrame(
                    t_ns=frameset.t_ns,
                    ids=np.concatenate([flow.ids(tile) for tile in range(2)]).astype(np.int64),
                    bearings=np.concatenate([tile_bearings(flow.positions(tile).astype(np.float64), origins[tile]) for tile in range(2)]),
                )
            )
        if stop_at_attempt and first_attempt(vio) is not None:
            break
    return frames


def print_report(report: CalibrationReport, offsets_ms: tuple[float, ...]) -> None:
    """The report as plain-text tables, then the converter flags it implies."""
    np.set_printoptions(precision=5, suppress=True)
    rotation: RotationReport = report.rotation
    print("\n1. R_P_imu (IMU -> panorama P), 5-DoF pair fits aligned with the gyro")
    print("| span start (s) | pairs | angle to current (deg) | bias (rad/s, imu) | rate residual (rad/s) | rate offset (a) (ms) |")
    print("|---|---|---|---|---|---|")
    for span in rotation.spans:
        print(
            f"| {span.start_s:.0f} | {span.pairs} | {span.angle_to_current_deg:.3f} | {np.round(span.fit.bias_rad_s, 4)} | "
            f"{span.fit.residual_rad_s:.4f} | {span.rate_offset_ms:+.2f} |"
        )
    print(f"joint fit (all spans):\n{rotation.joint.pano_R_imu}")
    print(f"current (recording):\n{report.current_pano_R_imu}")
    agreement: str = "n/a (one span)" if rotation.span_agreement_deg is None else f"{rotation.span_agreement_deg:.3f} deg"
    print(
        f"joint: {rotation.joint_angle_to_current_deg:.3f} deg from the current constant, {rotation.angle_to_permutation_deg:.3f} deg from the "
        f"permutation {rotation.nearest_permutation.astype(int).tolist()}; spans agree within {agreement}; residual {rotation.joint.residual_rad_s:.4f} rad/s"
    )
    print("\n2a. rate residual (rad/s) against the offset added to the frame times")
    print("| offset (ms) | " + " | ".join(f"span {span.start_s:.0f} s" for span in rotation.spans) + " |")
    print("|---|" + "---|" * len(rotation.spans))
    for index, offset in enumerate(offsets_ms):
        print(f"| {offset:+.1f} | " + " | ".join(f"{span.rate_curve[index]:.4f}" for span in rotation.spans) + " |")
    rate_minima: list[float] = [span.rate_offset_ms for span in rotation.spans]
    print(f"(a) minima {[round(value, 2) for value in rate_minima]} ms: mean {np.mean(rate_minima):+.2f} ms")
    if report.scans:
        print("\n2b. bootstrap residual (mrad, first 2 s attempt; * = rejected) against the offset added to the frame times")
        print("| offset (ms) | " + " | ".join(f"{scan.start_s:.0f} s" for scan in report.scans) + " |")
        print("|---|" + "---|" * len(report.scans))
        for index, offset in enumerate(offsets_ms):
            cells: list[str] = []
            for scan in report.scans:
                value: float | None = scan.residual_rad[index]
                cells.append("-" if value is None else f"{1e3 * value:.3f}{'' if scan.verdicts[index] == 'Accepted' else '*'}")
            print(f"| {offset:+.1f} | " + " | ".join(cells) + " |")
        minima: list[str] = ["-" if value is None else f"{value:+.2f}" for value in (scan.minimum_ms() for scan in report.scans)]
        mean_ms, spread_ms = offset_summary(report.scans)
        print(f"| minimum (ms) | {' | '.join(minima)} |\n(b) mean {mean_ms:+.2f} ms, spread (std) {spread_ms:.2f} ms")
        print(f"\n3. rolling shutter: bootstrap residual (mrad) at offset 0, readout {report.readout_s * 1e3:.3f} ms")
        print("| window start (s) | recording's readout | none | sign flipped |\n|---|---|---|---|")
        for check in report.shutter:
            print("| " + " | ".join([f"{check.start_s:.0f}"] + ["-" if value is None else f"{1e3 * value:.3f}" for value in (check.on, check.off, check.flipped)]) + " |")
        print(f"\nconverter flags implied (relative to this recording's frame times): --extra-frame-offset-ms {mean_ms:.1f}")


def main(config: Config) -> CalibrationReport:
    """Run the three measurements on the recording, print them and return them."""
    vio_config: _core.VioConfig = _core.VioConfig.from_json(load_slam_config().vio_config_text(config.dataset, profile="fast"))
    with open_segment(LocalSegment(config.rrd), profile=RigProfile(decode_threads=config.decode_threads), window_s=1.0, include_ground_truth=False) as feed:
        if feed.insta360 is None or feed.insta360.pano_from_imu is None:
            raise ValueError(f"{config.rrd}: no insta360 pano_from_imu property; convert it again with slam-rs-insv")
        current: Float64[ndarray, "3 3"] = feed.insta360.pano_from_imu
        calibration: _core.Calibration = _core.Calibration.from_catalog(feed.cameras, feed.imu)
        first_ns: int = int(feed.frame_t_ns[0])

        def new_vio(readout_s: float) -> _core.Vio:
            assert feed.insta360 is not None
            argument: tuple[str, tuple[int, int, int, int, int, int] | None, float] = feed.insta360.vio_argument()
            return _core.Vio(calibration, vio_config, backend=config.backend, dual_fisheye=(argument[0], argument[1], readout_s))

        def read(start_s: float, length_s: float) -> tuple[list[Frameset], ImuStream]:
            start_ns: int = first_ns + int(start_s * 1e9)
            stop_ns: int = start_ns + int(length_s * 1e9)
            framesets: list[Frameset] = [frameset for frameset in feed.framesets(stop_ns, start_ns) if frameset.t_ns <= stop_ns]
            return framesets, feed.imu_between(start_ns - 500_000_000, stop_ns + 500_000_000)

        spans: list[SpanFit] = []
        all_pairs: list[VisualPairs] = []
        for start_s in config.rotation_spans_s:
            framesets, imu = read(start_s, config.span_s)
            pairs: VisualPairs = visual_pairs(drive(new_vio(feed.insta360.readout_s), framesets, imu, 0, False), imu, current)
            del framesets
            curve: Float64[ndarray, " n_offsets"] = rate_offset_curve(pairs, imu, fit_hand_eye(pairs, imu, 0.0), config.offsets_ms)
            rate_offset_ms: float = parabola_minimum(np.asarray(config.offsets_ms), curve)
            fit: HandEyeFit = fit_hand_eye(pairs, imu, rate_offset_ms)
            spans.append(SpanFit(start_s, len(pairs.t0_ns), tuple(curve.tolist()), rate_offset_ms, fit, rotation_angle_deg(fit.pano_R_imu @ current.T)))
            all_pairs.append(pairs)
            print(f"span {start_s:.0f} s: {len(pairs.t0_ns)} pairs, {spans[-1].angle_to_current_deg:.3f} deg from the current R_P_imu", flush=True)
        joint_pairs: VisualPairs = VisualPairs(
            t0_ns=np.concatenate([pairs.t0_ns for pairs in all_pairs]),
            t1_ns=np.concatenate([pairs.t1_ns for pairs in all_pairs]),
            pano_rotvecs=np.concatenate([pairs.pano_rotvecs for pairs in all_pairs]),
        )
        joint_imu: ImuStream = feed.imu_between(int(joint_pairs.t0_ns.min()) - 500_000_000, int(joint_pairs.t1_ns.max()) + 500_000_000)
        joint: HandEyeFit = fit_hand_eye(joint_pairs, joint_imu, float(np.mean([span.rate_offset_ms for span in spans])))
        permutation: Float64[ndarray, "3 3"] = nearest_permutation(joint.pano_R_imu)
        rotation: RotationReport = RotationReport(
            spans=spans,
            joint=joint,
            joint_angle_to_current_deg=rotation_angle_deg(joint.pano_R_imu @ current.T),
            span_agreement_deg=rotation_angle_deg(spans[0].fit.pano_R_imu @ spans[1].fit.pano_R_imu.T) if len(spans) > 1 else None,
            nearest_permutation=permutation,
            angle_to_permutation_deg=rotation_angle_deg(joint.pano_R_imu @ permutation.T),
        )
        report: CalibrationReport = CalibrationReport(current_pano_R_imu=current, readout_s=feed.insta360.readout_s, rotation=rotation)
        for start_s in config.scan_windows_s:
            framesets, imu = read(start_s, config.window_s)

            def attempt(offset_ms: float, readout_s: float) -> _core.BootstrapAttempt | None:
                vio: _core.Vio = new_vio(readout_s)
                drive(vio, framesets, imu, int(round(offset_ms * 1e6)), True)  # noqa: B023 - called inside this iteration only
                return first_attempt(vio)

            readout_s: float = feed.insta360.readout_s
            results: list[_core.BootstrapAttempt | None] = [attempt(offset_ms, readout_s) for offset_ms in config.offsets_ms]
            report.scans.append(
                WindowScan(
                    start_s=start_s,
                    offsets_ms=config.offsets_ms,
                    residual_rad=tuple(None if result is None else result.residual_rms_rad for result in results),
                    verdicts=tuple("none" if result is None else result.verdict for result in results),
                )
            )
            at_zero: _core.BootstrapAttempt | None = results[config.offsets_ms.index(0.0)] if 0.0 in config.offsets_ms else attempt(0.0, readout_s)
            checks: list[_core.BootstrapAttempt | None] = [at_zero, attempt(0.0, 0.0), attempt(0.0, -readout_s)]
            values: list[float | None] = [None if check is None else check.residual_rms_rad for check in checks]
            report.shutter.append(ShutterCheck(start_s=start_s, on=values[0], off=values[1], flipped=values[2]))
            print(f"window {start_s:.0f} s: minimum {report.scans[-1].minimum_ms()} ms; shutter on/off/flipped {values}", flush=True)
    print_report(report, config.offsets_ms)
    return report
