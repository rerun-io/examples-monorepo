"""Check an Insta360 recording's IMU calibration from its own pixels: there is no board in this project, so this is the procedure.

It needs a recording converted with a device profile (the seed rotation, IMU noise and time model come from it); the
feed refuses one without (``catalog_feed.check_imu_is_rig_frame``). It reports, and changes nothing:

1. ``R_P_imu``, the rotation from the IMU frame to the rig frame P, which has camera 0's axes. The native two-lens
   rig's frontend tracks both lenses over chosen spans; each consecutive frame pair gets a relative pose (rotation and translation
   direction, robust epipolar fit on the sphere, seeded from the gyro under the recording's current calibration).
   Rotation-only fits are biased indoors. The visual rotation vectors are then aligned with
   the gyro-integrated ones in closed form, with the gyro bias as a free constant.
2. The frame time offset, two ways: (a) the residual of that alignment over a grid of offsets of the gyro window,
   and (b) the inertial bootstrap's own residual over the same grid on several 2 s windows, with the frame times
   handed to ``Vio`` shifted (no re-conversion). Both are the extra frame time offset to add to the device profile's
   per-frame model, relative to the frame times of the recording at hand.
3. The rolling-shutter readout as a check: the bootstrap residual with the recording's readout, none, and its
   sign flipped (the readout handed to ``Vio``, which corrects the keypoints of every camera with a non-zero readout).
"""

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from itertools import takewhile
from pathlib import Path

import numpy as np
from jaxtyping import Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation

from slam_rs import BackendRequest, _core
from slam_rs.catalog_feed import CameraCalib, Frameset, LocalSegment, RigProfile, SegmentFeed, open_segment
from slam_rs.catalog_timing import ImuStream
from slam_rs.config import load_slam_config

CAUCHY_RAD: float = 2.0 * np.pi / 960.0
"""Smallest Cauchy scale of the epipolar fit, radians: 0.375 deg, about 3.5 px at the centre of a 1920 px X4 Air lens."""
MIN_MATCHES: int = 30
"""A frame pair with fewer common tracks gives no relative pose."""


@dataclass
class Config:
    """Measure R_P_imu, the frame time offset and the rolling-shutter readout of a converted Insta360 recording."""

    rrd: Path
    """Converted Insta360 recording."""
    dataset: str = "insta360-native"
    """Dataset whose VIO config (``slam.toml``) the frontend and the bootstrap run with."""
    rotation_spans_s: tuple[float, ...] = (40.0, 190.0)
    """Start of each span the rotation is fitted on, seconds after the first frame; two disjoint spans give the agreement."""
    span_s: float = 20.0
    """Length of each rotation span, seconds."""
    scan_windows_s: tuple[float, ...] = (20.0, 60.0, 100.0, 140.0, 180.0, 220.0)
    """Start of each bootstrap window, seconds after the first frame; empty skips the bootstrap scan and the shutter check."""
    window_s: float = 2.3
    """Frames decoded per bootstrap window; the bootstrap's first attempt needs 2 s of them. The window's framesets are held
    for its repeated runs: about 70 framesets of two 1920x1920 gray images (0.5 GB) at the default."""
    offsets_ms: tuple[float, ...] = tuple(float(value) for value in range(-8, 9))
    """Offsets added to the frame times, ms, for both offset estimators."""
    backend: BackendRequest = "cpu"
    """Frontend lane."""
    decode_threads: int = 4
    """PyAV threads per lens stream."""


@dataclass(frozen=True, slots=True)
class TrackedFrame:
    """The frontend's tracks on one frameset, both lenses, as unit bearings in P."""

    t_ns: int
    """The frameset's stamp, unshifted."""
    ids: Int64[ndarray, " n"]
    """Keypoint ids."""
    bearings: Float64[ndarray, "n 3"]
    """Unit bearings in the rig frame P (camera 0's axes)."""


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


def rotation_angle_deg(rotation: Float64[ndarray, "3 3"]) -> float:
    """Angle of ``rotation`` in degrees."""
    return float(np.degrees(Rotation.from_matrix(rotation).magnitude()))


def nearest_axis_permutation(rotation: Float64[ndarray, "3 3"]) -> Float64[ndarray, "3 3"]:
    """The signed axis permutation (rotation of the cube group) nearest to ``rotation``."""
    group: Rotation = Rotation.create_group("O")
    return group[int(np.argmin((group * Rotation.from_matrix(rotation).inv()).magnitude()))].as_matrix()


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
        rotation = Rotation.from_rotvec(delta[:3]).as_matrix() @ rotation
        direction = direction + basis.T @ delta[3:]
        direction /= np.linalg.norm(direction)
        if np.linalg.norm(delta) < 1e-9:
            break
    return PairPose(rotation=Rotation.from_matrix(rotation).as_matrix(), direction=direction)


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
        pose: PairPose = relative_pose(first.bearings[common[1]], second.bearings[common[2]], Rotation.from_rotvec(-seeds[index]).as_matrix())
        kept.append(index)
        rotvecs.append(-Rotation.from_matrix(pose.rotation).as_rotvec())
    return VisualPairs(t0_ns=t0_ns[kept], t1_ns=t1_ns[kept], pano_rotvecs=np.array(rotvecs).reshape(-1, 3))


def fit_hand_eye(pairs: VisualPairs, imu: ImuStream, offset_ms: float, iterations: int = 30) -> HandEyeFit:
    """``R_P_imu`` and the gyro bias with ``v = R (g - b dt)``: weighted Kabsch (``align_vectors``) for R, least squares for b,
    Huber weights, alternated."""
    gyro: Float64[ndarray, "n 3"] = integrated_gyro(imu, pairs.t0_ns, pairs.t1_ns, offset_ms)
    dt_s: Float64[ndarray, " n"] = (pairs.t1_ns - pairs.t0_ns) * 1e-9
    weights: Float64[ndarray, " n"] = np.ones(dt_s.size)
    bias: Float64[ndarray, " 3"] = np.zeros(3)
    rotation: Float64[ndarray, "3 3"] = np.eye(3)
    errors: Float64[ndarray, " n"] = np.zeros(dt_s.size)
    for _ in range(iterations):
        rotation = Rotation.align_vectors(pairs.pano_rotvecs, gyro - np.outer(dt_s, bias), weights=weights)[0].as_matrix()
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
    report: _core.BootstrapReport | None = vio.bootstrap_report()
    return report.attempts[0] if report is not None and report.attempts else None


def drive(vio: _core.Vio, framesets: Iterable[Frameset], imu: ImuStream, offset_ns: int, stop_at_attempt: bool) -> list[TrackedFrame]:
    """Track framesets with their times shifted by ``offset_ns``, pushing the IMU one sample ahead; every camera's tracks in P.

    Only the tracks are kept, so ``framesets`` may be a stream of any length.
    """
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
            ids: Int64[ndarray, " n"] = np.concatenate([flow.ids(camera) for camera in range(vio.camera_count)]).astype(np.int64)
            bearings: Float64[ndarray, "n 3"] = np.concatenate([vio.bearings(camera, flow.positions(camera)) for camera in range(vio.camera_count)])
            # A track can drift past a lens's field, where the model has no bearing: drop it, one NaN breaks the fits.
            valid: np.ndarray = np.isfinite(bearings).all(axis=1)
            frames.append(TrackedFrame(t_ns=frameset.t_ns, ids=ids[valid], bearings=bearings[valid]))
        if stop_at_attempt and first_attempt(vio) is not None:
            break
    return frames


def vio_with_readout(feed: SegmentFeed, vio_config: _core.VioConfig, backend: BackendRequest, readout_s: float) -> _core.Vio:
    """The feed's ``Vio`` with every camera's rolling-shutter readout replaced by ``readout_s`` (the shutter check)."""
    cameras: tuple[CameraCalib, ...] = tuple(replace(camera, readout_s=readout_s) for camera in feed.cameras)
    # Only the frontend's tracks are read: the back end (on by default) stays off, as for every raw-only run.
    return _core.Vio(_core.Calibration.from_catalog(cameras, feed.imu), vio_config.with_backend("off"), backend=backend)


def scan_window(
    feed: SegmentFeed,
    vio_config: _core.VioConfig,
    backend: BackendRequest,
    framesets: list[Frameset],
    imu: ImuStream,
    start_s: float,
    offsets_ms: tuple[float, ...],
    recording_readout_s: float,
) -> tuple[WindowScan, ShutterCheck]:
    """Estimator (b) and the shutter check on one window: one ``Vio`` per offset (and per readout at offset 0), each run
    until the bootstrap's first attempt."""

    def attempt(offset_ms: float, readout_s: float) -> _core.BootstrapAttempt | None:
        vio: _core.Vio = vio_with_readout(feed, vio_config, backend, readout_s)
        drive(vio, framesets, imu, int(round(offset_ms * 1e6)), True)
        return first_attempt(vio)

    results: list[_core.BootstrapAttempt | None] = [attempt(offset_ms, recording_readout_s) for offset_ms in offsets_ms]
    scan: WindowScan = WindowScan(
        start_s=start_s,
        offsets_ms=offsets_ms,
        residual_rad=tuple(None if result is None else result.residual_rms_rad for result in results),
        verdicts=tuple("none" if result is None else result.verdict for result in results),
    )
    at_zero: _core.BootstrapAttempt | None = results[offsets_ms.index(0.0)] if 0.0 in offsets_ms else attempt(0.0, recording_readout_s)
    values: list[float | None] = [None if check is None else check.residual_rms_rad for check in (at_zero, attempt(0.0, 0.0), attempt(0.0, -recording_readout_s))]
    return scan, ShutterCheck(start_s=start_s, on=values[0], off=values[1], flipped=values[2])


def print_report(report: CalibrationReport, offsets_ms: tuple[float, ...]) -> None:
    """The report as plain-text tables, then the device-profile values it implies."""
    np.set_printoptions(precision=5, suppress=True)
    rotation: RotationReport = report.rotation
    print("\n1. R_P_imu (IMU -> rig P, camera 0's axes), 5-DoF pair fits aligned with the gyro")
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
        print(f"\nextra frame time offset, ms (relative to this recording's frame times): {mean_ms:+.1f}")
    print(f"IMU to rig rotation (R_P_imu, joint fit):\n{rotation.joint.pano_R_imu}")


def main(config: Config) -> CalibrationReport:
    """Run the three measurements on the recording, print them and return them."""
    vio_config: _core.VioConfig = _core.VioConfig.from_json(load_slam_config().vio_config_text(config.dataset, profile="fast"))
    with open_segment(LocalSegment(config.rrd), profile=RigProfile(decode_threads=config.decode_threads), window_s=1.0, include_ground_truth=False) as feed:
        readout_s: float | None = feed.cameras[0].readout_s
        if readout_s is None:
            raise ValueError(f"{config.rrd}: camera 0 states no rolling_shutter_readout_s; convert the recording again")
        # P has camera 0's axes, so R_P_imu is camera 0's rotation from the IMU.
        current: Float64[ndarray, "3 3"] = feed.cameras[0].imu_T_cam[:3, :3].T.copy()
        first_ns: int = int(feed.frame_t_ns[0])

        def read(start_s: float, length_s: float) -> tuple[Iterable[Frameset], ImuStream]:
            """The span's framesets as a stream (decoded as they are consumed) and its IMU with 0.5 s either side."""
            start_ns: int = first_ns + int(start_s * 1e9)
            stop_ns: int = start_ns + int(length_s * 1e9)
            framesets: Iterable[Frameset] = takewhile(lambda frameset: frameset.t_ns <= stop_ns, feed.framesets(stop_ns, start_ns))
            return framesets, feed.imu_between(start_ns - 500_000_000, stop_ns + 500_000_000)

        spans: list[SpanFit] = []
        all_pairs: list[VisualPairs] = []
        for start_s in config.rotation_spans_s:
            framesets, imu = read(start_s, config.span_s)
            tracked: list[TrackedFrame] = drive(vio_with_readout(feed, vio_config, config.backend, readout_s), framesets, imu, 0, False)
            pairs: VisualPairs = visual_pairs(tracked, imu, current)
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
        permutation: Float64[ndarray, "3 3"] = nearest_axis_permutation(joint.pano_R_imu)
        rotation: RotationReport = RotationReport(
            spans=spans,
            joint=joint,
            joint_angle_to_current_deg=rotation_angle_deg(joint.pano_R_imu @ current.T),
            span_agreement_deg=rotation_angle_deg(spans[0].fit.pano_R_imu @ spans[1].fit.pano_R_imu.T) if len(spans) > 1 else None,
            nearest_permutation=permutation,
            angle_to_permutation_deg=rotation_angle_deg(joint.pano_R_imu @ permutation.T),
        )
        report: CalibrationReport = CalibrationReport(current_pano_R_imu=current, readout_s=readout_s, rotation=rotation)
        for start_s in config.scan_windows_s:
            stream, imu = read(start_s, config.window_s)
            scan, check = scan_window(feed, vio_config, config.backend, list(stream), imu, start_s, config.offsets_ms, readout_s)
            report.scans.append(scan)
            report.shutter.append(check)
            print(f"window {start_s:.0f} s: minimum {scan.minimum_ms()} ms; shutter on/off/flipped {[check.on, check.off, check.flipped]}", flush=True)
    print_report(report, config.offsets_ms)
    return report
