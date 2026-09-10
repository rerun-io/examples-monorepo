"""The frozen reference set, read from ``reference_segments.toml``, and the gate's thresholds.

The manifest is the single place that says which segments the accuracy work runs
on, where their layers live, what the catalog is expected to report about them,
which decode path produces the pixels, which basalt VIO config the estimator runs
and which IMU noise numbers it is tuned with. The catalog carries none of the last
three; the decode path alone moves ATE by centimetres and the config's own
``vio_marg_lost_landmarks`` was worth up to 12 cm (C72), so a run is not
reproducible without them.

The V2 tolerances (:data:`ATE_VS_CPP_CM`, :data:`PATH_BOUND_MAX_CLIP_S`,
:data:`GT_BAND_RATIO`, :data:`SPEED_TOLERANCE`, :data:`DIVERGENCE_FACTOR`) sit
here rather than in the gate test, because they are the milestone's verdict and
it is decided from measurement: S15 measured the band and D60 is what this table
now says.
"""

import json
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, TypeAlias

from slam_rs import _core, reference_bundle
from slam_rs.reference_bundle import BundleFile
from slam_rs.trajectory import MIN_ASSOCIATED_POSES

MANIFEST_PATH: Path = Path(__file__).resolve().parents[1] / "reference_segments.toml"
"""The checked-in manifest, beside the package rather than inside it."""

Tier: TypeAlias = Literal["smoke", "accuracy", "long"]
"""How often a segment runs: every commit, per pull request, nightly."""
GatePolicy: TypeAlias = Literal["tight", "standard", "no_divergence"]
"""How hard a segment may be gated.

``tight`` segments are where basalt is comfortably accurate (about 1-2 cm) and a
regression is unambiguous. ``standard`` segments are harder but still stable, and
gate relative to basalt's own number rather than an absolute threshold.
``no_divergence`` segments are ones where basalt itself is near failure: on
``MGO01_low_light`` and ``MGO13_sudden_movements`` the C++ binary gets 68 cm, and
merely changing between two legitimate decode paths of the same estimator moves
the answer by 32 cm and 18 cm — on MGO01 the ordering even flips. A tolerance
there would measure noise, so those two only gate "kept tracking, did not
diverge".
"""
DecodePath: TypeAlias = Literal["cpu_gray8_dav1d_1thread", "cpu_gray8_swscale_area_downscale3"]
"""Frozen pixel provenance.

The MSD gate is frozen on the single-threaded dav1d ``gray8`` path (D28). RoboCap's
H.264 streams are decoded on the same one decoder thread but reformatted straight to
``gray8`` at a third of their size in one ``swscale`` call with ``SWS_AREA``, which is
the operation the C++ RoboCap reader performs.
"""
GroundTruthSource: TypeAlias = Literal["lighthouse", "mocap"]
"""How a segment's ground-truth rig poses were measured."""

TIER_BY_NAME: dict[str, Tier] = {"smoke": "smoke", "accuracy": "accuracy", "long": "long"}
"""Valid tier names, in increasing cost order; the lookup is also how a manifest string becomes a :data:`Tier`."""
DECODE_PATH_BY_NAME: dict[str, DecodePath] = {
    "cpu_gray8_dav1d_1thread": "cpu_gray8_dav1d_1thread",
    "cpu_gray8_swscale_area_downscale3": "cpu_gray8_swscale_area_downscale3",
}
"""Decode paths a gate may be frozen on."""
GT_SOURCE_BY_NAME: dict[str, GroundTruthSource] = {"lighthouse": "lighthouse", "mocap": "mocap"}
"""Ground-truth measurement systems the two MSD devices use."""
GATE_POLICY_BY_NAME: dict[str, GatePolicy] = {"tight": "tight", "standard": "standard", "no_divergence": "no_divergence"}
"""Valid gate policies, in decreasing strictness."""

def _one_of[LiteralName: str](value: object, allowed: dict[str, LiteralName], what: str, where: str) -> LiteralName:
    """Narrow one manifest string into its literal alphabet, or say what the alphabet is.

    Every one of these values is typed into the file by hand, so the message has
    to carry the alphabet: a typo used to get a helpful sentence or an unhelpful
    one depending on which of the four tables it was in.

    Args:
        value: The string the manifest carries.
        allowed: The identity table for the literal type, e.g. :data:`TIER_BY_NAME`.
        what: What the value names, for the error, e.g. ``"tier"``.
        where: Which segment or table it was read from, for the error.

    Returns:
        The same string, typed as the literal it is.

    Raises:
        ValueError: If it is not one of the alphabet.
    """
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{where}: unknown {what} {value!r}, expected one of {sorted(allowed)}")
    return allowed[value]


ATE_VS_CPP_CM: float = 2.0
"""Largest ATE RMSE against the basalt C++ trajectory the V2 gate accepts (D14's first rung)."""
PATH_BOUND_MAX_CLIP_S: float = 100.0
"""Longest replayed span :data:`ATE_VS_CPP_CM` is asked of (D60).

Past about a hundred seconds the C++ does not meet 2 cm against **itself**:
changing nothing but the floating-point width moves basalt's own `MIO14` (410 s)
trajectory by 4.24 cm rmse (S15 §1). A bound the reference cannot meet measures
the clip's length, not the port, so on longer clips the ground-truth band below
is the whole accuracy verdict.
"""
GT_BAND_RATIO: float = 1.2
"""How far outside the C++'s own precision band the port's ground-truth error may sit (D60).

The band is `[rmse_cm, rmse_cm_f64]`: the same C++ code on the same pixels with
one flag changed. It is 0.00007 cm wide on `MIO10` and 2.3 cm wide on `MIO14`, so
"inside the band" alone would gate the tight clips on rounding; the rule is
therefore this multiple of the band's worst member, which contains the band
itself.
"""
SPEED_TOLERANCE: float = 1.2
"""How much slower than the C++ single-thread wall on the same footage the port's replay may be (D58)."""
DIVERGENCE_FACTOR: float = 10.0
"""How much larger than the ground truth's extent a ``no_divergence`` run's may be (D36)."""
MIN_TRACKED_POSES: int = 10
"""Fewest poses that make a run a trajectory at all, rather than a comparison of noise."""
SMOKE_SEGMENTS: tuple[str, ...] = ("msd-g2__MGO_others__MGO09_short_1_updown", "msd-index__MIO_others__MIO10_short_2_panorama")
"""Both smoke clips, the four-camera 3 s one first: a broken machine says so sooner.

The ids live here rather than in the tool that runs them: the V2 gate, the fleet
tool and both suites name the same two clips, and a string spelled in four
modules is a manifest id nobody can rename.
"""


@dataclass(slots=True, frozen=True)
class ImuParameters:
    """Continuous-time IMU noise model and clock offset, in basalt's units.

    None of this is on the recordings; it comes from the device's own calibration
    file and is frozen here until dataforge logs it onto the IMU node.
    """

    rate_hz: float
    """Nominal IMU update rate."""
    gyro_noise_std: float
    """Gyroscope noise density, rad/s/sqrt(Hz)."""
    accel_noise_std: float
    """Accelerometer noise density, m/s^2/sqrt(Hz)."""
    gyro_bias_std: float
    """Gyroscope bias random walk, rad/s^2/sqrt(Hz)."""
    accel_bias_std: float
    """Accelerometer bias random walk, m/s^3/sqrt(Hz)."""
    cam_time_offset_ns: int
    """Added to a camera timestamp to reach the IMU clock; zero for MSD."""


@dataclass(slots=True, frozen=True)
class CaptureProperties:
    """What ``/__properties/capture`` reports for a segment."""

    duration_ns: int
    """Sensor time spanned by the segment."""
    num_frames: int
    """Frames per camera."""
    num_cameras: int
    """Cameras on the rig."""
    start_time_ns: int
    """Device-clock time of ``video_time`` zero; add it to reach the clock every basalt CSV uses."""


@dataclass(slots=True, frozen=True)
class LayerFingerprint:
    """What the catalog manifest reports about one registered layer.

    Pinning these turns a silent re-conversion into a failing test: the property
    columns alone would not notice a layer that grew, shrank or changed schema.
    """

    size_bytes: int
    """Registered size of the layer's ``.rrd``."""
    schema_sha256: str
    """Hex digest of the layer's schema, as ``DatasetEntry.manifest()`` reports it."""


@dataclass(slots=True, frozen=True)
class DatasetProperties:
    """The rig geometry and basalt config shared by every segment of one dataset.

    Calibration is byte-identical across a dataset's segments (33/33 for
    ``msd-index``, 15/15 for ``msd-g2``), so it is recorded once per dataset and
    asserted rather than assumed.
    """

    name: str
    """Catalog dataset name."""
    entry_id: str
    """Catalog entry id."""
    num_cameras: int
    """Cameras on the rig."""
    camera_resolution_wh: tuple[tuple[int, int], ...]
    """Per-camera ``(width, height)``, in rig order; the decoded array is ``(height, width)``."""
    image_rotation_cw_deg: tuple[int, ...]
    """Per-camera clockwise rotation already baked into the stored images and the calibration."""
    vio_config: Path
    """basalt VIO config this dataset's segments run with, relative to the manifest; vendored from the fork's ``data/msd/``."""


@dataclass(slots=True, frozen=True)
class GroundTruthProperties:
    """What ``/__properties/gt`` reports for a segment, on the ``gt`` layer."""

    num_poses: int
    """Ground-truth rig poses."""
    source: GroundTruthSource
    """Measurement system behind those poses."""


@dataclass(slots=True, frozen=True)
class CppAte:
    """The basalt C++ run's error against the ``gt.csv`` sidecar.

    The association is driven by the **estimate**: each basalt pose takes the
    nearest ground-truth pose within 5 ms, so ``associated`` counts estimate poses
    and ``total`` is the estimate's own length. Driving it the other way would
    pair ten 917 Hz truth poses to every 54 Hz estimate and weight the metric by
    truth density.
    """

    rmse_cm: float
    """Rigid-aligned RMSE, centimetres."""
    rmse_cm_f64: float
    """The same, from the same run in double precision: `use-double 1`, one thread, the same pixels.

    The other member of the C++'s own precision band (D60). One flag apart from
    :attr:`rmse_cm`, and the distance between the two is what basalt's answer is
    worth on that clip: 0.00007 cm on `MIO10`, 2.3 cm on the 410-second `MIO14`.
    """
    max_cm: float
    """Largest residual, centimetres."""
    median_cm: float
    """Median residual, centimetres."""
    associated: int
    """Estimate poses that found a truth pose within the tolerance."""
    total: int
    """Poses the C++ run produced, one per frameset."""


@dataclass(slots=True, frozen=True)
class CppReferenceRun:
    """The basalt C++ run this port is measured against, and how hard it may be gated."""

    gate_policy: GatePolicy
    """How hard this segment may be gated; see :data:`GatePolicy`."""
    fork_commit: str
    """Commit of the basalt fork that produced the run.

    The manifest also records ``fork_branch`` and ``fork_base`` beside it; the
    fork is machine-local and nothing here reads either, so they stay in the TOML
    as provenance rather than becoming fields.
    """
    decode_path: DecodePath
    """Decode path the run consumed, which must equal the segment's own."""
    deterministic: bool
    """``deterministic=1``: the tracker pops state in the calling loop instead of on a thread."""
    num_threads: int
    """``num-threads``: the TBB cap, so every reported wall time is single-thread."""
    use_double: bool
    """``use-double``: false, so the estimator ran in single precision."""
    vio_config: str
    """basalt VIO config the run used, relative to the fork."""
    optical_flow_image_safe_radius: float
    """``config.optical_flow_image_safe_radius`` for this device; 472 on Index, 340 on G2."""
    expected_cpp_wall_s: float
    """The C++ feed loop's wall for the whole segment: decode plus VIO, one thread, nothing logged."""
    hold_out: bool
    """True where tuning may not look at this segment (C56); the gate still runs it."""
    run_json: Path
    """Full run manifest, relative to the package root; always committed."""
    trajectory_sha256: str
    """Digest of ``basalt_traj.csv``, so a bundle copy can be checked."""
    bundle_only: bool
    """True when the trajectory is too large to check in and lives in the reference bundle."""
    trajectory_csv: Path
    """Trajectory path: relative to the package root when committed, to the bundle root when ``bundle_only``."""
    frames_sha256: Path | None
    """Per-frame pixel digests, relative to the package root; committed for the smoke pair only."""
    gt_csv_fixture: Path | None
    """Committed copy of the ``gt.csv`` sidecar, so the smoke gate runs offline."""
    expected_cpp_ate: CppAte
    """What the C++ run scored, reproduced by the fast and slow gate tests."""


@dataclass(slots=True, frozen=True)
class ReferenceSegment:
    """One frozen segment of the reference set."""

    dataset_name: str
    """Catalog dataset the segment belongs to."""
    dataset_entry_id: str
    """Catalog entry id of that dataset."""
    segment_id: str
    """Segment id, which is also the ``.rrd`` file stem on the NAS."""
    tier: Tier
    """How often the segment runs."""
    base_url: str
    """Storage URL of the ``base`` layer: video, IMU and calibration."""
    gt_url: str
    """Storage URL of the ``gt`` layer: the ground-truth rig poses."""
    gt_csv: Path
    """EuRoC-style ground-truth sidecar, absolute device-clock ns and a w-first quaternion."""
    decode_path: DecodePath
    """Frozen decode path that produced the gated pixels."""
    capture: CaptureProperties
    """Expected capture properties."""
    gt: GroundTruthProperties
    """Expected ground-truth properties."""
    layers: dict[str, LayerFingerprint]
    """Size and schema digest per layer name (``base``, ``gt``)."""
    reference: CppReferenceRun
    """The basalt C++ run this segment is measured against."""
    imu: ImuParameters
    """Frozen IMU noise model for this device."""

    @property
    def base_path(self) -> Path:
        """Local filesystem path behind :attr:`base_url`."""
        return Path(self.base_url.removeprefix("file://"))

    @property
    def gt_path(self) -> Path:
        """Local filesystem path behind :attr:`gt_url`."""
        return Path(self.gt_url.removeprefix("file://"))


@dataclass(slots=True, frozen=True)
class TrajectoryFixtures:
    """The checked-in basalt C++ outputs a gate test reproduces its numbers from."""

    golden: Path
    """Reference trajectory CSV, relative to the package root."""
    candidate: Path
    """Trajectory CSV under test, relative to the package root."""
    expected_ate_rmse_cm: float
    """ATE RMSE the pair is expected to produce, in centimetres."""
    expected_associated: int
    """Poses the association is expected to match."""


@dataclass(slots=True, frozen=True)
class RobocapSession:
    """One RoboCap recording session: where its two layers are and how long the C++ ran on it."""

    session_id: str
    """Recorder session, e.g. ``s00000015``."""
    segment_id: str
    """Segment id, which is also the ``.rrd`` file stem on the NAS."""
    base_url: str
    """Storage URL of the ``base`` layer: video, IMU and calibration."""
    slam_url: str
    """Storage URL of the ``slam`` layer, which holds the basalt C++ trajectory."""
    basalt_num_poses: int
    """Poses on the ``slam`` layer, which is also the session's complete frameset count."""
    expected_cpp_wall_s: float | None
    """The C++ feed loop's wall over the whole session, where one was measured; None where none was.

    Only session 15 has one, and it was measured on cap A rather than on an
    x86-64 host: 88.91 s for its 1,588 framesets at 30 fps with basalt's own
    Rerun logging on. The manifest comment names the machine, because — as with
    the MSD segments' own ``expected_cpp_wall_s`` — the ratio a row prints
    against it is a fact about the machine the row was measured on.
    """

    @property
    def fleet_id(self) -> str:
        """Short form a fleet row names this session by: ``robocap-s15`` for ``s00000015``."""
        return f"robocap-s{int(self.session_id.removeprefix('s'))}"

    @property
    def base_path(self) -> Path:
        """Local filesystem path behind :attr:`base_url`."""
        return Path(self.base_url.removeprefix("file://"))

    @property
    def slam_path(self) -> Path:
        """Local filesystem path behind :attr:`slam_url`."""
        return Path(self.slam_url.removeprefix("file://"))


@dataclass(slots=True, frozen=True)
class RobocapReference:
    """RoboCap: agreement with basalt C++ on a hard fisheye rig, with no ground truth.

    Everything the C++ lane was configured with lives here, because the port has
    to be fed the same configuration to be measured against it (C72): the four
    cameras of six, the downscale, and basalt's own calibration and VIO config
    files as the fork's converter and ``robocap_vit.toml`` produced them.

    The manifest's ``device_id`` — the capture device every session came from,
    which is what makes the calibration per device — stays in the TOML as
    provenance; nothing here reads it.
    """

    has_ground_truth: bool
    """Always false: RoboCap has no measured ground truth."""
    decode_path: DecodePath
    """Frozen decode path that produced the C++ pixels."""
    camera_names: tuple[str, ...]
    """The cameras the C++ ran, by their ``name`` static, in the calibration's own order."""
    downscale: int
    """Integer factor the C++ reader downscaled both frames and intrinsics by."""
    frameset_tolerance_ns: int
    """How far a camera's frame may sit from the anchor camera's and still be the same capture."""
    interpolate_accel_onto_gyro: bool
    """Whether the accelerometer has to be interpolated onto the gyroscope's timestamps."""
    video_time_is_absolute: bool
    """Whether ``video_time`` is already the device clock the C++ trajectories are on."""
    vio_config: str
    """basalt VIO config the C++ ran, relative to the package root."""
    calibration: str
    """basalt calibration the C++ ran, at :attr:`downscale`, relative to the package root."""
    imu: ImuParameters
    """Frozen IMU noise model, from the device's Kalibr calibration."""
    sessions: tuple[RobocapSession, ...]
    """The measured sessions, in manifest order."""
    fixtures: TrajectoryFixtures
    """Checked-in basalt outputs for session 15."""

    def session(self, session_id: str) -> RobocapSession:
        """The session with this id.

        Raises:
            ValueError: If the manifest has no such session.
        """
        for session in self.sessions:
            if session.session_id == session_id:
                return session
        raise ValueError(f"{session_id!r} is not a RoboCap session in the manifest; have {[s.session_id for s in self.sessions]}")


@dataclass(slots=True, frozen=True)
class ReferenceManifest:
    """The whole reference set."""

    schema_version: int
    """Manifest layout version; bumped when a field changes meaning."""
    catalog_url: str
    """Catalog the MSD properties were read from, and that the slow test checks against."""
    datasets: tuple[DatasetProperties, ...]
    """Rig geometry, one entry per catalog dataset the segments come from."""
    segments: tuple[ReferenceSegment, ...]
    """The ten MSD segments, in tier-then-dataset order."""
    robocap: RobocapReference
    """The RoboCap third reference."""
    package_root: Path
    """Directory the manifest was read from; fixture paths are relative to it."""

    def dataset(self, name: str) -> DatasetProperties:
        """The dataset with this name.

        Raises:
            ValueError: If the manifest has no such dataset.
        """
        for dataset in self.datasets:
            if dataset.name == name:
                return dataset
        raise ValueError(f"{name!r} is not in the reference set; have {[d.name for d in self.datasets]}")

    def vio_config_text(self, dataset_name: str, profile: str = "reference") -> str:
        """The basalt VIO config one dataset's segments run with, as its file's own text.

        Args:
            dataset_name: Catalog dataset name.
            profile: Named config overlay; reference preserves the original text.

        Returns:
            The overlaid JSON (original file text for reference), ready for :meth:`slam_rs._core.VioConfig.from_json`.

        Raises:
            ValueError: If the manifest has no such dataset.
            KeyError: If an overlay key is absent from the vendored config.
        """
        path: Path = self.package_root / self.dataset(dataset_name).vio_config
        return profiled_config_text(path, profile, path.parent / "profiles")

    def by_id(self, segment_id: str) -> ReferenceSegment:
        """The segment with this id.

        A command line is what reaches this — ``fleet_check --segments`` and the
        replay tool's ``--segment`` — so an id the manifest cannot satisfy is a
        ``ValueError`` naming the ten it has, the same kind of answer
        :func:`load_manifest` gives for a manifest it cannot read. A bare
        ``KeyError`` reads as a dictionary miss.

        Raises:
            ValueError: If no segment in the manifest has that id.
        """
        for segment in self.segments:
            if segment.segment_id == segment_id:
                return segment
        raise ValueError(f"{segment_id!r} is not in the reference set; have {[s.segment_id for s in self.segments]}")

    def in_tier(self, tier: Tier) -> tuple[ReferenceSegment, ...]:
        """Every segment in one tier, in manifest order."""
        return tuple(segment for segment in self.segments if segment.tier == tier)

    def cpp_trajectory(self, segment: ReferenceSegment) -> BundleFile:
        """Where one segment's basalt C++ trajectory is, and why it cannot be read when it cannot.

        Eight of the ten are committed beside the manifest and are always there;
        the two long-tier ones are too large for the history and live in the
        machine-local reference bundle (:mod:`slam_rs.reference_bundle`), so the
        answer is the same shape either way and a caller can skip with a reason.

        Args:
            segment: The segment whose reference trajectory is wanted.

        Returns:
            The path it occupies, and why it is unusable if it is missing.
        """
        if segment.reference.bundle_only:
            return reference_bundle.resolve(segment.segment_id, reference_bundle.TRAJECTORY_CSV)
        path: Path = self.package_root / segment.reference.trajectory_csv
        reason: str | None = None if path.is_file() else f"{path} is committed in the manifest but missing from this checkout"
        return BundleFile(path=path, reason=reason)


def band_cm_text(band: tuple[float, float]) -> str:
    """The C++'s own precision band as a row prints it, in centimetres.

    Args:
        band: The C++'s ``f32`` and ``f64`` ground-truth RMSE in centimetres.

    Returns:
        The band, both members labelled by the precision that produced them.
    """
    return f"[f32 {band[0]:.2f}, f64 {band[1]:.2f}]"


def pose_floor_text(*, tracked: int, framesets: int) -> str:
    """D60's tracked-pose floor as a row prints it: a run this short is not a trajectory.

    The one sentence for the V2 gate and for a fleet row, because a machine that
    tracked nothing is what the fleet lane exists to find and the two must not
    report it differently.

    Args:
        tracked: Poses the estimator reported.
        framesets: Framesets it was fed.

    Returns:
        The clause, ready for a verdict.
    """
    return f"{tracked} poses over {framesets} framesets is not a trajectory"


def d60_failures(
    *,
    gate_policy: GatePolicy,
    framesets: int,
    tracked: int,
    lost: int,
    associated: int,
    replayed_s: float,
    cpp_rmse_cm: float,
    gt_rmse_cm: float,
    band: tuple[float, float],
    extent_m: float,
    truth_extent_m: float,
    poses_finite: bool,
    band_text: str | None = None,
) -> list[str]:
    """Every D60 accuracy clause one replayed clip misses, in the order D60 states them.

    D60's rule is not "two error bounds": each bound is conditional, and the two
    conditions are what a second implementation gets wrong. The 2 cm path bound
    applies only under :data:`PATH_BOUND_MAX_CLIP_S` seconds, because past that
    the C++ does not meet it against its own other precision (`MIO14`: 4.24 cm).
    A ``no_divergence`` clip gates neither error at all — basalt itself sits at
    43 cm and 78 cm there and two of its own decode paths differ by 18 to 32 cm —
    and gates a bounded run instead. Finite poses are asked of every policy: an
    error against a non-finite estimate is NaN, and NaN passes every bound
    below. Both the V2 gate and
    :mod:`slam_rs.apis.fleet_check` read the verdict off these clauses, so they
    are written once: a fleet row that applied the bounds unconditionally called
    a healthy machine broken on any clip but the two smoke ones.

    The tracked-pose floor is the first clause and the only one that stands
    alone: a run below it was never scored, so it has no error to report and the
    rest of D60 has nothing to read.

    Speed is **not** here. It is a clause of the gate (D58) and a fact about the
    machine on a fleet row, because the C++ wall was measured on one host.

    Args:
        gate_policy: How hard D60 lets this clip be gated.
        framesets: Framesets fed to the estimator.
        tracked: Poses the estimator reported; below :data:`MIN_TRACKED_POSES`
            nothing else is read, because a run that short was never scored.
        lost: Framesets that never got the inertial samples covering them (D17).
        associated: Estimate poses that found a C++ pose inside the tolerance.
        replayed_s: Sensor seconds the estimate spans, whole clip or window.
        cpp_rmse_cm: ATE against the basalt C++ trajectory on the same footage.
        gt_rmse_cm: ATE against the ``gt.csv`` sidecar.
        band: The C++'s own ground-truth error in its ``f32`` and ``f64`` precisions.
        extent_m: Diagonal of the estimate's bounding box.
        truth_extent_m: The same for the ground truth, which bounds a ``no_divergence`` run.
        poses_finite: Whether every estimated position is finite.
        band_text: How the band prints; :func:`band_cm_text` when a caller has no
            labelled form of its own (a windowed gate row does).

    Returns:
        One line per missed clause; empty when the clip passes.
    """
    # First, and alone: :func:`slam_rs.trajectory.ate` needs a pose to align, so
    # a caller below the floor has no error to hand over and neither error is a
    # number. Both callers therefore stop at the floor as well.
    if tracked < MIN_TRACKED_POSES:
        return [pose_floor_text(tracked=tracked, framesets=framesets)]
    failures: list[str] = []
    if lost:
        failures.append(f"{lost} of {framesets} framesets never got the inertial samples that cover them")
    if associated < MIN_ASSOCIATED_POSES:
        failures.append(f"only {associated} poses associated with the C++ run")
    # Every policy and not only ``no_divergence``: a NaN error passes every ``>``
    # comparison there is, so a diverged run measured under ``tight`` or
    # ``standard`` came back with no failure at all and a fleet row printed it as
    # ``pass`` (S25 review). A run whose poses are not numbers has missed every
    # clause D60 has, whichever policy the clip carries.
    if not poses_finite:
        failures.append("a pose is not finite")
    if gate_policy == "no_divergence":
        # basalt itself is near failure here, so only a bounded run is asserted.
        if extent_m > DIVERGENCE_FACTOR * truth_extent_m:
            failures.append(f"spans {extent_m:.1f} m against the truth's {truth_extent_m:.1f} m")
        return failures
    # The path bound only where the C++ meets it itself: past about a hundred
    # seconds its own two precisions are 4.24 cm apart, so 2 cm there would gate
    # the clip's length (D60).
    if replayed_s < PATH_BOUND_MAX_CLIP_S and cpp_rmse_cm > ATE_VS_CPP_CM:
        failures.append(f"{cpp_rmse_cm:.2f} cm from the C++ trajectory, gate is {ATE_VS_CPP_CM:.0f} cm")
    # Inside the C++'s own band, or within GT_BAND_RATIO of its worst member,
    # whichever is looser — which is the second alone, because the ratio is above
    # one and the band's worst member is its upper end.
    allowed_cm: float = GT_BAND_RATIO * max(band)
    if gt_rmse_cm > allowed_cm:
        failures.append(
            f"{gt_rmse_cm:.2f} cm from ground truth, gate is {GT_BAND_RATIO}x the worst of the "
            f"C++'s own band {band_text if band_text is not None else band_cm_text(band)} cm = {allowed_cm:.2f} cm"
        )
    return failures


PORT_CONFIG_KEYS: frozenset[str] = frozenset({"port.redetect_survivor_ratio"})
"""Overlay keys the port adds to basalt's document, which no vendored config carries.

``configs/*.json`` are the files the C++ reference runs read, key for key
(``tests/test_cpp_reference.py``), so a knob basalt has no field for is never
written into them: it is spelled ``port.`` instead of ``config.``, an overlay
inserts it, and :class:`slam_rs._core.VioConfig` models it with a default that
reproduces basalt's behaviour for every path that does not. Listing them here
keeps :func:`profiled_config_text`'s typo check: a key that is neither in the
base document nor in this set is still a ``KeyError``.
"""


def profiled_config_text(path: Path, profile: str = "reference", profiles: Path = MANIFEST_PATH.parent / "configs/profiles") -> str:
    """Read a config and apply a named overlay; empty overlays preserve its text.

    Args:
        path: Base config JSON.
        profile: Overlay file stem.
        profiles: Directory holding the flat config-key overlays.

    Returns:
        Config JSON with the overlay applied.

    Raises:
        KeyError: If an overlay key is neither in the base value0 namespace nor
            one of :data:`PORT_CONFIG_KEYS`.
    """
    text: str = path.read_text()
    overlay: dict = json.loads((profiles / f"{profile}.json").read_text())
    if not overlay:
        return text
    document: dict = json.loads(text)
    values: dict = document["value0"]
    for key in overlay:
        if key not in values and key not in PORT_CONFIG_KEYS:
            raise KeyError(key)
    values.update(overlay)
    return json.dumps(document)


def flow_config(manifest: ReferenceManifest, segment: ReferenceSegment, profile: Literal["reference", "fast"] = "reference") -> _core.VioConfig:
    """The basalt config the C++ reference ran this segment's dataset with.

    basalt's constructor defaults are not its shipped files: ``msdmi_config.json``
    and ``msdmg_config.json`` set ``vio_marg_lost_landmarks`` to true where the
    constructor says false (``crates/slam-rs/src/config.rs:25``, pinned by that
    crate's own test). That one key is a different estimator — with the
    constructor's defaults the port sat 1.41 to 12.05 cm from the C++ on the
    reference clips, with the dataset's config 0.31 to 5.19 cm (C72) — so the
    file is read rather than reconstructed.

    An explicit speed profile overlays its keys. The manifest's per-device
    ``optical_flow_image_safe_radius`` is asserted against the file instead, so a
    manifest and a config that disagree stop the run rather than one of them
    silently winning. The replay tool and the V2 gate both build their estimator
    from here, so neither can drive a segment with another one's configuration.

    Args:
        manifest: The reference set the segment came from, which resolves the
            dataset's config file.
        segment: The segment about to be replayed.
        profile: Config overlay; reference preserves the C++ configuration.

    Returns:
        The config to build an estimator or a frontend for that segment with.

    Raises:
        ValueError: If the file's image safe radius is not the one the manifest
            froze for this segment.
    """
    config: _core.VioConfig = _core.VioConfig.from_json(manifest.vio_config_text(segment.dataset_name, profile=profile))
    frozen: float = segment.reference.optical_flow_image_safe_radius
    if config.optical_flow_image_safe_radius != frozen:
        raise ValueError(
            f"{segment.segment_id}: {manifest.dataset(segment.dataset_name).vio_config} sets "
            f"optical_flow_image_safe_radius = {config.optical_flow_image_safe_radius}, the manifest freezes {frozen}"
        )
    return config


def _imu(block: dict[str, Any]) -> ImuParameters:
    """One ``[*.imu]`` table."""
    return ImuParameters(
        rate_hz=float(block["rate_hz"]),
        gyro_noise_std=float(block["gyro_noise_std"]),
        accel_noise_std=float(block["accel_noise_std"]),
        gyro_bias_std=float(block["gyro_bias_std"]),
        accel_bias_std=float(block["accel_bias_std"]),
        cam_time_offset_ns=int(block["cam_time_offset_ns"]),
    )


def _reference_run(block: dict[str, Any], segment_id: str) -> CppReferenceRun:
    """One ``[segment.reference]`` table.

    Raises:
        ValueError: If the gate policy or decode path is unknown.
    """
    ate_block: dict[str, Any] = block["expected_cpp_ate"]
    return CppReferenceRun(
        gate_policy=_one_of(block["gate_policy"], GATE_POLICY_BY_NAME, "gate policy", segment_id),
        fork_commit=block["fork_commit"],
        decode_path=_one_of(block["decode_path"], DECODE_PATH_BY_NAME, "decode path", f"{segment_id} reference run"),
        deterministic=bool(block["deterministic"]),
        num_threads=int(block["num_threads"]),
        use_double=bool(block["use_double"]),
        vio_config=block["vio_config"],
        optical_flow_image_safe_radius=float(block["optical_flow_image_safe_radius"]),
        expected_cpp_wall_s=float(block["expected_cpp_wall_s"]),
        # Absent on the eight segments tuning may look at; C56 named the two.
        hold_out=bool(block.get("hold_out", False)),
        run_json=Path(block["run_json"]),
        trajectory_sha256=block["trajectory_sha256"],
        bundle_only=bool(block["bundle_only"]),
        trajectory_csv=Path(block["trajectory_csv"]),
        frames_sha256=Path(block["frames_sha256"]) if "frames_sha256" in block else None,
        gt_csv_fixture=Path(block["gt_csv_fixture"]) if "gt_csv_fixture" in block else None,
        expected_cpp_ate=CppAte(
            rmse_cm=float(ate_block["rmse_cm"]),
            rmse_cm_f64=float(ate_block["rmse_cm_f64"]),
            max_cm=float(ate_block["max_cm"]),
            median_cm=float(ate_block["median_cm"]),
            associated=int(ate_block["associated"]),
            total=int(ate_block["total"]),
        ),
    )


def _robocap(robocap_block: dict[str, Any]) -> RobocapReference:
    """The `[robocap]` table: the one rig whose reference is a C++ trajectory, not ground truth.

    Args:
        robocap_block: The parsed ``[robocap]`` table.

    Returns:
        What the C++ lane ran, so one place says it and the tools only read it.

    Raises:
        KeyError: If a key is missing; `load_manifest` names the table around it.
        ValueError: If the decode path is unknown.
    """
    fixtures_block: dict[str, Any] = robocap_block["fixtures"]
    return RobocapReference(
        has_ground_truth=bool(robocap_block["has_ground_truth"]),
        decode_path=_one_of(robocap_block["decode_path"], DECODE_PATH_BY_NAME, "decode path", "robocap"),
        camera_names=tuple(str(name) for name in robocap_block["camera_names"]),
        downscale=int(robocap_block["downscale"]),
        frameset_tolerance_ns=int(robocap_block["frameset_tolerance_ns"]),
        interpolate_accel_onto_gyro=bool(robocap_block["interpolate_accel_onto_gyro"]),
        video_time_is_absolute=bool(robocap_block["video_time_is_absolute"]),
        vio_config=str(robocap_block["vio_config"]),
        calibration=str(robocap_block["calibration"]),
        imu=_imu(robocap_block["imu"]),
        sessions=tuple(
            RobocapSession(
                session_id=block["session_id"],
                segment_id=block["segment_id"],
                base_url=block["base_url"],
                slam_url=block["slam_url"],
                basalt_num_poses=int(block["basalt_num_poses"]),
                # Absent on every session but the one whose C++ wall was measured.
                expected_cpp_wall_s=float(block["expected_cpp_wall_s"]) if "expected_cpp_wall_s" in block else None,
            )
            for block in robocap_block["session"]
        ),
        fixtures=TrajectoryFixtures(
            golden=Path(fixtures_block["golden"]),
            candidate=Path(fixtures_block["candidate"]),
            expected_ate_rmse_cm=float(fixtures_block["expected_ate_rmse_cm"]),
            expected_associated=int(fixtures_block["expected_associated"]),
        ),
    )


ARTIFACT_NAMES: tuple[str, ...] = ("base.rrd", "gt.rrd", "gt.csv", "slam.rrd")
"""What a relocated segment's directory holds; a segment has three of the four."""


def relocate(manifest: ReferenceManifest, root: Path) -> ReferenceManifest:
    """The same reference set with every artifact read from one directory per segment.

    The manifest's URLs are absolute paths on the NAS the corpus was converted
    on. A machine that is not that NAS holds a copy somewhere else, and used to
    say so with a shell ``sed`` over a 28 KB manifest copy per machine — one
    prefix for MSD and a different one for RoboCap, which is how the S20 probe
    found the base layer locally and then went to the NAS for the ``slam`` one.

    The layout is ``<root>/<segment_id>/`` holding
    :data:`ARTIFACT_NAMES`. The names are the layer's, not the file's, because
    a segment's ``base`` and ``gt`` layers are both stored as ``<segment_id>.rrd``
    and would collide in one directory.

    Nothing else moves: the thresholds, the C++ numbers, the vendored configs and
    the checked-in fixtures are the manifest and not the mount, and they stay
    relative to :attr:`ReferenceManifest.package_root`.

    Args:
        manifest: The parsed reference set.
        root: Directory the per-segment directories sit in.

    Returns:
        The same manifest against the corpus under ``root``.
    """
    segments: tuple[ReferenceSegment, ...] = tuple(
        replace(
            segment,
            base_url=f"file://{root / segment.segment_id / 'base.rrd'}",
            gt_url=f"file://{root / segment.segment_id / 'gt.rrd'}",
            gt_csv=root / segment.segment_id / "gt.csv",
        )
        for segment in manifest.segments
    )
    sessions: tuple[RobocapSession, ...] = tuple(
        replace(
            session,
            base_url=f"file://{root / session.segment_id / 'base.rrd'}",
            slam_url=f"file://{root / session.segment_id / 'slam.rrd'}",
        )
        for session in manifest.robocap.sessions
    )
    return replace(manifest, segments=segments, robocap=replace(manifest.robocap, sessions=sessions))


def load_manifest(path: Path = MANIFEST_PATH, artifact_root: Path | None = None) -> ReferenceManifest:
    """Parse the reference manifest.

    Args:
        path: Manifest file; defaults to the copy checked in beside the package.
        artifact_root: Read every recording and sidecar from under this
            directory instead of the absolute NAS paths the manifest carries;
            see :func:`relocate`. None reads them where the manifest says.

    Returns:
        The parsed manifest, with tiers, decode paths and ground-truth sources
        validated against their literal alphabets.

    Raises:
        ValueError: If a tier, decode path or ground-truth source is unknown, a
            segment id repeats, or the ``[robocap]`` table is absent or missing a key.
    """
    document: dict[str, Any] = tomllib.loads(path.read_text())
    datasets: list[DatasetProperties] = []
    for entry in document["dataset"]:
        resolutions: tuple[tuple[int, int], ...] = tuple((int(pair[0]), int(pair[1])) for pair in entry["camera_resolution_wh"])
        rotations: tuple[int, ...] = tuple(int(value) for value in entry["image_rotation_cw_deg"])
        if len(resolutions) != entry["num_cameras"] or len(rotations) != entry["num_cameras"]:
            raise ValueError(f"{entry['name']}: {entry['num_cameras']} cameras but {len(resolutions)} resolutions and {len(rotations)} rotations")
        datasets.append(
            DatasetProperties(
                name=entry["name"],
                entry_id=entry["entry_id"],
                num_cameras=int(entry["num_cameras"]),
                camera_resolution_wh=resolutions,
                image_rotation_cw_deg=rotations,
                vio_config=Path(entry["vio_config"]),
            )
        )
    segments: list[ReferenceSegment] = []
    for entry in document["segment"]:
        identifier: str = entry["segment_id"]
        tier: Tier = _one_of(entry["tier"], TIER_BY_NAME, "tier", identifier)
        decode_path: DecodePath = _one_of(entry["decode_path"], DECODE_PATH_BY_NAME, "decode path", identifier)
        source: GroundTruthSource = _one_of(entry["gt"]["source"], GT_SOURCE_BY_NAME, "ground-truth source", identifier)
        segments.append(
            ReferenceSegment(
                dataset_name=entry["dataset_name"],
                dataset_entry_id=entry["dataset_entry_id"],
                segment_id=entry["segment_id"],
                tier=tier,
                base_url=entry["base_url"],
                gt_url=entry["gt_url"],
                gt_csv=Path(entry["gt_csv"]),
                decode_path=decode_path,
                capture=CaptureProperties(
                    duration_ns=int(entry["capture"]["duration_ns"]),
                    num_frames=int(entry["capture"]["num_frames"]),
                    num_cameras=int(entry["capture"]["num_cameras"]),
                    start_time_ns=int(entry["capture"]["start_time_ns"]),
                ),
                gt=GroundTruthProperties(num_poses=int(entry["gt"]["num_poses"]), source=source),
                layers={
                    name: LayerFingerprint(size_bytes=int(block["size_bytes"]), schema_sha256=block["schema_sha256"])
                    for name, block in entry["layers"].items()
                },
                reference=_reference_run(entry["reference"], identifier),
                imu=_imu(entry["imu"]),
            )
        )
    identifiers: list[str] = [segment.segment_id for segment in segments]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"duplicate segment ids in {path}: {sorted({i for i in identifiers if identifiers.count(i) > 1})}")

    if "robocap" not in document:
        raise ValueError(f"{path}: the manifest has no [robocap] table")
    try:
        robocap: RobocapReference = _robocap(document["robocap"])
    except KeyError as missing:
        raise ValueError(f"{path}: the [robocap] table is missing the key {missing}") from missing
    parsed: ReferenceManifest = ReferenceManifest(
        schema_version=int(document["schema_version"]),
        catalog_url=document["catalog_url"],
        datasets=tuple(datasets),
        segments=tuple(segments),
        robocap=robocap,
        package_root=path.parent,
    )
    return parsed if artifact_root is None else relocate(parsed, artifact_root)
