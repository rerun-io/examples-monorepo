"""Catalog replay metadata, dataset configuration, and ground-truth gate rules."""

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from tomllib import TOMLDecodeError
from pathlib import Path
from typing import Literal, TypeAlias

from serde import SerdeError, coerce, field, serde
from serde.toml import from_toml

from slam_rs import _core
from slam_rs.trajectory import MIN_ASSOCIATED_POSES

MANIFEST_PATH: Path = Path(__file__).resolve().parents[1] / "gate.toml"
"""The checked-in gate, beside the package rather than inside it."""

Tier: TypeAlias = Literal["smoke", "release", "listed"]
"""Smoke checks, release checks, or additional listed segments."""
DecodePath: TypeAlias = Literal["cpu_gray8_dav1d_1thread", "cpu_gray8_swscale_area_downscale3"]
"""Frozen pixel provenance.

The MSD gate is frozen on the single-threaded dav1d ``gray8`` path (D28). RoboCap's
H.264 streams are decoded on the same one decoder thread but reformatted straight to
``gray8`` at a third of their size in one ``swscale`` call with ``SWS_AREA``, which is
the combined area-resampling operation selected for RoboCap.
"""
TIER_BY_NAME: dict[str, Tier] = {"smoke": "smoke", "release": "release", "listed": "listed"}
"""Valid tier names, in increasing cost order; the lookup is also how a manifest string becomes a :data:`Tier`."""
DECODE_PATH_BY_NAME: dict[str, DecodePath] = {
    "cpu_gray8_dav1d_1thread": "cpu_gray8_dav1d_1thread",
    "cpu_gray8_swscale_area_downscale3": "cpu_gray8_swscale_area_downscale3",
}
"""Decode paths a gate may be frozen on."""
GATE_RATIO: float = 1.10
"""Largest allowed ratio to a matched accuracy or speed baseline."""
DIVERGENCE_FACTOR: float = 10.0
"""Largest allowed ratio of estimated extent to ground-truth extent."""
MIN_TRACKED_POSES: int = 10
"""Fewest poses that make a run a trajectory at all, rather than a comparison of noise."""
SMOKE_SEGMENTS: tuple[str, ...] = ("msd-g2__MGO_others__MGO09_short_1_updown", "msd-index__MIO_others__MIO10_short_2_panorama")
"""Both smoke clips, the four-camera 3 s one first: a broken machine says so sooner.

The ids live here rather than in the tool that runs them: the V2 gate, the fleet
tool and both suites name the same two clips, and a string spelled in four
modules is a manifest id nobody can rename.
"""


@serde(type_check=coerce, deny_unknown_fields=True)
@dataclass(slots=True, frozen=True)
class ImuParameters:
    """Continuous-time IMU noise model and clock offset, in the estimator's units.

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


@serde(type_check=coerce, deny_unknown_fields=True)
@dataclass(slots=True, frozen=True)
class DatasetProperties:
    """The sensor model and VIO config shared by every segment of one dataset.

    The catalog supplies geometry; the gate supplies the sensor noise model.
    """

    name: str
    """Catalog dataset name."""
    imu: ImuParameters
    """Noise model from the rig calibration."""
    vio_config: Path
    """Dataset configuration path, relative to the manifest."""


@serde(type_check=coerce, deny_unknown_fields=True)
@dataclass(slots=True, frozen=True)
class Baseline:
    """A measured reference for one profile and lane."""

    profile: Literal["fast", "reference"]
    """Configuration overlay."""
    lane: Literal["gpu", "cpu"]
    """Execution lane."""
    host: str
    """Host on which tracker time was measured."""
    core_sha256: str
    """Digest of the measured extension."""
    framesets: int
    """Framesets processed."""
    gt_rmse_cm: float
    """ATE against ground truth, in centimetres."""
    median_tracker_ms: float
    """Median accepted tracker call duration."""
    measured_on: str
    """Measurement date."""


@serde(type_check=coerce, deny_unknown_fields=True)
@dataclass(slots=True, frozen=True)
class ReferenceSegment:
    """One frozen segment of the reference set."""

    dataset_name: str
    """Catalog dataset the segment belongs to."""
    segment_id: str
    """Segment id, used to locate the recording on the catalog."""
    tier: Tier
    """How often the segment runs."""
    decode_path: DecodePath
    """Frozen decode path that produced the gated pixels."""
    hold_out: bool = False
    """Whether this segment is excluded from tuning."""
    baseline: tuple[Baseline, ...] = ()
    """Measurements by profile and execution lane."""

    def baseline_for(self, lane: Literal["cpu", "gpu"], profile: Literal["reference", "fast"]) -> Baseline | None:
        """Return the unique baseline for this execution lane and profile."""
        return next((row for row in self.baseline if row.lane == lane and row.profile == profile), None)


@serde(type_check=coerce, deny_unknown_fields=True)
@dataclass(slots=True, frozen=True)
class RobocapSession:
    """One catalog session and its optional regression reference."""

    session_id: str
    """Recorder session, e.g. ``s00000015``."""
    segment_id: str
    """Segment id, used to locate the recording on the catalog."""

    reference_csv: Path | None = None
    """Optional regression trajectory, relative to the package."""

    @property
    def fleet_id(self) -> str:
        """Short form a fleet row names this session by: ``robocap-s15`` for ``s00000015``."""
        return f"robocap-s{int(self.session_id.removeprefix('s'))}"


@serde(type_check=coerce, deny_unknown_fields=True)
@dataclass(slots=True, frozen=True)
class RobocapReference:
    """RoboCap rig parameters. This dataset has no ground truth."""

    device_id: str
    """The device whose sessions the catalog holds: segment ids are ``robocap__<device_id>__<session_id>``."""
    has_ground_truth: bool
    """Always false: RoboCap has no measured ground truth."""
    decode_path: DecodePath
    """Frozen decode path that produced the reference pixels."""
    camera_names: tuple[str, ...]
    """The cameras the reference ran, by their ``name`` static, in the calibration's own order."""
    downscale: int
    """Integer factor the reference reader downscaled both frames and intrinsics by."""
    frameset_tolerance_ns: int
    """How far a camera's frame may sit from the anchor camera's and still be the same capture."""
    interpolate_accel_onto_gyro: bool
    """Whether the accelerometer has to be interpolated onto the gyroscope's timestamps."""
    video_time_is_absolute: bool
    """Whether ``video_time`` is already the device clock the reference trajectories are on."""
    vio_config: str
    """VIO configuration selected for replay, relative to the package root."""
    calibration: str
    """Rig calibration selected for replay, at :attr:`downscale`, relative to the package root."""
    imu: ImuParameters
    """Frozen IMU noise model, from the device's Kalibr calibration."""
    sessions: tuple[RobocapSession, ...] = field(rename="session")
    """The measured sessions, in manifest order."""

    def session(self, session_id: str) -> RobocapSession:
        """The session with this id: the listed one, or any other session of this device on the catalog.

        An unlisted session has no regression reference; the tools report it unscored.

        Raises:
            ValueError: If the id is not shaped like a RoboCap session id.
        """
        for session in self.sessions:
            if session.session_id == session_id:
                return session
        if re.fullmatch(r"s\d{8}", session_id) is None:
            raise ValueError(
                f"{session_id!r} is not a RoboCap session id (expected s00000015-style); listed: {[s.session_id for s in self.sessions]}"
            )
        return RobocapSession(reference_csv=None, session_id=session_id, segment_id=f"robocap__{self.device_id}__{session_id}")

    def is_listed(self, session_id: str) -> bool:
        return any(session.session_id == session_id for session in self.sessions)


@serde(type_check=coerce, deny_unknown_fields=True)
@dataclass(slots=True, frozen=True)
class ReferenceManifest:
    """The whole reference set."""

    schema_version: int
    """Manifest layout version; bumped when a field changes meaning."""
    catalog_url: str
    """Catalog used to resolve the gate segments."""
    datasets: tuple[DatasetProperties, ...] = field(rename="dataset")
    """Rig geometry, one entry per catalog dataset the segments come from."""
    segments: tuple[ReferenceSegment, ...] = field(rename="segment")
    """The ten MSD segments, in tier-then-dataset order."""
    robocap: RobocapReference
    """The RoboCap third reference."""
    package_root: Path = field(skip=True, default=MANIFEST_PATH.parent, compare=False)
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
        """The VIO config one dataset's segments run with, as its file's own text.

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


def pose_floor_text(*, tracked: int, framesets: int) -> str:
    """Pose floor text."""
    return f"{tracked} poses over {framesets} framesets is not a trajectory"


PORT_CONFIG_KEYS: frozenset[str] = frozenset({"port.redetect_survivor_ratio", "port.frame_update_max_iterations"})
"""Additional port configuration keys accepted in profile overlays."""


def config_text_sha256(text: str) -> str:
    """Identify the resolved configuration without changing its serialization.

    Args:
        text: The exact text returned by :func:`profiled_config_text`.

    Returns:
        The hexadecimal SHA-256 of the text encoded as UTF-8.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def resolved_flow_config(
    manifest: ReferenceManifest, segment: ReferenceSegment, profile: Literal["reference", "fast"] = "reference"
) -> tuple[_core.VioConfig, str]:
    """Read the dataset configuration and apply the requested profile."""
    text: str = manifest.vio_config_text(segment.dataset_name, profile=profile)
    config: _core.VioConfig = _core.VioConfig.from_json(text)
    return config, text


def load_manifest(path: Path = MANIFEST_PATH) -> ReferenceManifest:
    """Deserialize the gate and validate relationships and finite baselines."""
    try:
        parsed: ReferenceManifest = from_toml(ReferenceManifest, path.read_text())
    except (SerdeError, TOMLDecodeError) as error:
        raise ValueError(f"{path}: {error}") from error
    if parsed.schema_version != 10:
        raise ValueError(f"{path}: expected schema_version 10")
    identifiers: set[str] = set()
    dataset_names: set[str] = {dataset.name for dataset in parsed.datasets}
    for segment in parsed.segments:
        where: str = f"{path}: [segment] {segment.segment_id}"
        if segment.segment_id in identifiers:
            raise ValueError(f"{where}: duplicate segment ids")
        identifiers.add(segment.segment_id)
        if segment.dataset_name not in dataset_names:
            raise ValueError(f"{where}: unknown dataset {segment.dataset_name!r}")
        baseline_keys: set[tuple[str, str]] = set()
        for baseline in segment.baseline:
            key: tuple[str, str] = (baseline.profile, baseline.lane)
            if key in baseline_keys:
                raise ValueError(f"{where}: [segment.baseline] duplicate baseline {key}")
            baseline_keys.add(key)
            for name, value in (
                ("gt_rmse_cm", baseline.gt_rmse_cm),
                ("median_tracker_ms", baseline.median_tracker_ms),
                ("framesets", baseline.framesets),
            ):
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError(f"{where}: [segment.baseline] {key} {name} must be finite and positive")
    return replace(parsed, package_root=path.parent)


@dataclass(slots=True, frozen=True)
class Measurement:
    """Measured inputs to the tracking, accuracy, and speed clauses."""

    framesets: int
    """Framesets fed."""
    tracked: int
    """Estimated poses."""
    lost: int
    """Framesets left waiting for IMU."""
    associated: int
    """Estimate poses associated with ground truth."""
    gt_rmse_cm: float
    """Ground-truth ATE in centimetres."""
    extent_m: float
    """Estimated bounding-box diagonal."""
    truth_extent_m: float
    """Ground-truth bounding-box diagonal."""
    poses_finite: bool
    """Whether estimated positions are finite."""
    median_tracker_ms: float
    """Median accepted tracker call duration."""
    hostname: str
    """Measuring host."""
    lane: Literal["cpu", "gpu"]
    """Execution lane."""
    profile: Literal["reference", "fast"]
    """Configuration overlay."""


def gate_failures(measurement: Measurement, baseline: Baseline | None) -> list[str]:
    """Return failed clauses; the caller supplies the matched lane/profile baseline."""
    if baseline is not None and (baseline.lane, baseline.profile) != (measurement.lane, measurement.profile):
        raise ValueError(f"baseline {baseline.lane}/{baseline.profile} does not match measurement {measurement.lane}/{measurement.profile}")
    failures: list[str] = []
    if measurement.tracked < MIN_TRACKED_POSES:
        failures.append(f"tracked: {pose_floor_text(tracked=measurement.tracked, framesets=measurement.framesets)}")
    if measurement.lost != 0:
        failures.append(f"lost: {measurement.lost} of {measurement.framesets} framesets")
    if measurement.associated < MIN_ASSOCIATED_POSES:
        failures.append(f"associated: only {measurement.associated} poses matched ground truth")
    if not measurement.poses_finite or not all(
        math.isfinite(value) for value in (measurement.gt_rmse_cm, measurement.extent_m, measurement.truth_extent_m, measurement.median_tracker_ms)
    ):
        failures.append("finite: poses and measurements must be finite")
    if measurement.extent_m > DIVERGENCE_FACTOR * measurement.truth_extent_m:
        failures.append(f"divergence: estimate spans {measurement.extent_m:.2f} m, truth {measurement.truth_extent_m:.2f} m")
    if baseline is not None:
        if measurement.gt_rmse_cm > GATE_RATIO * baseline.gt_rmse_cm:
            failures.append(f"accuracy: {measurement.gt_rmse_cm:.3f} cm exceeds {GATE_RATIO * baseline.gt_rmse_cm:.3f} cm")
        if measurement.hostname == baseline.host and measurement.median_tracker_ms > GATE_RATIO * baseline.median_tracker_ms:
            failures.append(f"speed: {measurement.median_tracker_ms:.3f} ms exceeds {GATE_RATIO * baseline.median_tracker_ms:.3f} ms")
    return failures
