"""Regression cases, measured baselines, and ground-truth acceptance rules."""

import math
import re
from dataclasses import dataclass
from pathlib import Path
from tomllib import TOMLDecodeError
from typing import Literal, TypeAlias

from serde import SerdeError, coerce, field, serde
from serde.toml import from_toml

from slam_rs import _core
from slam_rs.config import SlamConfig
from slam_rs.trajectory import MIN_ASSOCIATED_POSES

BENCHMARKS_PATH: Path = Path(__file__).resolve().parents[1] / "benchmarks.toml"
"""Checked-in regression cases and measured baselines."""

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
class Baseline:
    """A measured reference for one profile and lane."""

    profile: Literal["fast", "reference"]
    """Configuration overlay."""
    lane: Literal["gpu", "cpu"]
    """Execution lane."""
    host: str
    """Short host name on which tracker time was measured (no domain suffix)."""
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

    def baseline_for(self, lane: Literal["cpu", "gpu"], profile: Literal["reference", "fast"], host: str) -> Baseline | None:
        """Prefer this host's row; otherwise use the first row for the lane/profile."""
        reference: Baseline | None = None
        for row in self.baseline:
            if row.lane == lane and row.profile == profile:
                if row.host == host.split(".")[0]:
                    return row
                if reference is None:
                    reference = row
        return reference


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
class RobocapBenchmarks:
    """Frozen RoboCap regression recordings and their pixel provenance."""

    has_ground_truth: bool
    """Whether these regression sessions have measured ground truth."""
    decode_path: DecodePath
    """Frozen decoder used by regression comparisons."""
    sessions: tuple[RobocapSession, ...] = field(rename="session")
    """The measured sessions, in manifest order."""

    def session(self, session_id: str, device_id: str) -> RobocapSession:
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
        return RobocapSession(reference_csv=None, session_id=session_id, segment_id=f"robocap__{device_id}__{session_id}")

    def is_listed(self, session_id: str) -> bool:
        return any(session.session_id == session_id for session in self.sessions)


@serde(type_check=coerce, deny_unknown_fields=True)
@dataclass(slots=True, frozen=True)
class Benchmarks:
    """Regression cases and baselines; runtime settings are loaded separately."""

    schema_version: int
    """Benchmark schema version."""
    segments: tuple[ReferenceSegment, ...] = field(rename="segment")
    """MSD reference cases with tiers, hold-outs and measured baselines."""
    robocap: RobocapBenchmarks
    """RoboCap regression sessions."""

    def by_id(self, segment_id: str) -> ReferenceSegment:
        """The segment with this id.

        A command line is what reaches this — ``fleet_check --segments`` and the
        replay tool's ``--segment`` — so an id the manifest cannot satisfy is a
        ``ValueError`` naming the ten it has, the same kind of answer
        :func:`load_benchmarks` gives for a manifest it cannot read. A bare
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


def resolved_flow_config(
    settings: SlamConfig, segment: ReferenceSegment, profile: Literal["reference", "fast"] = "reference"
) -> tuple[_core.VioConfig, str]:
    """Read the dataset configuration and apply the requested profile."""
    text: str = settings.vio_config_text(segment.dataset_name, profile=profile)
    config: _core.VioConfig = _core.VioConfig.from_json(text)
    return config, text


def load_benchmarks(settings: SlamConfig, path: Path = BENCHMARKS_PATH) -> Benchmarks:
    """Read benchmark definitions and validate their dataset references and baselines."""
    try:
        parsed: Benchmarks = from_toml(Benchmarks, path.read_text())
    except (SerdeError, TOMLDecodeError) as error:
        raise ValueError(f"{path}: {error}") from error
    if parsed.schema_version != 1:
        raise ValueError(f"{path}: expected schema_version 1")
    identifiers: set[str] = set()
    dataset_names: set[str] = {dataset.name for dataset in settings.datasets}
    for segment in parsed.segments:
        where: str = f"{path}: [segment] {segment.segment_id}"
        if segment.segment_id in identifiers:
            raise ValueError(f"{where}: duplicate segment ids")
        identifiers.add(segment.segment_id)
        if segment.dataset_name not in dataset_names:
            raise ValueError(f"{where}: unknown dataset {segment.dataset_name!r}")
        baseline_keys: set[tuple[str, str, str]] = set()
        for baseline in segment.baseline:
            if "." in baseline.host:
                raise ValueError(f"{where}: [segment.baseline] host {baseline.host!r} must be short (no domain suffix)")
            key: tuple[str, str, str] = (baseline.lane, baseline.profile, baseline.host)
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
    return parsed


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
        if measurement.hostname.split(".")[0] == baseline.host and measurement.median_tracker_ms > GATE_RATIO * baseline.median_tracker_ms:
            failures.append(f"speed: {measurement.median_tracker_ms:.3f} ms exceeds {GATE_RATIO * baseline.median_tracker_ms:.3f} ms")
    return failures
