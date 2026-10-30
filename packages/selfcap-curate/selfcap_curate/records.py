"""Shared input schemas and strict JSON records for SelfCap time alignment."""

import math
import re
from dataclasses import dataclass, field
from json import loads
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from dataforge.records import read_json
from dataforge.schema import video_path
from jaxtyping import Bool, Float64
from numpy import ndarray
from serde import SerdeError, from_dict, serde
from serde.json import to_json


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Lag:
    """A source-to-target rotation-speed lag."""

    lag_s: float | None
    """Seconds to add to source time to reach target time; null when missing."""
    peak: float | None
    """Maximum normalized correlation."""
    second_peak: float | None
    """Best correlation more than 100 ms from the peak."""
    overlap_s: float
    """Common interval retained for every tested lag."""
    clipped: bool = False
    """Peak fell within 5 ms of the search edge."""
    clipped_at_s: float | None = None
    """Legacy clipped peak, retained as provenance rather than a usable lag."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class RepeatPattern:
    """Native-frame repeat-pattern diagnostics from catalog motion or frozen records."""

    ppm_vs_24: int
    """Rate difference relative to 24 fps."""
    phi: float
    """Repeat phase."""
    agreement: float
    """Fraction of agreeing pairs."""
    pairs_used: int
    """Number of tested pairs."""
    moving_frac: float
    """Fraction with motion."""
    repeat_frac_pred: float
    """Predicted fraction of repeated frames."""
    native_frames_pred: int
    """Predicted native frames."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Links:
    """Coarse links and fine gyro links, including the new Quest measurement."""

    lag_left_to_pose: Lag
    """OAK left to Quest pose."""
    lag_rgb_to_left: Lag
    """Coarse video-grid lag, used only to seed gyro refinement."""
    lag_quest_video_to_pose: Lag
    """Quest right to pose, retained for comparison."""
    lag_left_to_gyro: Lag
    """Coarse OAK left to gyro."""
    lag_gyro_to_pose: Lag
    """Gyro to Quest pose."""
    oak_left_pattern: RepeatPattern | None = None
    """Frozen OAK left pattern, if available."""
    oak_rgb_pattern: RepeatPattern | None = None
    """Frozen OAK RGB pattern, if available."""
    lag_left_to_quest_video: Lag | None = None
    """Coarse left to Quest right."""
    left_to_gyro: Lag | None = None
    """1 kHz OAK left refinement."""
    rgb_to_gyro: Lag | None = None
    """1 kHz OAK RGB refinement."""
    quest_video_to_gyro: Lag | None = None
    """1 kHz Quest right refinement."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class RgbModel:
    """Episode's session RGB clock model, preserved without fitting here."""

    a_ms: float
    """Session intercept."""
    b_ms_per_s: float
    """Drift per second."""
    t_start_session_s: float
    """Episode start on the session clock."""
    how: str
    """Source of model."""
    formula: str
    """Human-readable offset formula."""

    def offset_ms(self, t_s: float | Float64[ndarray, "n"]) -> float | Float64[ndarray, "n"]:
        """RGB drift offset in milliseconds, preserving the presented-time origin."""
        return self.a_ms + self.b_ms_per_s * (self.t_start_session_s + t_s)

    def video_time_s(self, times: Float64[ndarray, "n"]) -> Float64[ndarray, "n"]:
        """Add the stored RGB drift offset to presented seconds."""
        return times + self.offset_ms(times) / 1000.0


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class ExoFit:
    """Per-phone measured or inferred timing and fit diagnostics."""

    tau_phone_to_pose_ms: float | None
    """Final phone-to-pose offset."""
    how: str
    """Measured or filled provenance."""
    tau_ms: float | None = None
    """Measured minimum."""
    cmin_px: float | None = None
    """Residual at minimum."""
    basin_halfwidth_ms: float | None = None
    """Width diagnostic."""
    points: int | None = None
    """Correspondence count."""
    edge: bool | None = None
    """Fit reached its search edge."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Confidence:
    """Existing sync confidence fields."""

    closure_ms: float | None
    """Link closure error."""
    peak_left_to_pose: float | None
    """Left-to-pose peak."""
    peak_left_to_gyro: float | None
    """Left-to-gyro peak."""
    peak_quest_video_to_pose: float | None
    """Quest-to-pose peak."""
    peak_rgb_to_left: float | None
    """RGB-to-left peak."""
    rgb_episode_minus_model_ms: float | None
    """RGB residual from the session model."""
    hand_latency_ms: float
    """Measured hand latency."""
    ego_exo_abs_uncertainty_ms: float
    """Absolute clock uncertainty."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class TimeSync:
    """Per-episode v2 sync JSON, with offsets to ADD to capture time."""

    key: str
    """Date, session prefix, episode."""
    session: str
    """Session UUID."""
    date: str
    """Capture date."""
    episode: str
    """Episode name."""
    method: str
    """Method version."""
    aligned_clock: str
    """Description of aligned clock."""
    offsets_ms: dict[str, float | None] | None
    """Per-stream offsets, including named exo phones."""
    flags: list[str]
    """Quality and fallback flags."""
    oak_rgb_model: RgbModel
    """Session RGB drift model."""
    exo_fit: dict[str, ExoFit]
    """Named phone fit records."""
    confidence: Confidence
    """Confidence diagnostics."""
    links: Links
    """Rotation-speed links."""
    video_start_ms: dict[str, float] | None = None
    """Per-video container starts restored by the served-base migration."""
    quest_video_delta_ms: float | None = None
    """Additional Quest video alignment shift, independent of capture_time."""
    base_patch_version: int | None = None
    """Migration marker; a patched base cannot be patched a second time."""
    hand_latency_ms: float | None = None
    """Quest hand tracking latency subtracted on video_time; capture_time stays raw."""
    version: str | None = None
    """CPU fit implementation version; absent in legacy v2 JSON."""

    def __post_init__(self) -> None:
        identity = EpisodeKey.parse(self.key)
        if (identity.date, identity.episode, identity.sid8) != (self.date, self.episode, self.session.split("-", 1)[0]):
            raise ValueError(f"{self.key}: sync identity disagrees with date, episode or session")

    @property
    def episode_key(self) -> "EpisodeKey":
        """Canonical identity derived from the existing JSON key."""
        return EpisodeKey.parse(self.key)


def read_sync(path: Path) -> TimeSync:
    """Read a strict machine-written sync record."""
    return read_json(path, TimeSync)


def write_sync(path: Path, record: TimeSync) -> None:
    """Write a machine JSON record (missing measurements are null)."""
    path.write_text(to_json(record))


def read_frozen_sync(path: Path, *, text: str | None = None) -> TimeSync:
    """Decode a legacy record, replacing unscored NaN cells and naming its source."""
    try:
        return from_dict(TimeSync, loads(path.read_text() if text is None else text, parse_constant=lambda _token: None))
    except (SerdeError, ValueError) as error:
        raise ValueError(f"{path}: invalid frozen SelfCap sync record: {error}") from error


def finite(value: float | None) -> float | None:
    """Keep unscored values out of machine JSON: use null, never NaN or infinity."""
    return value if value is not None and math.isfinite(value) else None


def oak_quest_offset_ms(sync: TimeSync) -> float | None:
    """Quest offset usable by OAK: flagged as-cut origins do not anchor OAK clocks."""
    return (
        None
        if any(flag in sync.flags for flag in ("quest_unsynced_kept_as_cut", "quest_pose_unsynced"))
        else (sync.offsets_ms or {}).get("quest_pose")
    )


@dataclass(frozen=True, slots=True)
class GyroHeadOffsets:
    """Original offsets whose relative relation supports gyro/head fitting."""

    quest_pose_ms: float
    """Raw Quest pose offset, before any origin subtraction."""
    oak_imu_ms: float
    """Raw IMU offset; keep the solver's original arithmetic order."""


def gyro_head_offsets(sync: TimeSync) -> GyroHeadOffsets | None:
    """A via-pose IMU may share Quest's as-cut origin; both raw offsets are still needed."""
    offsets = sync.offsets_ms or {}
    qp, oi = offsets.get("quest_pose"), offsets.get("oak_imu")
    if qp is None or oi is None or "oak_imu_unsynced" in sync.flags or (oak_quest_offset_ms(sync) is None and "oak_imu_via_pose" not in sync.flags):
        return None
    return GyroHeadOffsets(qp, oi)


@dataclass(frozen=True, slots=True)
class PhoneDetections:
    """One phone's detections on presented container PTS, in seconds."""

    times_s: Float64[ndarray, "n"]
    """torchcodec pts_seconds, including the container start; no rebasing."""
    xy: Float64[ndarray, "n 133 2"]
    """COCO-133 image coordinates in pixels; absent detections may be NaN."""
    scores: Float64[ndarray, "n 133"]
    """Detection confidence, widened from the source float16 without changing values."""

    def __post_init__(self) -> None:
        if len(self.times_s) != len(self.xy) or len(self.xy) != len(self.scores):
            raise ValueError("detection frame counts must agree")
        if not np.isfinite(self.times_s).all() or not np.all(np.diff(self.times_s) > 0.0):
            raise ValueError("detection times must be finite and strictly increasing")


Side: TypeAlias = Literal["left", "right"]
MAPPED: tuple[int, ...] = (2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20)
HAND_BASE: dict[Side, int] = {"left": 91, "right": 112}


@dataclass(frozen=True, slots=True)
class CatalogHands:
    """Raw catalog streams on capture_time, before tracking-quality filtering."""

    times_s: Float64[ndarray, "n"]
    """Raw hand CSV stamp in seconds."""
    positions: Float64[ndarray, "n 133 3"]
    """COCO-133 positions; absent points are NaN."""
    head_times_s: Float64[ndarray, "h"]
    """Raw Quest rig pose stamps."""
    head_positions: Float64[ndarray, "h 3"]
    """Base rig translations: the left-eye origin, not a fabricated cyclopean pose."""
    head_quaternions: Float64[ndarray, "h 4"]
    """Head-to-world xyzw orientations."""


@dataclass(frozen=True, slots=True)
class HandTrack:
    """One hand's positions, validity and motion weights."""

    times_s: Float64[ndarray, "n"]
    """Raw Quest hand clock in seconds."""
    joints: Float64[ndarray, "n 21 3"]
    """Mapped hand points; index 0 is Quest joint 1 (wrist)."""
    valid: Bool[ndarray, "n"]
    """Tracked, without jumps/freezes and within the episode's view cone."""
    speed: Float64[ndarray, "n"]
    """Wrist speed in metres/second."""


@dataclass(frozen=True, slots=True)
class SelfcapLayout:
    """Segment-local device assignment."""

    oak_rig: int
    """OAK rig, equal to the discovered iPhone count."""
    quest_rig: int
    """Quest rig immediately after OAK."""
    phones: tuple[str, ...]
    """Physical exo names in rig order."""

    @property
    def video_entities(self) -> tuple[str, str, str]:
        """OAK left, OAK RGB, Quest right, in that order."""
        return video_path(self.oak_rig, 1), video_path(self.oak_rig, 0), video_path(self.quest_rig, 1)


@dataclass(frozen=True, slots=True)
class OakDetections:
    video_time: Float64[ndarray, "n"]
    """Frozen aligned RGB video_time (not raw PTS)."""
    xy: Float64[ndarray, "n 133 2"]
    """Sapiens full-frame COCO-133 image observations."""
    scores: Float64[ndarray, "n 133"]
    """Frozen confidence widened from float16."""


@dataclass(frozen=True, slots=True)
class SessionEpisode:
    """One calibration episode owns its clocks, observations and physical layout."""

    sync: TimeSync
    """Alignment offsets in milliseconds, to add to raw capture times."""
    hands: CatalogHands
    """Catalog hands and left-eye Quest pose on capture_time."""
    gyro_times: Float64[ndarray, "n"]
    """Raw catalog OAK gyro capture seconds."""
    gyro: Float64[ndarray, "n 3"]
    """OAK angular velocity in radians/s, preserving its axes."""

    quest: dict[Side, HandTrack]
    """Quest hands filtered once, on the raw capture clock."""
    layout: SelfcapLayout
    """Physical camera assignment of this segment."""
    detections: dict[str, PhoneDetections] = field(default_factory=dict)
    """Phone stage streams; empty until detection or when only OAK is fitted."""
    oak: OakDetections = field(default_factory=lambda: OakDetections(np.empty(0), np.empty((0, 133, 2)), np.empty((0, 133))))
    """OAK stage stream; empty when only focal profiles are produced."""


HAND_LATENCY_MS: float = 48.0
"""Torch-flash estimate (3 sessions, 11 phone measurements): subtract 48 ms when converting hand-fit tau into a head-clock offset; guards the measured hand/head latency (±12 ms)."""


EPISODE_KEY = re.compile(r"\d{4}-\d{2}-\d{2}__[0-9a-f]{8}__ep\d{3}")


@dataclass(frozen=True, slots=True)
class EpisodeKey:
    """One episode identity, with canonical catalog and local output names."""

    date: str
    """Capture date."""
    sid8: str
    """Eight-character session UUID prefix."""
    episode: str
    """Episode label, epNNN."""

    def __post_init__(self) -> None:
        if EPISODE_KEY.fullmatch(self.key) is None:
            raise ValueError(f"invalid session episode key: {self.key}")

    @classmethod
    def parse(cls, text: str) -> "EpisodeKey":
        """Accept one explicit catalog segment id or unprefixed key."""
        try:
            date, sid8, episode = text.removeprefix("selfcap__").split("__")
        except ValueError as error:
            raise ValueError(f"invalid session episode key: {text}") from error
        return cls(date, sid8, episode)

    @property
    def key(self) -> str:
        return f"{self.date}__{self.sid8}__{self.episode}"

    @property
    def segment(self) -> str:
        return f"selfcap__{self.key}"

    @property
    def session(self) -> str:
        return f"{self.date}__{self.sid8}"

    @property
    def rrd_name(self) -> str:
        return f"{self.segment}.rrd"

    @property
    def sync_name(self) -> str:
        return f"{self.segment}.json"

    def __str__(self) -> str:
        return self.key


def episode_keys(ids: tuple[str, ...], key_file: Path | None) -> tuple[EpisodeKey, ...]:
    """Read named episodes only, preserving first-occurrence order; never enumerate."""
    selected = [*ids, *(line.strip() for line in key_file.read_text().splitlines() if line.strip())] if key_file is not None else list(ids)
    if not selected:
        raise ValueError("supply explicit session episode keys: date__sid8__epNNN")
    return tuple(dict.fromkeys(EpisodeKey.parse(text) for text in selected))


def one_session(keys: tuple[EpisodeKey, ...]) -> str:
    """Require one selected session at the CLI boundary."""
    if len({key.session for key in keys}) != 1:
        raise ValueError("supply explicit episode ids for exactly one session")
    return keys[0].session


@serde
@dataclass(frozen=True, slots=True)
class GyroFit:
    """Result and partial adapter for frozen gyro_mount records."""

    session: str  # Date__sid8 session id.
    ok: bool  # At least one episode supplied 50 usable aligned samples.
    R_head_from_imu: Float64[ndarray, "3 3"] | None = None  # IMU-to-Quest-rig rotation; the mount has no translational estimate.
    max_episode_dev_deg: float | None = None  # Maximum episode deviation from the pooled rotation.


@serde
@dataclass(frozen=True, slots=True)
class OakFit:
    """Mount result and partial adapter for frozen oak_free/session records."""

    session: str = ""  # Explicit date__sid8 session id.
    ok: bool = False  # At least 100 pairs and a successful PnP initialization.
    oak_device_id: str | None = None  # Full OAK device id used to select its factory-fixed RGB/IMU rotation.
    median_px: float | None = None  # Median over retained hand groups.
    questrig_T_rgb: Float64[ndarray, "4 4"] | None = None  # RGB-camera to Quest-left-rig rigid transform in metres.
    hands_kept_frac: float | None = None  # Fraction of complete frame/side groups retained.
    rotation_source: str = "hands"  # Free hand rotation or gyro mount rotation @ device rgb_R_imu.
    observation_source: Literal["sapiens", "egohand"] = "sapiens"  # Image observations used by the mount fit.
