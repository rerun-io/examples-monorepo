"""Catalog replay metadata, dataset configuration, and ground-truth gate rules."""

import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

from slam_rs import _core

MANIFEST_PATH: Path = Path(__file__).resolve().parents[1] / "reference_segments.toml"
"""The checked-in manifest, beside the package rather than inside it."""

Tier: TypeAlias = Literal["smoke", "release", "listed"]
"""Smoke checks, release checks, or additional listed segments."""
DecodePath: TypeAlias = Literal["cpu_gray8_dav1d_1thread", "cpu_gray8_swscale_area_downscale3"]
"""Frozen pixel provenance.

The MSD gate is frozen on the single-threaded dav1d ``gray8`` path (D28). RoboCap's
H.264 streams are decoded on the same one decoder thread but reformatted straight to
``gray8`` at a third of their size in one ``swscale`` call with ``SWS_AREA``, which is
the operation the reference RoboCap reader performs.
"""
GroundTruthSource: TypeAlias = Literal["lighthouse", "mocap"]
"""How a segment's ground-truth rig poses were measured."""

TIER_BY_NAME: dict[str, Tier] = {"smoke": "smoke", "release": "release", "listed": "listed"}
"""Valid tier names, in increasing cost order; the lookup is also how a manifest string becomes a :data:`Tier`."""
DECODE_PATH_BY_NAME: dict[str, DecodePath] = {
    "cpu_gray8_dav1d_1thread": "cpu_gray8_dav1d_1thread",
    "cpu_gray8_swscale_area_downscale3": "cpu_gray8_swscale_area_downscale3",
}
"""Decode paths a gate may be frozen on."""
GT_SOURCE_BY_NAME: dict[str, GroundTruthSource] = {"lighthouse": "lighthouse", "mocap": "mocap"}
"""Ground-truth measurement systems the two MSD devices use."""


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
    imu: ImuParameters
    """Noise model from the rig calibration."""
    vio_config: Path
    """Dataset configuration path, relative to the manifest."""


@dataclass(slots=True, frozen=True)
class GroundTruthProperties:
    """What ``/__properties/gt`` reports for a segment, on the ``gt`` layer."""

    num_poses: int
    """Ground-truth rig poses."""
    source: GroundTruthSource
    """Measurement system behind those poses."""


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


@dataclass(slots=True, frozen=True)
class ReferenceSegment:
    """One frozen segment of the reference set."""

    dataset_name: str
    """Catalog dataset the segment belongs to."""
    dataset_entry_id: str
    """Catalog entry id of that dataset."""
    segment_id: str
    """Segment id, used to locate the recording on the catalog."""
    hold_out: bool
    """Whether this segment is excluded from tuning."""
    baseline: tuple[Baseline, ...]
    """Measurements by profile and execution lane."""
    tier: Tier
    """How often the segment runs."""
    decode_path: DecodePath
    """Frozen decode path that produced the gated pixels."""
    capture: CaptureProperties
    """Expected capture properties."""
    gt: GroundTruthProperties
    """Expected ground-truth properties."""
    layers: dict[str, LayerFingerprint]
    """Size and schema digest per layer name (``base``, ``gt``)."""
    imu: ImuParameters
    """Frozen IMU noise model for this device."""


@dataclass(slots=True, frozen=True)
class RobocapSession:
    """One catalog session and its optional regression reference."""

    reference_csv: Path | None
    """Optional regression trajectory, relative to the package."""
    session_id: str
    """Recorder session, e.g. ``s00000015``."""
    segment_id: str
    """Segment id, used to locate the recording on the catalog."""

    @property
    def fleet_id(self) -> str:
        """Short form a fleet row names this session by: ``robocap-s15`` for ``s00000015``."""
        return f"robocap-s{int(self.session_id.removeprefix('s'))}"


@dataclass(slots=True, frozen=True)
class RobocapReference:
    """RoboCap rig parameters. This dataset has no ground truth."""

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
    """basalt VIO config the reference ran, relative to the package root."""
    calibration: str
    """basalt calibration the reference ran, at :attr:`downscale`, relative to the package root."""
    imu: ImuParameters
    """Frozen IMU noise model, from the device's Kalibr calibration."""
    sessions: tuple[RobocapSession, ...]
    """The measured sessions, in manifest order."""

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


def _robocap(robocap_block: dict[str, Any]) -> RobocapReference:
    """Read the RoboCap rig and catalog session table."""
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
                reference_csv=Path(block["reference_csv"]) if "reference_csv" in block else None,
                session_id=block["session_id"],
                segment_id=block["segment_id"],
            )
            for block in robocap_block["session"]
        ),
    )


def load_manifest(path: Path = MANIFEST_PATH) -> ReferenceManifest:
    """Parse the catalog manifest and validate segment identifiers and rig properties."""
    document: dict[str, Any] = tomllib.loads(path.read_text())
    if document.get("schema_version") != 9:
        raise ValueError(f"{path}: expected schema_version 9")
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
                imu=_imu(entry["imu"]),
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
                dataset_entry_id=next(d.entry_id for d in datasets if d.name == entry["dataset_name"]),
                segment_id=entry["segment_id"],
                tier=tier,
                hold_out=bool(entry.get("hold_out", False)),
                baseline=tuple(Baseline(**row) for row in entry.get("baseline", [])),
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
                imu=next(d.imu for d in datasets if d.name == entry["dataset_name"]),
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
    return parsed


def gate_failures(
    *,
    framesets: int,
    tracked: int,
    lost: int,
    associated: int,
    gt_rmse_cm: float,
    extent_m: float,
    truth_extent_m: float,
    poses_finite: bool,
    baseline: Baseline | None,
    median_tracker_ms: float,
    hostname: str,
    lane: Literal["cpu", "gpu"],
    profile: Literal["reference", "fast"],
) -> list[str]:
    """Return each failed ground-truth, tracking, or same-host speed clause."""
    import math

    from slam_rs.trajectory import MIN_ASSOCIATED_POSES

    failures: list[str] = []
    if tracked < MIN_TRACKED_POSES:
        failures.append(f"tracked: {pose_floor_text(tracked=tracked, framesets=framesets)}")
    if lost != 0:
        failures.append(f"lost: {lost} of {framesets} framesets")
    if associated < MIN_ASSOCIATED_POSES:
        failures.append(f"associated: only {associated} poses matched ground truth")
    if not poses_finite or not all(math.isfinite(value) for value in (gt_rmse_cm, extent_m, truth_extent_m, median_tracker_ms)):
        failures.append("finite: poses and measurements must be finite")
    if extent_m > DIVERGENCE_FACTOR * truth_extent_m:
        failures.append(f"divergence: estimate spans {extent_m:.2f} m, truth {truth_extent_m:.2f} m")
    if baseline is not None and baseline.lane == lane and baseline.profile == profile:
        if gt_rmse_cm > 1.10 * baseline.gt_rmse_cm:
            failures.append(f"accuracy: {gt_rmse_cm:.3f} cm exceeds {1.10 * baseline.gt_rmse_cm:.3f} cm")
        if hostname == baseline.host and median_tracker_ms > 1.10 * baseline.median_tracker_ms:
            failures.append(f"speed: {median_tracker_ms:.3f} ms exceeds {1.10 * baseline.median_tracker_ms:.3f} ms")
    return failures
