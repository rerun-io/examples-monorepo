"""The frozen reference set, read from ``reference_segments.toml``.

The manifest is the single place that says which segments the accuracy work runs
on, where their layers live, what the catalog is expected to report about them,
which decode path produces the pixels, and which IMU noise numbers the estimator
is tuned with. The catalog carries none of the last two, and the decode path
alone moves ATE by centimetres, so a run is not reproducible without them.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

MANIFEST_PATH: Path = Path(__file__).resolve().parents[1] / "reference_segments.toml"
"""The checked-in manifest, beside the package rather than inside it."""

Tier: TypeAlias = Literal["smoke", "accuracy", "long"]
"""How often a segment runs: every commit, per pull request, nightly."""
DecodePath: TypeAlias = Literal["cpu_gray8_dav1d_1thread"]
"""Frozen pixel provenance. Only the single-threaded dav1d ``gray8`` path is gated in V0."""
GroundTruthSource: TypeAlias = Literal["lighthouse", "mocap"]
"""How a segment's ground-truth rig poses were measured."""

TIER_BY_NAME: dict[str, Tier] = {"smoke": "smoke", "accuracy": "accuracy", "long": "long"}
"""Valid tier names, in increasing cost order; the lookup is also how a manifest string becomes a :data:`Tier`."""
DECODE_PATH_BY_NAME: dict[str, DecodePath] = {"cpu_gray8_dav1d_1thread": "cpu_gray8_dav1d_1thread"}
"""Decode paths a gate may be frozen on."""
GT_SOURCE_BY_NAME: dict[str, GroundTruthSource] = {"lighthouse": "lighthouse", "mocap": "mocap"}
"""Ground-truth measurement systems the two MSD devices use."""


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
    """The rig geometry shared by every segment of one dataset.

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


@dataclass(slots=True, frozen=True)
class GroundTruthProperties:
    """What ``/__properties/gt`` reports for a segment, on the ``gt`` layer."""

    num_poses: int
    """Ground-truth rig poses."""
    source: GroundTruthSource
    """Measurement system behind those poses."""


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
class RobocapReference:
    """RoboCap session 15: agreement with basalt C++ on a hard fisheye rig, with no ground truth."""

    session_id: str
    """Recorder session, e.g. ``s00000015``."""
    device_id: str
    """Capture device the session came from."""
    segment_id: str
    """Segment id, which is also the ``.rrd`` file stem on the NAS."""
    base_url: str
    """Storage URL of the ``base`` layer: video, IMU and calibration."""
    slam_url: str
    """Storage URL of the ``slam`` layer, which holds the basalt trajectory."""
    has_ground_truth: bool
    """Always false: RoboCap has no measured ground truth."""
    decode_path: DecodePath
    """Frozen decode path that produced the gated pixels."""
    basalt_num_poses: int
    """Poses on the ``slam`` layer."""
    imu: ImuParameters
    """Frozen IMU noise model, from the device's Kalibr calibration."""
    fixtures: TrajectoryFixtures
    """Checked-in basalt outputs for this session."""

    @property
    def base_path(self) -> Path:
        """Local filesystem path behind :attr:`base_url`."""
        return Path(self.base_url.removeprefix("file://"))

    @property
    def slam_path(self) -> Path:
        """Local filesystem path behind :attr:`slam_url`."""
        return Path(self.slam_url.removeprefix("file://"))


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
            KeyError: If the manifest has no such dataset.
        """
        for dataset in self.datasets:
            if dataset.name == name:
                return dataset
        raise KeyError(f"{name!r} is not in the reference set; have {[d.name for d in self.datasets]}")

    def by_id(self, segment_id: str) -> ReferenceSegment:
        """The segment with this id.

        Raises:
            KeyError: If no segment in the manifest has that id.
        """
        for segment in self.segments:
            if segment.segment_id == segment_id:
                return segment
        raise KeyError(f"{segment_id!r} is not in the reference set; have {[s.segment_id for s in self.segments]}")

    def in_tier(self, tier: Tier) -> tuple[ReferenceSegment, ...]:
        """Every segment in one tier, in manifest order."""
        return tuple(segment for segment in self.segments if segment.tier == tier)


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


def load_manifest(path: Path = MANIFEST_PATH) -> ReferenceManifest:
    """Parse the reference manifest.

    Args:
        path: Manifest file; defaults to the copy checked in beside the package.

    Returns:
        The parsed manifest, with tiers, decode paths and ground-truth sources
        validated against their literal alphabets.

    Raises:
        ValueError: If a tier, decode path or ground-truth source is unknown, or a segment id repeats.
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
            )
        )
    segments: list[ReferenceSegment] = []
    for entry in document["segment"]:
        identifier: str = entry["segment_id"]
        if entry["tier"] not in TIER_BY_NAME:
            raise ValueError(f"{identifier}: unknown tier {entry['tier']!r}, expected one of {sorted(TIER_BY_NAME)}")
        if entry["decode_path"] not in DECODE_PATH_BY_NAME:
            raise ValueError(f"{identifier}: unknown decode path {entry['decode_path']!r}, expected one of {sorted(DECODE_PATH_BY_NAME)}")
        if entry["gt"]["source"] not in GT_SOURCE_BY_NAME:
            raise ValueError(f"{identifier}: unknown ground-truth source {entry['gt']['source']!r}, expected one of {sorted(GT_SOURCE_BY_NAME)}")
        tier: Tier = TIER_BY_NAME[entry["tier"]]
        decode_path: DecodePath = DECODE_PATH_BY_NAME[entry["decode_path"]]
        source: GroundTruthSource = GT_SOURCE_BY_NAME[entry["gt"]["source"]]
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
                imu=_imu(entry["imu"]),
            )
        )
    identifiers: list[str] = [segment.segment_id for segment in segments]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"duplicate segment ids in {path}: {sorted({i for i in identifiers if identifiers.count(i) > 1})}")

    robocap_block: dict[str, Any] = document["robocap"]
    fixtures_block: dict[str, Any] = robocap_block["fixtures"]
    if robocap_block["decode_path"] not in DECODE_PATH_BY_NAME:
        raise ValueError(f"robocap: unknown decode path {robocap_block['decode_path']!r}")
    robocap: RobocapReference = RobocapReference(
        session_id=robocap_block["session_id"],
        device_id=robocap_block["device_id"],
        segment_id=robocap_block["segment_id"],
        base_url=robocap_block["base_url"],
        slam_url=robocap_block["slam_url"],
        has_ground_truth=bool(robocap_block["has_ground_truth"]),
        decode_path=DECODE_PATH_BY_NAME[robocap_block["decode_path"]],
        basalt_num_poses=int(robocap_block["basalt_num_poses"]),
        imu=_imu(robocap_block["imu"]),
        fixtures=TrajectoryFixtures(
            golden=Path(fixtures_block["golden"]),
            candidate=Path(fixtures_block["candidate"]),
            expected_ate_rmse_cm=float(fixtures_block["expected_ate_rmse_cm"]),
            expected_associated=int(fixtures_block["expected_associated"]),
        ),
    )
    return ReferenceManifest(
        schema_version=int(document["schema_version"]),
        catalog_url=document["catalog_url"],
        datasets=tuple(datasets),
        segments=tuple(segments),
        robocap=robocap,
        package_root=path.parent,
    )
