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

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

from slam_rs import _core, reference_bundle
from slam_rs.reference_bundle import BundleFile

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
    """Commit of the basalt fork that produced the run."""
    fork_branch: str
    """Branch the commit sits on. Not pushed anywhere; the fork is machine-local."""
    fork_base: str
    """Upstream commit the branch was cut from."""
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
    """

    device_id: str
    """Capture device every session came from; the calibration is per device."""
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
            KeyError: If the manifest has no such session.
        """
        for session in self.sessions:
            if session.session_id == session_id:
                return session
        raise KeyError(f"{session_id!r} is not a RoboCap session in the manifest; have {[s.session_id for s in self.sessions]}")


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

    def vio_config_text(self, dataset_name: str) -> str:
        """The basalt VIO config one dataset's segments run with, as its file's own text.

        Args:
            dataset_name: Catalog dataset name.

        Returns:
            The vendored file's text, ready for :meth:`slam_rs._core.VioConfig.from_json`.

        Raises:
            KeyError: If the manifest has no such dataset.
        """
        return (self.package_root / self.dataset(dataset_name).vio_config).read_text()

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
            return reference_bundle.resolve(segment.segment_id, "basalt_traj.csv")
        path: Path = self.package_root / segment.reference.trajectory_csv
        reason: str | None = None if path.is_file() else f"{path} is committed in the manifest but missing from this checkout"
        return BundleFile(path=path, reason=reason)


def flow_config(manifest: ReferenceManifest, segment: ReferenceSegment) -> _core.VioConfig:
    """The basalt config the C++ reference ran this segment's dataset with.

    basalt's constructor defaults are not its shipped files: ``msdmi_config.json``
    and ``msdmg_config.json`` set ``vio_marg_lost_landmarks`` to true where the
    constructor says false (``crates/slam-rs/src/config.rs:25``, pinned by that
    crate's own test). That one key is a different estimator — with the
    constructor's defaults the port sat 1.41 to 12.05 cm from the C++ on the
    reference clips, with the dataset's config 0.31 to 5.19 cm (C72) — so the
    file is read rather than reconstructed.

    Nothing is written on top of it. The manifest's per-device
    ``optical_flow_image_safe_radius`` is asserted against the file instead, so a
    manifest and a config that disagree stop the run rather than one of them
    silently winning. The replay tool and the V2 gate both build their estimator
    from here, so neither can drive a segment with another one's configuration.

    Args:
        manifest: The reference set the segment came from, which resolves the
            dataset's config file.
        segment: The segment about to be replayed.

    Returns:
        The config to build an estimator or a frontend for that segment with.

    Raises:
        ValueError: If the file's image safe radius is not the one the manifest
            froze for this segment.
    """
    config: _core.VioConfig = _core.VioConfig.from_json(manifest.vio_config_text(segment.dataset_name))
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
    if block["gate_policy"] not in GATE_POLICY_BY_NAME:
        raise ValueError(f"{segment_id}: unknown gate policy {block['gate_policy']!r}, expected one of {sorted(GATE_POLICY_BY_NAME)}")
    if block["decode_path"] not in DECODE_PATH_BY_NAME:
        raise ValueError(f"{segment_id}: reference run has unknown decode path {block['decode_path']!r}")
    ate_block: dict[str, Any] = block["expected_cpp_ate"]
    return CppReferenceRun(
        gate_policy=GATE_POLICY_BY_NAME[block["gate_policy"]],
        fork_commit=block["fork_commit"],
        fork_branch=block["fork_branch"],
        fork_base=block["fork_base"],
        decode_path=DECODE_PATH_BY_NAME[block["decode_path"]],
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
                vio_config=Path(entry["vio_config"]),
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
                reference=_reference_run(entry["reference"], identifier),
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
        device_id=robocap_block["device_id"],
        has_ground_truth=bool(robocap_block["has_ground_truth"]),
        decode_path=DECODE_PATH_BY_NAME[robocap_block["decode_path"]],
        camera_names=tuple(str(name) for name in robocap_block["camera_names"]),
        downscale=int(robocap_block["downscale"]),
        frameset_tolerance_ns=int(robocap_block["frameset_tolerance_ns"]),
        interpolate_accel_onto_gyro=bool(robocap_block["interpolate_accel_onto_gyro"]),
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
    return ReferenceManifest(
        schema_version=int(document["schema_version"]),
        catalog_url=document["catalog_url"],
        datasets=tuple(datasets),
        segments=tuple(segments),
        robocap=robocap,
        package_root=path.parent,
    )
