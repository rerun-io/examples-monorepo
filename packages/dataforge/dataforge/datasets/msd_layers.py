"""The machinery one MSD sequence goes through: its streams, and the two layers written from them.

Split out of ``dataforge.datasets.msd`` so neither file hides the other. This one
holds the vocabulary a headset is described in, the entity layout the two layers
write into, and the three steps a conversion runs — read the archive, write base,
write gt — as plain functions over explicit arguments. ``msd.py`` holds the three
headsets themselves and the verbs that drive these steps.

Nothing here reaches back for a dataset instance, which is the point: the gt step
takes a base rrd and a sidecar *path*, so one call serves both a full conversion
(against staged temp files) and a gt-only rebuild (against last week's published
pair). See ``packages/dataforge/README.md#the-layer-rule``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import pyarrow as pa
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation

from dataforge import schema, writing
from dataforge.archives import MemberReader
from dataforge.basalt import CalibratedCamera, FollowFrame, camera_parameters
from dataforge.euroc import (
    GT_VALUE_COLUMNS,
    IMU_VALUE_COLUMNS,
    MAG_VALUE_COLUMNS,
    CameraRow,
    GtTrajectory,
    TimestampedSamples,
    first_timestamp_ns,
    gt_trajectory,
    nominal_fps,
    read_camera_index,
    read_numeric_csv,
)
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import (
    ImuChannel,
    log_camera_node,
    log_imu,
    log_magnetometer,
    log_pose_track,
    log_rig_node,
    log_video_stream,
    time_column,
)
from dataforge.video_encoding import FrameSource, encode_frames_to_mp4

RIG: int = 0
"""MSD is one headset; the whole device is ``rig_00``."""
RIG_REFERENCE: str = "imu_00"
"""The rig frame *is* the IMU frame: every extrinsic in the calibration is a ``T_imu_cam``."""
IMU: int = 0
"""Every MSD sequence ships exactly one IMU, ``imu0`` → ``imu_00``."""
MAG: int = 0
"""The G2 and the Odyssey+ ship exactly one magnetometer, ``mag0`` → ``mag_00``."""
IMAGE_PLANE_DISTANCE: float = 0.1
"""Frustum length in metres; a headset's baseline is ~10 cm, so the frusta stay legible."""
PROPERTIES_ENTITY: str = "/__properties"
"""Where Rerun keeps a recording's properties; the gt layer reads base's clock origin from it."""
GT_SIDECAR_NAME: str = "gt.csv"
"""The sidecar the gt layer rebuilds from: the archive's ``gt/data.csv``, byte for byte.

Kept verbatim rather than as parsed columns so it stays the upstream artifact —
comparable with the corpus, and readable by the same tools — and because at ~1 kHz
it is the only part of a multi-gigabyte archive small enough to keep.
"""

MsdDeviceChoice: TypeAlias = Literal["index", "g2", "odyssey"]
"""``--device``: which headset's corpus to work on, and which catalog dataset."""
GtSource: TypeAlias = Literal["lighthouse", "mocap"]
"""What produced a device's ground truth: SteamVR Lighthouse, or a MoCap system."""
WorldUpAxis: TypeAlias = Literal["+x", "-x", "+y", "-y", "+z", "-z"]
"""Signed axis of a tracking world that gravity points *away* from."""
POSITIVE_WORLD_AXES: tuple[WorldUpAxis, WorldUpAxis, WorldUpAxis] = ("+x", "+y", "+z")
"""Axis names by column index, for a positive mean; the negative row is below."""
NEGATIVE_WORLD_AXES: tuple[WorldUpAxis, WorldUpAxis, WorldUpAxis] = ("-x", "-y", "-z")
"""Axis names by column index, for a negative mean."""
STANDARD_GRAVITY_MS2: float = 9.80665
"""Standard gravity; ``measured_world_up`` reports its result as a fraction of this."""
MEASURED_UP_WINDOW_NS: int = 2_000_000_000
"""How much of a sequence's start ``measured_world_up`` averages over."""

WORLD_UP_VIEW_COORDINATES: dict[WorldUpAxis, rr.components.ViewCoordinates] = {
    "+x": rr.ViewCoordinates.RIGHT_HAND_X_UP,
    "-x": rr.ViewCoordinates.RIGHT_HAND_X_DOWN,
    "+y": rr.ViewCoordinates.RIGHT_HAND_Y_UP,
    "-y": rr.ViewCoordinates.RIGHT_HAND_Y_DOWN,
    "+z": rr.ViewCoordinates.RIGHT_HAND_Z_UP,
    "-z": rr.ViewCoordinates.RIGHT_HAND_Z_DOWN,
}
"""Root ``ViewCoordinates`` per world up axis, right-handed throughout.

Rerun's ``RIGHT_HAND_*`` aliases are exactly this table (``RIGHT_HAND_Y_UP`` is
``RUB``, ``RIGHT_HAND_Z_UP`` is ``RFU``), so naming the up axis is the whole
decision — the remaining two axes are then fixed by handedness. MSD's csvs say
nothing about world axes at all; see ``MSD_DEVICES`` for how each device's is fixed.
"""
GT_TRAJECTORY_COLOR: tuple[int, int, int] = (110, 180, 255)
"""Fixed tint of the whole gt path; one trajectory is one quantity, not a per-row class."""
GT_TRAJECTORY_RADIUS_M: float = 0.002
"""Line radius of the gt path, in metres — thin, because it overlays the rig itself."""
GT_TRAIL_COLOR: tuple[int, int, int] = (255, 215, 90)
"""Fixed tint of the recent-motion trail; warm, so it reads against the cool full path."""
GT_TRAIL_RADIUS_M: float = 0.004
"""Point radius of the trail, in metres; a 1 kHz trail is dense, so the dots stay small."""


@dataclass(frozen=True, slots=True)
class MsdDevice:
    """Everything that varies between the three headsets."""

    hf_dir: str
    """Device directory under ``M_monado_datasets`` (e.g. ``MI_valve_index``)."""
    collections: tuple[str, ...]
    """Sequence collections to enumerate, relative to ``hf_dir``; the
    ``*C_calibration`` collections are deliberately excluded (they hold
    calibration-target recordings, not trajectories)."""
    num_cameras: int
    """Cameras in the headset, logged as ``cam_00..cam_NN``."""
    has_magnetometer: bool
    """Whether the sequences ship a ``mag0/data.csv``."""
    label: str
    """Human device name for the rig node's ``name`` AnyValue."""
    world_up: WorldUpAxis
    """Up axis of this device's tracking world; every rrd of the device carries
    the matching root ``ViewCoordinates``. Measured, not assumed — see ``MSD_DEVICES``."""
    follow: FollowFrame
    """Forward and up of this headset in its rig frame, which place its follow
    camera. Derived from the device's calibration — see ``MSD_DEVICES``."""
    gt_source: GtSource
    """What produced ``gt/data.csv``; goes into the gt layer's properties."""


@dataclass(frozen=True, slots=True)
class MeasuredUp:
    """What one sequence's own gravity measurement found; the gt layer records both."""

    axis: WorldUpAxis
    """The dominant signed world axis the mean acceleration points along."""
    fraction: float
    """That component as a fraction of standard gravity; near 1 is a clean measurement."""


def measured_world_up(gt: GtTrajectory, accel: ImuChannel, *, window_ns: int = MEASURED_UP_WINDOW_NS) -> MeasuredUp:
    """Measure which world axis is up, from gravity as the accelerometer sees it.

    An accelerometer at rest measures the *reaction* to gravity, so its reading
    points **up**; rotating each sample into the world with the ground truth's
    own orientation (``world_R_rig @ a_rig``) and averaging therefore yields a
    vector along the world's up axis. Only the first couple of seconds are used:
    a headset is typically still on the floor or on a head that has not started
    moving, so the mean is nearly pure gravity there and gets noisier the longer
    the window. Why this is measured at all, and what the three devices answer,
    is on ``MSD_DEVICES``.

    Args:
        gt: The sequence's ground truth, already in xyzw order and sanitized.
        accel: Accelerometer samples in m/s^2, on the same clock as ``gt``.
        window_ns: Length of the averaging window, from the first sample both
            streams cover.

    Returns:
        The axis and how much of gravity it carried — a health check, not a
        calibration: a much smaller fraction means the mean is not gravity.

    Raises:
        ValueError: Either stream is empty, or they do not overlap inside the window.
    """
    if gt.times_ns.size == 0 or accel.times_ns.size == 0:
        raise ValueError("measuring the world up axis needs both a gt pose and an accelerometer sample")
    start_ns: int = max(int(gt.times_ns[0]), int(accel.times_ns[0]))
    inside: Bool[ndarray, "n_samples"] = (accel.times_ns >= start_ns) & (accel.times_ns < start_ns + window_ns)
    if not inside.any():
        raise ValueError(f"no accelerometer sample within {window_ns / 1e9:g} s of {start_ns}, where the gt starts")

    window_times_ns: Int64[ndarray, "n_window"] = accel.times_ns[inside]
    after: Int64[ndarray, "n_window"] = np.clip(np.searchsorted(gt.times_ns, window_times_ns), 0, gt.times_ns.size - 1)
    before: Int64[ndarray, "n_window"] = np.clip(after - 1, 0, gt.times_ns.size - 1)
    nearest: Int64[ndarray, "n_window"] = np.where(
        np.abs(gt.times_ns[before] - window_times_ns) <= np.abs(gt.times_ns[after] - window_times_ns), before, after
    )
    world_accel_xyz: Float64[ndarray, "n_window 3"] = Rotation.from_quat(gt.quaternions_xyzw[nearest]).apply(accel.values_xyz[inside])
    mean_xyz: Float64[ndarray, "3"] = world_accel_xyz.mean(axis=0)

    axis_index: int = int(np.argmax(np.abs(mean_xyz)))
    names: tuple[WorldUpAxis, WorldUpAxis, WorldUpAxis] = POSITIVE_WORLD_AXES if mean_xyz[axis_index] >= 0.0 else NEGATIVE_WORLD_AXES
    return MeasuredUp(axis=names[axis_index], fraction=float(abs(mean_xyz[axis_index]) / STANDARD_GRAVITY_MS2))


@dataclass(frozen=True, slots=True)
class EncodedCamera:
    """One camera of one sequence: its calibration, its frame clock, and its clip.

    The three travel together because the base layer writes them together and a
    camera missing any one of them cannot be logged at all — as three parallel
    tuples they could go out of step between the encode and the write.
    """

    calibration: CalibratedCamera
    """This camera's validated record out of the device's ``calibration.json``."""
    times_ns: Int64[ndarray, "n_samples"]
    """Frame times on the zero-based ``video_time`` clock, one per sample in ``clip``."""
    clip: Path
    """The mp4 this camera's PNGs were encoded into."""


@dataclass(frozen=True, slots=True)
class SequenceStreams:
    """Everything one archive read yields; both layers are written from it and nothing else.

    Every clock here is already zero-based ``video_time`` — the archive's device
    clock minus ``start_time_ns`` — because the two layers must share one origin
    and the shift is a property of the sequence, not of either layer.
    """

    cameras: tuple[EncodedCamera, ...]
    """Every camera, in ``cam0``…``camN`` order, with its clock and its clip."""
    gyro: ImuChannel
    """Angular velocity in rad/s."""
    accel: ImuChannel
    """Linear acceleration in m/s^2; also what the world-up measurement reads."""
    magnetometer: ImuChannel | None
    """Field samples in the sensor's own units, or ``None`` on a device without one."""
    start_time_ns: int
    """The device-clock origin every stream above was shifted by; a recording property.

    Deliberately the only trace of ``gt`` left here: its first stamp is part of
    this origin, but the trajectory itself belongs to the gt layer, which parses
    it out of the sidecar rather than being handed a copy the base layer kept."""
    duration_ns: int
    """Span of the *sensor* streams; deliberately not bounded by gt."""

    @property
    def num_frames(self) -> int:
        """Longest per-camera sample count, which is what the capture properties report."""
        return max(camera.times_ns.size for camera in self.cameras)


@dataclass(frozen=True, slots=True)
class BaseClock:
    """What the gt layer reads back out of a published base layer, and only that."""

    start_time_ns: int
    """The device-clock origin the base layer shifted every stream by, off
    ``property:capture:start_time_ns``. The gt csv holds raw device stamps, so
    this is what puts its poses on the same ``video_time`` as the video."""
    accel: ImuChannel
    """The accelerometer, already on ``video_time``; the world-up measurement's other half."""


@dataclass(frozen=True, slots=True)
class GtSummary:
    """What one gt-layer write reports back: its size, and what it measured."""

    num_poses: int
    """Rows the sidecar held, which is what the layer's properties report."""
    measured: MeasuredUp
    """This sequence's own gravity measurement, for ``warn_on_world_up``."""


def read_base_clock(base_rrd: Path) -> BaseClock:
    """Read the clock origin and the accelerometer back out of a base-layer rrd.

    Chunks are filtered **while streaming** rather than loaded into a store and
    queried: a base rrd is mostly video, and the long sessions run to 2 GB, so
    materializing one to reach a 3-column sensor stream would cost gigabytes of
    resident memory per sequence. Keeping only the properties entity and the
    accelerometer holds the same read to a few hundred megabytes on the largest
    sequence in the corpus.

    Args:
        base_rrd: A published (or staged) base layer of the sequence.

    Returns:
        The origin and the accelerometer, ready for ``measured_world_up``.

    Raises:
        ValueError: The rrd carries no ``capture:start_time_ns`` property or no
            accelerometer samples, so it is not a base layer this package wrote.
    """
    accel_path: str = schema.accel_path(RIG, IMU)
    wanted: list[rr.experimental.Chunk] = [
        chunk
        for chunk in rr.experimental.RrdReader(base_rrd).stream()
        if chunk.entity_path == accel_path or chunk.entity_path.startswith(PROPERTIES_ENTITY)
    ]
    store: rr.experimental.ChunkStore = rr.experimental.ChunkStore.from_chunks(wanted)
    properties: pa.Table = store.reader(index=None, contents=f"{PROPERTIES_ENTITY}/**").to_arrow_table()
    origin: list[int] | None = properties.to_pylist()[0].get(schema.capture_property("start_time_ns")) if properties.num_rows else None
    if not origin:
        raise ValueError(f"{base_rrd} carries no {schema.capture_property('start_time_ns')}; it is not a dataforge base layer")

    samples: pa.Table = (
        store.reader(index=schema.TIMELINE)
        .to_arrow_table()
        .sort_by(schema.TIMELINE)
        .select([schema.TIMELINE, f"{accel_path}:Scalars:scalars"])
        .drop_null()
    )
    if samples.num_rows == 0:
        raise ValueError(f"{base_rrd} holds no {accel_path} samples; the world up axis cannot be measured from it")
    times_ns: Int64[ndarray, "n_samples"] = samples.column(schema.TIMELINE).combine_chunks().cast(pa.int64()).to_numpy()
    values_xyz: Float64[ndarray, "n_samples 3"] = np.asarray(samples.column(1).to_pylist(), dtype=np.float64)
    return BaseClock(
        start_time_ns=int(origin[0]),
        accel=ImuChannel(times_ns=np.ascontiguousarray(times_ns, dtype=np.int64), values_xyz=values_xyz),
    )


def read_sequence(
    reader: MemberReader,
    sequence: str,
    cameras: Sequence[CalibratedCamera],
    *,
    profile: MsdDevice,
    work_dir: Path,
    staged_sidecar: Path,
) -> SequenceStreams:
    """Read every csv and encode every camera: the whole archive, in one pass.

    The csvs come first so a sequence that is missing a stream fails before
    the expensive half, and the mp4s all exist before any of them is remuxed
    — a split archive's reader extracts one camera at a time, so a converter
    cannot hold a reader open across the two.

    This is also the only place ``gt/data.csv`` is ever in reach, so it is
    copied out to the sidecar here, byte for byte and before anything
    expensive runs. Only its first stamp is read, because that stamp is part
    of the sequence's clock origin: parsing the trajectory is the gt layer's
    job, and holding a parsed copy here would make it the second parse.

    Args:
        reader: Open reader over the sequence's archive volume(s).
        sequence: Archive stem, which is also its top-level directory inside it.
        cameras: The device's validated calibration records, one per camera;
            each is carried into the ``EncodedCamera`` beside its own clip.
        profile: The headset this sequence came off; its camera count and whether
            it carries a magnetometer decide which members are read.
        work_dir: Scratch directory the encoded mp4s are written into.
        staged_sidecar: Temp path ``gt/data.csv`` is copied to; the caller
            publishes it beside the two rrds.

    Raises:
        ValueError: A camera index, ``imu0/data.csv`` or ``gt/data.csv`` holds
            no data rows, so the sequence has no clock to place the rest on.
    """
    mav0: str = f"{sequence}/mav0"
    camera_rows: list[list[CameraRow]] = []
    for index in range(profile.num_cameras):
        rows: list[CameraRow] = read_camera_index(reader.read_member(f"{mav0}/cam{index}/data.csv"))
        if not rows:
            raise ValueError(f"{sequence} cam{index} has an empty data.csv")
        camera_rows.append(rows)
    camera_times_ns: list[Int64[ndarray, "n_samples"]] = [
        np.array([row.timestamp_ns for row in rows], dtype=np.int64) for rows in camera_rows
    ]
    inertial: TimestampedSamples = read_numeric_csv(reader.read_member(f"{mav0}/imu0/data.csv"), num_values=IMU_VALUE_COLUMNS)
    if inertial.times_ns.size == 0:
        raise ValueError(f"{sequence} imu0/data.csv has no data rows, so the sequence has no inertial stream")
    magnetometer: TimestampedSamples | None = (
        read_numeric_csv(reader.read_member(f"{mav0}/mag0/data.csv"), num_values=MAG_VALUE_COLUMNS) if profile.has_magnetometer else None
    )
    # The gt *layer* is a sibling rrd built from this file later, but both layers
    # share one zero-based video_time, so the base layer needs gt's clock origin.
    gt_csv: bytes = reader.read_member(f"{mav0}/gt/data.csv")
    staged_sidecar.write_bytes(gt_csv)

    sensor_times_ns: list[Int64[ndarray, "n_samples"]] = [*camera_times_ns, inertial.times_ns]
    if magnetometer is not None and magnetometer.times_ns.size:
        sensor_times_ns.append(magnetometer.times_ns)
    start_time_ns: int = min(first_timestamp_ns(gt_csv), *(int(times_ns[0]) for times_ns in sensor_times_ns))
    # Deliberately not bounded by gt: duration_ns describes the *sensor* layer.
    duration_ns: int = max(int(times_ns[-1]) for times_ns in sensor_times_ns) - start_time_ns

    encoded: list[EncodedCamera] = []
    for index, (calibration, rows, times_ns) in enumerate(zip(cameras, camera_rows, camera_times_ns, strict=True)):
        clip: Path = work_dir / f"cam{index}.mp4"
        encode_frames_to_mp4(
            reader.iter_members([f"{mav0}/cam{index}/data/{row.filename}" for row in rows]),
            clip,
            source=FrameSource("png"),
            fps=nominal_fps(times_ns),
        )
        encoded.append(EncodedCamera(calibration=calibration, times_ns=times_ns - start_time_ns, clip=clip))

    inertial_times_ns: Int64[ndarray, "n_samples"] = inertial.times_ns - start_time_ns
    return SequenceStreams(
        cameras=tuple(encoded),
        gyro=ImuChannel(times_ns=inertial_times_ns, values_xyz=inertial.values[:, :3]),
        accel=ImuChannel(times_ns=inertial_times_ns, values_xyz=inertial.values[:, 3:6]),
        magnetometer=(
            None if magnetometer is None else ImuChannel(times_ns=magnetometer.times_ns - start_time_ns, values_xyz=magnetometer.values)
        ),
        start_time_ns=start_time_ns,
        duration_ns=duration_ns,
    )


def write_base_layer(
    identity: SequenceIdentity,
    streams: SequenceStreams,
    staged_base: Path,
    *,
    profile: MsdDevice,
    device: MsdDeviceChoice,
    collection: str,
    hf_revision: str,
    default_blueprint: rrb.Blueprint,
) -> int:
    """Write the sensor layer: every camera's video, the IMU, the magnetometer.

    Saved into ``staged_base`` rather than published, because the gt layer
    reads this file next and both are published together once it has. The
    recording is flushed and closed on return, so it is readable then.

    Args:
        identity: The sequence's identity; its recording id names the recording.
        streams: One archive read's cameras and sensor channels.
        staged_base: Temp path to save into; the caller publishes it.
        profile: The headset, for its camera count and its label.
        device: The ``--device`` literal, recorded as the ``device`` capture key.
        collection: Collection the sequence came from, recorded likewise.
        hf_revision: The resolved repo sha this conversion read.
        default_blueprint: Layout embedded in the file.

    Returns:
        Frames the longest camera holds, which is the recording's ``num_frames``.
    """
    with writing.recording_to(staged_base, recording_id=identity.recording_id, default_blueprint=default_blueprint) as recording:
        # Deliberately NO ViewCoordinates at "/": the gt layer owns the root
        # ViewCoordinates, because it is what establishes a world frame at all.
        log_rig_node(recording, RIG, reference=RIG_REFERENCE, num_cameras=profile.num_cameras, name=profile.label, kind="ego")
        for camera in streams.cameras:
            index: int = camera.calibration.index
            # The model tag saves a consumer from inferring the projection from the
            # distortion component. The record reports no validity radius for a kb4
            # camera and for a radtan8 one whose rpmax is non-positive — basalt reads
            # that as the check being off, not as a zero-radius limit — and AnyValues
            # then leaves the key off entirely.
            log_camera_node(
                recording,
                RIG,
                index,
                camera_parameters(camera.calibration, name=f"cam{index}"),
                name=f"cam{index}",
                kind="grayscale",
                image_plane_distance=IMAGE_PLANE_DISTANCE,
                camera_model=camera.calibration.camera_model,
                distortion_valid_radius=camera.calibration.distortion_valid_radius,
            )
            log_video_stream(recording, camera.clip, schema.video_path(RIG, index), times_ns=camera.times_ns)
        log_imu(recording, RIG, IMU, gyro=streams.gyro, accel=streams.accel, name="imu0")
        if streams.magnetometer is not None:
            log_magnetometer(recording, RIG, MAG, field=streams.magnetometer, name="mag0")
        writing.send_capture_properties(
            recording,
            identity,
            num_cameras=profile.num_cameras,
            num_frames=streams.num_frames,
            start_time_ns=streams.start_time_ns,
            device=device,
            device_label=profile.label,
            collection=collection,
            hf_revision=hf_revision,
            duration_ns=streams.duration_ns,
        )
    return streams.num_frames


def write_gt_layer(
    identity: SequenceIdentity,
    staged_gt: Path,
    *,
    base_rrd: Path,
    sidecar: Path,
    profile: MsdDevice,
) -> GtSummary:
    """Write the ground-truth layer from the sidecar and the base rrd, and nothing else.

    Deliberately not handed the in-memory streams of a conversion: reading
    the published inputs is what makes a gt rebuild possible at all, and a
    step that *could* take a shortcut on the fetch path would only be
    exercised there — the rebuild path would then be the untested one. So the
    clock origin and the accelerometer come out of the base rrd and the poses
    out of the sidecar csv, whether those two are a full conversion's staged
    temp files or last week's published pair.

    Written with ``send_properties=False`` and no recording name: this is the
    same recording as the base it stacks onto, so a second ``RecordingInfo``
    would duplicate base's name and record a ``start_time`` of whenever this
    layer was last rebuilt. Its own ``property:gt:*`` still lands.

    Args:
        identity: The sequence's identity; its recording id is the base
            layer's, which is what makes the catalog stack the two.
        staged_gt: Temp path to save into; the caller publishes it.
        base_rrd: The base layer to read the clock origin and accelerometer from.
        sidecar: The archive's ``gt/data.csv``, kept verbatim.
        profile: The headset, for the world axes it declares and what produced
            its ground truth.
        profile: The headset, for the world axes it declares and what produced
            its ground truth.

    Returns:
        The pose count and this sequence's own gravity measurement.
    """
    raw: TimestampedSamples = read_numeric_csv(sidecar.read_bytes(), num_values=GT_VALUE_COLUMNS)
    published: BaseClock = read_base_clock(base_rrd)
    gt: GtTrajectory = gt_trajectory(replace(raw, times_ns=raw.times_ns - published.start_time_ns))
    measured: MeasuredUp = measured_world_up(gt, published.accel)
    with writing.recording_to(staged_gt, recording_id=identity.recording_id, send_properties=False) as recording:
        # The gt layer establishes a world frame at all, so it — not the base
        # layer — owns the root ViewCoordinates. The axis is the device's
        # declared one, not this sequence's measurement: every rrd of a device
        # must agree, and a disagreement is a warning, not a silent
        # per-sequence reorientation.
        rr.log("/", WORLD_UP_VIEW_COORDINATES[profile.world_up], static=True, recording=recording)
        log_pose_track(
            recording,
            schema.rig_path(RIG),
            times_ns=gt.times_ns,
            translations_xyz=gt.translations_xyz,
            quaternions_xyzw=gt.quaternions_xyzw,
        )
        # Two views of one trajectory: the static strip is the whole path for the
        # overview, and the per-pose points are what the blueprint's cursor-relative
        # time range turns into a recent-motion trail in the follow view.
        rr.log(
            schema.trajectory_path(schema.GT_RUN_SOURCE),
            rr.LineStrips3D([gt.translations_xyz], colors=GT_TRAJECTORY_COLOR, radii=GT_TRAJECTORY_RADIUS_M),
            static=True,
            recording=recording,
        )
        rr.log(
            schema.trail_path(schema.GT_RUN_SOURCE),
            rr.Points3D.from_fields(colors=GT_TRAIL_COLOR, radii=GT_TRAIL_RADIUS_M),
            static=True,
            recording=recording,
        )
        rr.send_columns(
            schema.trail_path(schema.GT_RUN_SOURCE),
            indexes=[time_column(gt.times_ns)],
            columns=rr.Points3D.columns(positions=gt.translations_xyz),
            recording=recording,
        )
        recording.send_property(
            "gt",
            rr.AnyValues(
                num_poses=int(gt.times_ns.size),
                duration_ns=int(gt.times_ns[-1] - gt.times_ns[0]),
                num_sanitized=gt.num_sanitized,
                source=profile.gt_source,
                world_up=profile.world_up,
                measured_up=measured.axis,
                measured_up_fraction=measured.fraction,
            ),
        )
    return GtSummary(num_poses=int(gt.times_ns.size), measured=measured)
