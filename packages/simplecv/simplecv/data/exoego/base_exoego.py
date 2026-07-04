from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Protocol, TypeVar, runtime_checkable

import cv2
import numpy as np
from jaxtyping import Bool, Float, Float32, Int, UInt8, UInt16
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates
from typing_extensions import Self

from simplecv.camera_parameters import Extrinsics, Fisheye62Parameters, PinholeParameters
from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.exo.base_exo import BaseExoSequence, ManoStack, SmplxStack
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig
from simplecv.data.exoego.sequence_identity import SequenceIdentity
from simplecv.image_types import BGRList
from simplecv.rig import (
    CameraSensor,
    Rig,
    RigCalibration,
    RigPoseStream,
    entity_id,
    identity_extrinsics,
    kind_for_name,
    rebuild_camera_with_extrinsics,
    reference_index_for_names,
)

#: A worn ego device is physically rigid, but some loaders expose each of its
#: cameras as an independent per-frame ``world_T_cam`` rather than a shared device
#: pose. We collapse them into one rig only if the inter-camera offset is constant
#: within this tolerance (max abs element of ``cam_T_rig`` drift, metres / unitless
#: rotation entries); otherwise we fall back to one rig per camera (still correct,
#: just more rig nodes). Datasets built from a shared device pose (Aria, HOT3D)
#: are exactly rigid, so they always collapse.
RIGIDITY_TOL: float = 1e-2

#: How many valid frames to sample when checking rigidity (cheap constant-offset test).
RIGIDITY_SAMPLE_FRAMES: int = 16

ConfigT = TypeVar("ConfigT", bound=BaseExoEgoDatasetConfig)


@dataclass
class ExoEgoLabels:
    """3D COCO-133 annotations plus optional MANO and timing metadata.

    Attributes
    ----------
    xyzc_stack:
        3D keypoints in meters with per-joint confidence ``[x, y, z, c]``.
    timestamps_ns:
        Nanosecond timestamps aligned with ``xyzc_stack``; when provided they are
        used as-is for logging to keep original recording cadence. When ``None``
        the visualization falls back to the video timeline, matching the prior
        behaviour but risking drift when label and video frame rates differ.
    mano_stack:
        Optional MANO parameters associated with the same frames.
    smplx_stack:
        Optional SMPL/SMPL-X body parameters associated with the same frames.
    """

    # TODO(multi-person-keypoints): xyzc_stack holds ONE skeleton, so multi-person
    # datasets (MAMMA dance/multi-people) only get keypoints for person 0 — extra
    # people get SMPL-X meshes only (SmplxStack has an n_people axis; ManoStack and
    # this field do not). Lifting it means a people axis here, per-person entity
    # paths in view_exoego's 3D/2D keypoint logging, and updating rrd_exoego.load_labels.
    xyzc_stack: Float[ndarray, "num_frames 133 4"]
    timestamps_ns: Int[ndarray, "num_frames"] | None = None
    mano_stack: ManoStack | None = None
    smplx_stack: SmplxStack | None = None


@dataclass
class EnvironmentMesh:
    vertex_positions: Float32[ndarray, "num_vertices 3"]
    triangle_indices: Int[ndarray, "num_faces 3"]
    vertex_normals: Float32[ndarray, "num_vertices 3"] | None = None
    vertex_colors: UInt8[ndarray, "num_vertices 4"] | None = None


@dataclass
class ExoEgoSample:
    """Time-synced ego/exo frames plus optional labels at one canonical timestamp."""

    canonical_index: int
    canonical_timestamp_ns: int
    ego_cam_params_list: list[PinholeParameters | Fisheye62Parameters] | None = None
    ego_bgr_list: BGRList | None = None
    ego_depth_list: list[UInt16[ndarray, "H W"]] | None = None
    exo_cam_params_list: list[PinholeParameters | Fisheye62Parameters | None] | None = None
    exo_bgr_list: BGRList | None = None
    exo_depth_list: list[UInt16[ndarray, "H W"]] | None = None
    labels: ExoEgoLabels | None = None


@dataclass
class RigLayout:
    """Canonical COLMAP-style rig/camera entity-path assignment for one sequence.

    One rig per physical device: each exo camera is its own static, world-anchored
    rig; the worn ego device is one moving rig (its cameras share ``world_T_rig(t)``).
    This is the single source of truth that keeps every logging seam — pinholes,
    video, 2D keypoints, depth, blueprint — pointing at the same entity paths.

    Attributes
    ----------
    rigs:
        Every rig with a calibration skeleton (``log_rig_static`` each; the moving
        ones additionally carry a ``pose_stream`` for ``log_rig_pose_stream``).
        Exo rigs come first (``rig_00..``), then the ego rig(s).
    exo_cam_paths / ego_cam_paths:
        Maps **every** video stream name (including uncalibrated exo cameras, which
        get no skeleton but still need a video panel) to its ``/world/rig_NN/cam_MM``
        entity path. Downstream seams append ``pinhole/video`` etc.
    """

    rigs: list[Rig]
    exo_cam_paths: dict[str, Path]
    ego_cam_paths: dict[str, Path]


@runtime_checkable
class EgoCameraSource(Protocol):
    """Minimal read surface :func:`_build_ego_rigs` needs from an ego sequence.

    A real :class:`~simplecv.data.ego.base_ego.BaseEgoSequence` satisfies this (its
    members are read-only properties, hence the ``@property`` declarations); so does
    any lightweight stand-in in tests, which keeps the rig math testable without
    constructing a full video-backed sequence.
    """

    @property
    def image_plane_distance(self) -> int | float: ...

    @property
    def ego_cam_dict(self) -> dict[str, list[PinholeParameters | Fisheye62Parameters]]: ...

    @property
    def ego_video_names(self) -> list[str]: ...


def _build_ego_rigs(
    ego_sequence: EgoCameraSource,
    *,
    start_index: int,
    world_path: Path,
) -> tuple[list[Rig], dict[str, Path]]:
    """Build the moving ego rig(s) from an ego sequence's per-frame cameras.

    The whole ego sequence is treated as ONE physical device (a single worn
    headset/glasses). If its cameras are rigidly coupled (constant inter-camera
    offset, the usual case) they collapse into ONE rig whose ``world_T_rig(t)`` is
    the reference camera's trajectory and whose other cameras are fixed
    ``rig_T_cam`` offsets. If not rigidly factorable, each camera falls back to its
    own single-camera moving rig (still exact). Returns ``(rigs, cam_paths)`` where
    ``cam_paths`` maps every ego stream name to its ``/world/rig_NN/cam_MM`` path.
    """
    image_plane_distance: float = float(ego_sequence.image_plane_distance)
    ego_cam_dict: dict[str, list[PinholeParameters | Fisheye62Parameters]] = ego_sequence.ego_cam_dict
    stream_names: list[str] = [name for name in ego_sequence.ego_video_names if ego_cam_dict.get(name)]
    if not stream_names:
        return [], {}

    n_frames: int = min(len(ego_cam_dict[name]) for name in stream_names)
    per_stream_params: list[list[PinholeParameters | Fisheye62Parameters]] = [ego_cam_dict[name][:n_frames] for name in stream_names]

    reference_idx: int = reference_index_for_names(stream_names)

    def _single_cam_rig(rig_index: int, name: str, params: list[PinholeParameters | Fisheye62Parameters]) -> Rig:
        """One moving single-camera rig: identity ``rig_T_cam``, pose = its trajectory."""
        cam_transforms: Float[ndarray, "n_frames 4 4"] = np.stack([np.asarray(c.extrinsics.world_T_cam, dtype=float) for c in params])
        sensor = CameraSensor(index=0, name=name, kind=kind_for_name(name), pinhole=rebuild_camera_with_extrinsics(params[0], identity_extrinsics()))
        pose_stream = RigPoseStream(
            world_t_rig=cam_transforms[:, :3, 3].copy(),
            world_R_rig=cam_transforms[:, :3, :3].copy(),
        )
        return Rig(
            index=rig_index,
            calibration=RigCalibration(cameras=[sensor], reference_index=0),
            pose_stream=pose_stream,
            image_plane_distance=image_plane_distance,
        )

    if len(stream_names) == 1:
        rig: Rig = _single_cam_rig(start_index, stream_names[0], per_stream_params[0])
        cam_paths: dict[str, Path] = {stream_names[0]: world_path / entity_id("rig", start_index) / entity_id("cam", 0)}
        return [rig], cam_paths

    # Multi-camera ego device: the reference camera's trajectory drives world_T_rig(t).
    reference_params: list[PinholeParameters | Fisheye62Parameters] = per_stream_params[reference_idx]
    world_T_cam_ref: Float[ndarray, "n_frames 4 4"] = np.stack([np.asarray(cam.extrinsics.world_T_cam, dtype=float) for cam in reference_params])
    valid: Bool[ndarray, "n_frames"] = np.isfinite(world_T_cam_ref).all(axis=(1, 2))
    world_t_rig: Float[ndarray, "n_frames 3"] = world_T_cam_ref[:, :3, 3].copy()
    world_R_rig: Float[ndarray, "n_frames 3 3"] = world_T_cam_ref[:, :3, :3].copy()

    # Compute each camera's fixed rig_T_cam offset at its first jointly-valid frame
    # and verify the offset is constant across sampled frames (rigid). cam_T_world is
    # a precomputed field, so each camera's per-frame poses stack once (mirroring the
    # reference camera above) and the validity mask is computed vectorized.
    cam_T_rig_by_name: dict[str, Float[ndarray, "4 4"]] = {stream_names[reference_idx]: np.eye(4)}
    is_rigid: bool = True
    for stream_idx, (name, params) in enumerate(zip(stream_names, per_stream_params, strict=True)):
        if stream_idx == reference_idx:
            continue
        cam_T_world_stack: Float[ndarray, "n_frames 4 4"] = np.stack([np.asarray(c.extrinsics.cam_T_world, dtype=float) for c in params])
        cam_valid: Bool[ndarray, "n_frames"] = np.isfinite(cam_T_world_stack).all(axis=(1, 2))
        valid_frames: Int[ndarray, "n_valid"] = np.flatnonzero(valid & cam_valid)
        if valid_frames.size == 0:
            is_rigid = False
            break
        t0: int = int(valid_frames[0])
        offset_t0: Float[ndarray, "4 4"] = cam_T_world_stack[t0] @ world_T_cam_ref[t0]
        cam_T_rig_by_name[name] = offset_t0
        stride: int = max(1, valid_frames.size // RIGIDITY_SAMPLE_FRAMES)
        for frame in valid_frames[::stride][:RIGIDITY_SAMPLE_FRAMES]:
            offset_t: Float[ndarray, "4 4"] = cam_T_world_stack[int(frame)] @ world_T_cam_ref[int(frame)]
            if float(np.max(np.abs(offset_t - offset_t0))) > RIGIDITY_TOL:
                is_rigid = False
                break
        if not is_rigid:
            break

    ego_rig_path: Path = world_path / entity_id("rig", start_index)
    if is_rigid:
        cameras: list[CameraSensor] = []
        for stream_idx, (name, params) in enumerate(zip(stream_names, per_stream_params, strict=True)):
            offset: Float[ndarray, "4 4"] = cam_T_rig_by_name[name]
            rig_extrinsics: Extrinsics = Extrinsics(cam_R_world=offset[:3, :3].copy(), cam_t_world=offset[:3, 3].copy())
            cameras.append(
                CameraSensor(index=stream_idx, name=name, kind=kind_for_name(name), pinhole=rebuild_camera_with_extrinsics(params[0], rig_extrinsics))
            )
        rig = Rig(
            index=start_index,
            calibration=RigCalibration(cameras=cameras, reference_index=reference_idx),
            pose_stream=RigPoseStream(world_t_rig=world_t_rig, world_R_rig=world_R_rig),
            image_plane_distance=image_plane_distance,
        )
        cam_paths = {name: ego_rig_path / entity_id("cam", stream_idx) for stream_idx, name in enumerate(stream_names)}
        return [rig], cam_paths

    # Fallback: device not rigidly factorable -> one moving rig per ego camera.
    print(f"[rig] ego device cameras not rigidly factorable (offset drift > {RIGIDITY_TOL} m); using one rig per ego camera")
    rigs: list[Rig] = []
    cam_paths = {}
    for offset_idx, (name, params) in enumerate(zip(stream_names, per_stream_params, strict=True)):
        rig_index: int = start_index + offset_idx
        rigs.append(_single_cam_rig(rig_index, name, params))
        cam_paths[name] = world_path / entity_id("rig", rig_index) / entity_id("cam", 0)
    return rigs, cam_paths


class BaseExoEgoSequence(Generic[ConfigT], ABC):  # noqa: UP046
    config: ConfigT
    stream_timestamps_ns: dict[str, Int[ndarray, "n_frames"]]
    canonical_timestamps_ns: Int[ndarray, "n_events"]
    canonical_end_ns: int
    canonical_stream_name: str

    def __init__(
        self,
        cfg: ConfigT,
    ) -> None:
        self.config: ConfigT = cfg
        # Stream name caches (populated by child load_stream_timestamps_ns implementations).
        self._ego_stream_names: list[str] = []
        self._exo_stream_names: list[str] = []
        self.ego_sequence: BaseEgoSequence[ConfigT] | None = self._build_ego()
        self.exo_sequence: BaseExoSequence[ConfigT] | None = self._build_exo()
        self.stream_timestamps_ns = self.load_stream_timestamps_ns()
        (
            self.canonical_stream_name,
            self.canonical_timestamps_ns,
            self.canonical_end_ns,
        ) = self._select_canonical_timeline(self.stream_timestamps_ns)

        if self.config.load_labels:
            self._exoego_labels: ExoEgoLabels | None = self.load_labels()
        self._environment_mesh: EnvironmentMesh | None = self.load_environment_mesh()

    def __len__(self) -> int:
        return int(self.canonical_timestamps_ns.shape[0])

    def __iter__(self) -> Generator["ExoEgoSample", None, None]:
        for idx in range(len(self)):
            # Yield the result of __getitem__ for iteration
            yield self.__getitem__(idx=idx)

    def iter_dataset(self):
        """Sugar so you can call this on an *instance*."""
        yield from self.__class__.iter_episode_sequences(self.config)

    def num_sequences(self) -> int:
        """Return how many episode sequences ``iter_dataset`` will yield."""
        return self.__class__.num_sequences_for_config(self.config)

    @property
    def sequence_identity(self) -> SequenceIdentity:
        """Stable catalog identity for this sequence."""
        return self.__class__.sequence_identity_for_config(self.config)

    @classmethod
    def sequence_identity_for_config(cls: type[Self], cfg: ConfigT) -> SequenceIdentity:
        """Build a generic identity from ``sequence_name`` when a dataset does not override it."""
        sequence_name: str = str(getattr(cfg, "sequence_name", cls.__name__.removesuffix("Sequence")))
        dataset_name: str = cls.__name__.removesuffix("Sequence").lower()
        return SequenceIdentity(dataset=dataset_name, parts=(sequence_name,))

    def build_rig_layout(self, *, world_path: Path = Path("world"), log_exo: bool = True, log_ego: bool = True) -> RigLayout:
        """Assign every camera to a COLMAP-style rig (one rig per physical device).

        Generic and dataset-agnostic: it consumes only the normalized
        ``exo_cam_list`` / ``exo_video_names`` and ``ego_cam_dict`` /
        ``ego_video_names``, so every dataset flows through unchanged. Exo cameras
        each become a static world-anchored rig (``rig_00..``, one camera each;
        a future multi-sensor exo unit simply adds cameras under its rig); the
        worn ego device becomes a moving rig (or a fallback of per-camera rigs if
        not rigidly factorable). See :class:`RigLayout`.

        ``log_exo`` / ``log_ego`` mirror the viewer toggles so a disabled side
        contributes no rigs and reserves no indices (keeping ``rig_NN`` dense).
        """
        rigs: list[Rig] = []
        exo_cam_paths: dict[str, Path] = {}
        next_rig_index: int = 0

        if self.exo_sequence is not None and log_exo:
            image_plane_distance: float = float(self.exo_sequence.image_plane_distance)
            for name, cam in zip(self.exo_sequence.exo_video_names, self.exo_sequence.exo_cam_list, strict=True):
                rig_index: int = next_rig_index
                next_rig_index += 1
                exo_cam_paths[name] = world_path / entity_id("rig", rig_index) / entity_id("cam", 0)
                # Uncalibrated exo cameras (cam is None) reserve a rig slot for their
                # video panel but get no calibration skeleton / frustum.
                if cam is not None:
                    rigs.append(
                        Rig(
                            index=rig_index,
                            calibration=RigCalibration(cameras=[CameraSensor(index=0, name=name, kind=kind_for_name(name), pinhole=cam)], reference_index=0),
                            pose_stream=None,
                            image_plane_distance=image_plane_distance,
                        )
                    )

        ego_cam_paths: dict[str, Path] = {}
        if self.ego_sequence is not None and log_ego:
            ego_rigs, ego_cam_paths = _build_ego_rigs(self.ego_sequence, start_index=next_rig_index, world_path=world_path)
            rigs.extend(ego_rigs)

        return RigLayout(rigs=rigs, exo_cam_paths=exo_cam_paths, ego_cam_paths=ego_cam_paths)

    @abstractmethod
    def _build_ego(self) -> BaseEgoSequence[ConfigT] | None:
        """Build the ego sequence based on the configuration."""

    @abstractmethod
    def _build_exo(self) -> BaseExoSequence[ConfigT] | None:
        """Build the exo sequence based on the configuration."""

    @abstractmethod
    def __getitem__(
        self,
        idx: int | None = None,
        ts_nano: np.timedelta64 | None = None,
    ) -> ExoEgoSample:
        """Get the time-synced sample for a specific canonical index or timestamp."""

    # Convenience helpers to keep call sites readable.
    def at_idx(self, idx: int) -> ExoEgoSample:
        """Fetch a sample by canonical index (sugar for ``self[idx]``)."""
        return self.__getitem__(idx=idx)

    def at_ts(self, ts_nano: np.timedelta64) -> ExoEgoSample:
        """Fetch a sample by nanosecond ``np.timedelta64`` timestamp."""
        return self.__getitem__(ts_nano=ts_nano)

    # ── Shared sampling utilities (reduce duplication across datasets) ──
    def _resolve_canonical(self, idx: int | None, ts_nano: np.timedelta64 | None) -> tuple[int, int]:
        """
        Map either a canonical index or a nanosecond timestamp to
        ``(canonical_idx, canonical_ts_ns)``.
        """
        canonical_ts: Int[ndarray, "n_events"] = self.canonical_timestamps_ns
        if (idx is None) == (ts_nano is None):
            raise AssertionError("Provide exactly one of idx or ts_nano.")

        if ts_nano is not None:
            if not isinstance(ts_nano, np.timedelta64):
                raise TypeError("ts_nano must be a numpy.timedelta64")
            req_ts_ns: int = int(np.int64(ts_nano.astype("timedelta64[ns]")))
            lower: int = int(canonical_ts[0])
            upper: int = int(self.canonical_end_ns)
            clamped: int = int(np.clip(req_ts_ns, lower, upper))
            canonical_idx: int = int(np.searchsorted(canonical_ts, clamped, side="right") - 1)
        else:
            if not isinstance(idx, int | np.integer):
                raise TypeError("idx must be int")
            canonical_idx = int(idx)
        canonical_idx = max(0, min(canonical_idx, len(canonical_ts) - 1))
        ts_ns: int = int(canonical_ts[canonical_idx])
        return canonical_idx, ts_ns

    def _sample_ego(self, ts_ns: int) -> tuple[list[PinholeParameters | Fisheye62Parameters] | None, BGRList | None]:
        """Fetch ego frames + per-frame cam params at timestamp ``ts_ns``."""
        if self.ego_sequence is None:
            return None, None
        cam_params_list: list[PinholeParameters | Fisheye62Parameters] = []
        bgr_list: list = []
        for stream_idx, stream_name in enumerate(self._ego_stream_names):
            stream_ts: Int[ndarray, "n_frames"] = self.stream_timestamps_ns[stream_name]
            frame_idx: int = self.timestamp_to_frame_index(ts_ns, stream_ts)
            reader = self.ego_sequence.ego_video_readers.video_readers[stream_idx]
            bgr_frame = reader.get_frame(frame_idx)
            bgr_list.append(bgr_frame)

            cam_name: str = self.ego_sequence.ego_video_names[stream_idx]
            cam_params_for_cam: list = self.ego_sequence.ego_cam_dict[cam_name]
            cam_idx: int = min(frame_idx, len(cam_params_for_cam) - 1)
            cam_params_list.append(cam_params_for_cam[cam_idx])

        return cam_params_list, bgr_list

    def _sample_exo(self, ts_ns: int) -> tuple[list[PinholeParameters | Fisheye62Parameters | None] | None, BGRList | None]:
        """Fetch exo frames + cam params at timestamp ``ts_ns`` (static transforms assumed).

        Returns None for uncalibrated cameras' cam_params while still returning their frames.
        """
        if self.exo_sequence is None:
            return None, None
        cam_params_list: list[PinholeParameters | Fisheye62Parameters | None] = []
        bgr_list: list = []
        for stream_idx, stream_name in enumerate(self._exo_stream_names):
            stream_ts: Int[ndarray, "n_frames"] = self.stream_timestamps_ns[stream_name]
            frame_idx: int = self.timestamp_to_frame_index(ts_ns, stream_ts)
            reader = self.exo_sequence.exo_video_readers.video_readers[stream_idx]
            bgr_frame = reader.get_frame(frame_idx)
            bgr_list.append(bgr_frame)

            cam_params_for_cam: PinholeParameters | None = self.exo_sequence.exo_cam_list[stream_idx]
            cam_params_list.append(cam_params_for_cam)  # May be None for uncalibrated cameras

        return cam_params_list, bgr_list

    def _sample_exo_depths(self, ts_ns: int) -> list[UInt16[ndarray, "H W"]] | None:
        """Fetch exo depth maps aligned to ``ts_ns`` (uint16 millimetres)."""
        if self.exo_sequence is None:
            return None
        depth_paths_seq = getattr(self.exo_sequence, "depth_paths", None)
        if depth_paths_seq is None:
            return None

        depth_list: list[UInt16[ndarray, "H W"]] = []
        for stream_idx, stream_name in enumerate(self._exo_stream_names):
            stream_ts: Int[ndarray, "n_frames"] = self.stream_timestamps_ns[stream_name]
            frame_idx: int = self.timestamp_to_frame_index(ts_ns, stream_ts)
            clamped_idx: int = min(frame_idx, len(depth_paths_seq) - 1)

            # Use video name (works even if cam_params is None for uncalibrated cameras)
            cam_name: str = self.exo_sequence.exo_video_names[stream_idx]
            depth_path = depth_paths_seq[clamped_idx].get(cam_name)
            if depth_path is None or not depth_path.exists():
                return None

            depth_img: UInt16[ndarray, "H W"] | None = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
            if depth_img is None:
                return None
            depth_list.append(depth_img.astype(np.uint16))

        return depth_list

    def _sample_ego_depths(self, ts_ns: int) -> list[UInt16[ndarray, "H W"]] | None:
        """Fetch ego depth maps aligned to ``ts_ns`` if available."""
        if self.ego_sequence is None:
            return None
        depth_paths_seq = getattr(self.ego_sequence, "depth_paths", None)
        if depth_paths_seq is None:
            return None

        depth_list: list[UInt16[ndarray, "H W"]] = []
        for stream_idx, stream_name in enumerate(self._ego_stream_names):
            stream_ts: Int[ndarray, "n_frames"] = self.stream_timestamps_ns[stream_name]
            frame_idx: int = self.timestamp_to_frame_index(ts_ns, stream_ts)
            clamped_idx: int = min(frame_idx, len(depth_paths_seq) - 1)

            cam_name: str = self.ego_sequence.ego_video_names[stream_idx]
            depth_path = depth_paths_seq[clamped_idx].get(cam_name)
            if depth_path is None or not depth_path.exists():
                return None

            depth_img: UInt16[ndarray, "H W"] | None = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
            if depth_img is None:
                return None
            depth_list.append(depth_img.astype(np.uint16))

        return depth_list

    def _sample_labels(self, canonical_idx: int, ts_ns: int) -> ExoEgoLabels | None:
        """Clamp label stack to timestamp / canonical index."""
        labels: ExoEgoLabels | None = self.exoego_labels
        if labels is None:
            return None
        if labels.timestamps_ns is not None:
            label_idx: int = self.timestamp_to_frame_index(ts_ns, labels.timestamps_ns)
        else:
            max_idx: int = int(labels.xyzc_stack.shape[0] - 1)
            label_idx = min(canonical_idx, max_idx)
        xyzc_stack_frame = labels.xyzc_stack[label_idx]
        return ExoEgoLabels(
            xyzc_stack=xyzc_stack_frame[np.newaxis, ...],
            timestamps_ns=labels.timestamps_ns,
            mano_stack=labels.mano_stack,
            smplx_stack=labels.smplx_stack,
        )

    @abstractmethod
    def load_stream_timestamps_ns(self) -> dict[str, Int[ndarray, "n_frames"]]:
        """Return per-stream nanosecond timestamps keyed by stream name."""

    @abstractmethod
    def load_labels(self) -> ExoEgoLabels | None:
        """Load labels for the sequence, if applicable."""

    def load_environment_mesh(self) -> EnvironmentMesh | None:
        """Optional hook for loading a static environment mesh."""
        return None

    @classmethod
    @abstractmethod
    def iter_episode_sequences(cls: type[Self], cfg: ConfigT) -> Generator[Self, None, None]: ...

    @classmethod
    @abstractmethod
    def num_sequences_for_config(cls: type[Self], cfg: ConfigT) -> int: ...

    @property
    @abstractmethod
    def world_coordinate_system(self) -> ViewCoordinates:
        """Return the world coordinate system for the sequence."""

    @property
    def exoego_labels(self) -> ExoEgoLabels | None:
        """Return the labels for the sequence, if available."""
        return getattr(self, "_exoego_labels", None)

    @property
    def environment_mesh(self) -> EnvironmentMesh | None:
        """Return the static environment mesh, if available."""
        return getattr(self, "_environment_mesh", None)

    @staticmethod
    def timestamp_to_frame_index(time_ns: int, frame_timestamps_ns: Int[ndarray, "num_frames"]) -> int:
        """Map a timestamp (ns) to the closest frame idx at-or-before that time."""

        idx: int = int(np.searchsorted(frame_timestamps_ns, time_ns, side="right") - 1)
        return max(0, min(idx, len(frame_timestamps_ns) - 1))

    @staticmethod
    def _select_canonical_timeline(
        stream_ts: dict[str, Int[ndarray, "n_frames"]],
    ) -> tuple[str, Int[ndarray, "n_events"], int]:
        """Choose a canonical timeline (shortest duration) and clip it to its own end."""
        if not stream_ts:
            raise ValueError("No stream timestamps provided for canonical timeline selection.")

        end_times: dict[str, int] = {name: int(ts[-1]) for name, ts in stream_ts.items() if ts.size > 0}
        if not end_times:
            raise ValueError("All provided stream timestamp arrays are empty.")

        canonical_stream: str = min(end_times, key=lambda stream_name: end_times[stream_name])
        canonical_end_ns: int = end_times[canonical_stream]
        canonical_ts: Int[ndarray, "n_events"] = stream_ts[canonical_stream]
        # clip to its own end in case of trailing padding
        mask: ndarray = canonical_ts <= canonical_end_ns
        canonical_ts = canonical_ts[mask]
        return canonical_stream, canonical_ts, canonical_end_ns
