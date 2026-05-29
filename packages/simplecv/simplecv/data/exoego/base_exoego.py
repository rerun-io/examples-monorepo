from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from typing import Generic, TypeVar

import cv2
import numpy as np
from jaxtyping import Float, Float32, Int, UInt8, UInt16
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates
from typing_extensions import Self

from simplecv.camera_parameters import Fisheye62Parameters, PinholeParameters
from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.exo.base_exo import BaseExoSequence, ManoStack
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig
from simplecv.data.exoego.sequence_identity import SequenceIdentity
from simplecv.image_types import BGRList

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
    """

    xyzc_stack: Float[ndarray, "num_frames 133 4"]
    timestamps_ns: Int[ndarray, "num_frames"] | None = None
    mano_stack: ManoStack | None = None


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
