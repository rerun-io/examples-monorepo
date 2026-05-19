from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self, cast

import numpy as np
import rerun as rr
from jaxtyping import Int, UInt8
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates
from simplecv.camera_parameters import PinholeParameters
from simplecv.data.ego.base_ego import BaseEgoSequence, CameraParam, CamNameType, EgoData
from simplecv.data.exo.base_exo import BaseExoSequence, ExoData
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoLabels, ExoEgoSample
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig
from simplecv.image_types import BGRList


def _require_root_directory(root_directory: Path | None) -> Path:
    """Return a configured root directory or raise a targeted configuration error."""
    if root_directory is None:
        msg: str = "SyncedVideoExoEgoConfig.root_directory must be provided for synced video datasets."
        raise ValueError(msg)
    return root_directory


def _sorted_mp4_paths(*, root_directory: Path | None, glob_pattern: str, stream_kind: str) -> list[Path]:
    """Return sorted MP4 paths for one stream kind."""
    video_root: Path = _require_root_directory(root_directory)
    paths: list[Path] = sorted(video_root.glob(glob_pattern))
    if not paths:
        msg: str = f"No {stream_kind} MP4 files matched '{glob_pattern}' under {video_root}."
        raise FileNotFoundError(msg)

    for path in paths:
        if path.suffix.lower() != ".mp4":
            msg = f"Expected {stream_kind} video to be an MP4 file, got {path}."
            raise ValueError(msg)
    return paths


def _entity_name_from_video_path(video_path: Path) -> str:
    """Return a Rerun entity-safe stream name from a video path."""
    return "_".join(video_path.stem.split())


def frame_timestamps_from_reader(reader: Any) -> Int[ndarray, "n_frames"]:
    """Compute nanosecond timestamps from a decoded video reader."""
    fps: float = float(reader.fps)
    frame_count: int = int(reader.frame_cnt)
    if fps <= 0.0:
        msg: str = f"Video reader reported invalid fps: {fps}."
        raise ValueError(msg)
    ns_per_frame: float = 1e9 / fps
    timestamps: Int[ndarray, "n_frames"] = (np.arange(frame_count) * ns_per_frame).astype(np.int64)
    return timestamps


@dataclass
class SyncedVideoExoEgoConfig(BaseExoEgoDatasetConfig):
    """Configuration for generic synced exo/ego MP4 folders."""

    _target: type = field(default_factory=lambda: SyncedVideoExoEgoSequence)
    """Target class to instantiate."""
    load_labels: bool = False
    """Whether to load labels for this sequence. Generic synced videos do not provide labels."""
    root_directory: Path | None = None
    """Directory containing synced exo and ego MP4 subdirectories. Required for setup."""
    exo_glob: str = "exo/*.mp4"
    """Glob, relative to root_directory, selecting exo camera videos."""
    ego_glob: str = "ego/*.mp4"
    """Glob, relative to root_directory, selecting ego camera videos."""


class SyncedVideoExoSequence(BaseExoSequence[SyncedVideoExoEgoConfig]):
    """Exocentric sequence backed by synced MP4 files without camera metadata."""

    def __getitem__(self, idx: int) -> ExoData:
        bgr_list: BGRList = cast(BGRList, self.exo_video_readers[idx])
        return ExoData(cam_params_list=[], bgr_list=bgr_list, xyz=None, uv_dict=None)

    def load_video_paths(self) -> list[Path]:
        """Load synced exo MP4 paths."""
        return _sorted_mp4_paths(
            root_directory=self.config.root_directory,
            glob_pattern=self.config.exo_glob,
            stream_kind="exo",
        )

    def load_exo_cams(self) -> list[PinholeParameters | None]:
        """Return one uncalibrated camera slot per exo video."""
        return [None for _ in self.exo_video_paths]

    @property
    def image_plane_distance(self) -> float:
        return 0.1

    @property
    def exo_video_names(self) -> list[str]:
        """Rerun-safe stream names aligned with exo_video_paths."""
        return [_entity_name_from_video_path(path) for path in self.exo_video_paths]


class SyncedVideoEgoSequence(BaseEgoSequence[SyncedVideoExoEgoConfig]):
    """Egocentric sequence backed by synced MP4 files without camera metadata."""

    def __getitem__(self, idx: int) -> EgoData:
        bgr_list: BGRList = cast(BGRList, self.ego_video_readers[idx])
        return EgoData(cam_params_list=[], bgr_list=bgr_list)

    def load_video_paths(self) -> list[Path]:
        """Load synced ego MP4 paths."""
        return _sorted_mp4_paths(
            root_directory=self.config.root_directory,
            glob_pattern=self.config.ego_glob,
            stream_kind="ego",
        )

    def load_ego_cams(self) -> dict[str, list[CameraParam]]:
        """Return no precomputed ego cameras; the mv-api calibrator estimates them."""
        return {}

    def align_cams_and_videos(
        self,
        video_path_list: list[Path],
        ego_cam_dict: dict[CamNameType, list[CameraParam]],
    ) -> tuple[dict[CamNameType, list[CameraParam]], dict[CamNameType, Path]]:
        """Align camera metadata with videos when metadata exists."""
        del video_path_list
        return ego_cam_dict, {}

    @property
    def image_plane_distance(self) -> float:
        return 0.02

    @property
    def ego_video_names(self) -> list[str]:
        """Rerun-safe stream names aligned with ego_video_paths."""
        return [_entity_name_from_video_path(path) for path in self.ego_video_paths]


class SyncedVideoExoEgoSequence(BaseExoEgoSequence[SyncedVideoExoEgoConfig]):
    """Combined exo/ego sequence backed by synced MP4 folders."""

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        return rr.ViewCoordinates.RIGHT_HAND_Z_UP

    def _build_ego(self) -> BaseEgoSequence[SyncedVideoExoEgoConfig] | None:
        return SyncedVideoEgoSequence(self.config)

    def _build_exo(self) -> BaseExoSequence[SyncedVideoExoEgoConfig] | None:
        return SyncedVideoExoSequence(self.config)

    def __getitem__(
        self,
        idx: int | None = None,
        ts_nano: np.timedelta64 | None = None,
    ) -> ExoEgoSample:
        canonical_idx: int
        timestamp_ns: int
        canonical_idx, timestamp_ns = self._resolve_canonical(idx=idx, ts_nano=ts_nano)
        ego_bgr_list: BGRList | None = self._sample_ego_frames(timestamp_ns)
        exo_cam_params_list, exo_bgr_list = self._sample_exo(timestamp_ns)
        return ExoEgoSample(
            canonical_index=canonical_idx,
            canonical_timestamp_ns=timestamp_ns,
            ego_cam_params_list=[],
            ego_bgr_list=ego_bgr_list,
            exo_cam_params_list=exo_cam_params_list,
            exo_bgr_list=exo_bgr_list,
            labels=None,
        )

    def load_stream_timestamps_ns(self) -> dict[str, Int[ndarray, "n_frames"]]:
        """Return per-stream timestamps and cache aligned ego/exo stream names."""
        stream_timestamps: dict[str, Int[ndarray, "n_frames"]] = {}
        self._ego_stream_names.clear()
        self._exo_stream_names.clear()

        if self.exo_sequence is not None:
            for stream_idx, name in enumerate(
                self.exo_sequence.exo_video_names,
            ):
                stream_name: str = f"exo/{name}"
                reader: Any = self.exo_sequence.exo_video_readers.video_readers[stream_idx]
                timestamps: Int[ndarray, "n_frames"] = frame_timestamps_from_reader(reader)
                stream_timestamps[stream_name] = timestamps
                self._exo_stream_names.append(stream_name)

        if self.ego_sequence is not None:
            for stream_idx, name in enumerate(
                self.ego_sequence.ego_video_names,
            ):
                stream_name = f"ego/{name}"
                reader = self.ego_sequence.ego_video_readers.video_readers[stream_idx]
                timestamps = frame_timestamps_from_reader(reader)
                stream_timestamps[stream_name] = timestamps
                self._ego_stream_names.append(stream_name)

        return stream_timestamps

    def load_labels(self) -> ExoEgoLabels | None:
        """Return no labels for generic synced video folders."""
        return None

    @classmethod
    def iter_episode_sequences(
        cls: type[Self],
        cfg: SyncedVideoExoEgoConfig,
    ) -> Generator[Self, None, None]:
        yield cls(cfg)

    @classmethod
    def num_sequences_for_config(
        cls: type[Self],
        cfg: SyncedVideoExoEgoConfig,
    ) -> int:
        del cfg
        return 1

    def _sample_ego_frames(self, timestamp_ns: int) -> BGRList | None:
        """Fetch ego frames at the closest frame index at or before a timestamp."""
        if self.ego_sequence is None:
            return None

        bgr_list: list[UInt8[ndarray, "H W 3"]] = []
        for stream_idx, stream_name in enumerate(self._ego_stream_names):
            stream_ts: Int[ndarray, "n_frames"] = self.stream_timestamps_ns[stream_name]
            frame_idx: int = self.timestamp_to_frame_index(timestamp_ns, stream_ts)
            reader: Any = self.ego_sequence.ego_video_readers.video_readers[stream_idx]
            frame_obj: object = reader[frame_idx]
            if frame_obj is None:
                msg: str = f"Missing ego frame {frame_idx} for {stream_name}."
                raise ValueError(msg)
            bgr_frame: UInt8[ndarray, "H W 3"] = np.asarray(frame_obj, dtype=np.uint8)
            bgr_list.append(bgr_frame)
        return cast(BGRList, bgr_list)
