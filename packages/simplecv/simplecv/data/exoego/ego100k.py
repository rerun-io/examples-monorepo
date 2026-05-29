from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rerun as rr
from jaxtyping import Float32, Int
from numpy import ndarray
from rerun.components.view_coordinates import ViewCoordinates

from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.ego.ego100k_ego import Ego100KEgoSequence
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoLabels, ExoEgoSample
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig


@dataclass
class Egocentric100KConfig(BaseExoEgoDatasetConfig):
    _target: type = field(default_factory=lambda: Ego100KSequence)
    repo_id: str = "builddotai/Egocentric-100K"
    factory_id: str = "factory001"
    worker_id: str = "worker001"
    download_root: Path | None = None
    tmp_dir: Path | None = None
    load_labels: bool = True


class Ego100KSequence(BaseExoEgoSequence[Egocentric100KConfig]):
    """Egocentric-100K adapter (ego-only, no labels, no exo)."""

    def __init__(self, cfg: Egocentric100KConfig) -> None:
        self._ego_stream_names: list[str] = []
        super().__init__(cfg)

    def __getitem__(self, idx: int | None = None, ts_nano: np.timedelta64 | None = None) -> ExoEgoSample:
        canonical_idx, ts_ns = self._resolve_canonical(idx=idx, ts_nano=ts_nano)
        ego_cam_params_list, ego_bgr_list = self._sample_ego(ts_ns)
        exo_cam_params_list, exo_bgr_list = (None, None)  # no exo cameras
        labels: ExoEgoLabels | None = self._sample_labels(canonical_idx, ts_ns)

        return ExoEgoSample(
            canonical_index=canonical_idx,
            canonical_timestamp_ns=ts_ns,
            ego_cam_params_list=ego_cam_params_list,
            ego_bgr_list=ego_bgr_list,
            exo_cam_params_list=exo_cam_params_list,
            exo_bgr_list=exo_bgr_list,
            labels=labels,
        )

    def _build_ego(self) -> BaseEgoSequence[Egocentric100KConfig] | None:
        return Ego100KEgoSequence(cfg=self.config)

    def _build_exo(self) -> BaseExoSequence[Egocentric100KConfig] | None:
        return None

    def load_stream_timestamps_ns(self) -> dict[str, Int[ndarray, "n_frames"]]:
        """Return per-stream timestamps for ego video."""

        stream_ts: dict[str, Int[ndarray, "n_frames"]] = {}
        self._ego_stream_names.clear()

        if self.ego_sequence is not None:
            for name, video_path in zip(
                self.ego_sequence.ego_video_names,
                self.ego_sequence.ego_video_paths,
                strict=True,
            ):
                stream_name: str = f"ego/{name}"
                timestamps: Int[ndarray, "n_frames"] = rr.AssetVideo(path=video_path).read_frame_timestamps_nanos()
                stream_ts[stream_name] = timestamps
                self._ego_stream_names.append(stream_name)

        labels: ExoEgoLabels | None = self.exoego_labels
        if labels is not None and labels.timestamps_ns is not None:
            stream_ts["labels"] = labels.timestamps_ns

        return stream_ts

    def load_labels(self) -> ExoEgoLabels | None:
        """Return NaN-filled COCO-133 stack to satisfy viewer expectations."""
        if self.ego_sequence is None:
            return None
        num_frames: int = len(self.ego_sequence)
        xyzc_stack: Float32[ndarray, "num_frames 133 4"] = np.full(
            (num_frames, 133, 4),
            np.nan,
            dtype=np.float32,
        )
        xyzc_stack[..., 3] = np.float32(0.0)
        timestamps_ns: Int[ndarray, "num_frames"] | None = None
        return ExoEgoLabels(
            xyzc_stack=xyzc_stack,
            timestamps_ns=timestamps_ns,
        )

    @classmethod
    def iter_episode_sequences(cls, cfg: Egocentric100KConfig) -> Generator["Ego100KSequence", None, None]:
        """Yield exactly one sequence for the configured factory/worker."""
        yield cls(cfg)

    @classmethod
    def num_sequences_for_config(cls, cfg: Egocentric100KConfig) -> int:
        return 1

    @property
    def world_coordinate_system(self) -> ViewCoordinates:
        return rr.ViewCoordinates.RDF

    @property
    def image_plane_distance(self) -> int | float:
        """Get the image plane distance for the camera."""
        if self.ego_sequence is not None:
            return self.ego_sequence.image_plane_distance
        return 0.05
