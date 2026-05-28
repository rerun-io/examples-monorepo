from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from jaxtyping import Float32
from numpy import ndarray

from simplecv.camera_parameters import PinholeParameters
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig
from simplecv.image_types import BGRList
from simplecv.video_io import MultiVideoReader

CamNameType = TypeVar("CamNameType", bound=str)
ConfigT = TypeVar("ConfigT", bound=BaseExoEgoDatasetConfig)


@dataclass
class ManoStack:
    """Per-sequence MANO parameters grouped for both hands.

    - betas: Shape vector. (10,) means the same shape is shared across hands;
      (2, 10) means per-hand shape (index 0 = right, 1 = left), as used by
      datasets that fit left/right independently (e.g. EPFL Smart Kitchen).
    - so3: Axis-angle pose coefficients (48 = 16 joints x 3) per frame/hand.
    - trans: Global translation per frame/hand.
    - use_pca: Whether the final 45 pose coefficients are MANO PCA coefficients.

    Notes
    - Hand index convention: 0 = right, 1 = left.
    - This splits the previous 51-vector (0:48 so3, 48:51 trans) into explicit fields.
    """

    betas: Float32[ndarray, "10"] | Float32[ndarray, "n_hands=2 10"]
    so3: Float32[ndarray, "n_frames n_hands=2 48"]
    trans: Float32[ndarray, "n_frames n_hands=2 3"]
    use_pca: bool = True

    def betas_for(self, hand_idx: int) -> Float32[ndarray, "10"]:
        """Return the 10-D shape vector for hand_idx (0=right, 1=left)."""
        b: Float32[ndarray, "..."] = self.betas
        if b.ndim == 1:
            return b
        return b[hand_idx]


@dataclass
class ExoData:
    cam_params_list: list[PinholeParameters]
    bgr_list: BGRList
    # assumes left | right hand
    xyz: Float32[ndarray, "2 21 3"] | None
    uv_dict: dict[str, Float32[ndarray, "2 21 2"]] | None


@dataclass
class ExoBatchData:
    uv_stack_dict: dict[str, Float32[ndarray, "n_frames 2 21 2"]]
    xyz_stack: Float32[ndarray, "n_frames 2 21 3"]
    mano_stack: ManoStack | None = None


class BaseExoSequence(ABC, Generic[ConfigT]):
    config: ConfigT

    def __init__(
        self,
        cfg: ConfigT,
    ) -> None:
        self.config: ConfigT = cfg
        self._video_path_list: list[Path] = self.load_video_paths()
        self._exo_cam_list: list[PinholeParameters | None] = self.load_exo_cams()
        # Only create MultiVideoReader if not already set by subclass (e.g., RRD sequences)
        if not hasattr(self, "exo_video_readers") or self.exo_video_readers is None:
            self.exo_video_readers: MultiVideoReader = MultiVideoReader(video_paths=[video_path for video_path in self._video_path_list])

    def __len__(self) -> int:
        return len(self.exo_video_readers)

    def __iter__(self) -> Generator[ExoData, None, None]:
        for idx in range(len(self)):
            # Yield the result of __getitem__ for iteration
            yield self[idx]

    @abstractmethod
    def __getitem__(self, idx: int) -> ExoData:
        pass

    @abstractmethod
    def load_video_paths(self) -> list[Path]:
        """Load the paths to the video files."""
        pass

    @abstractmethod
    def load_exo_cams(self) -> list[PinholeParameters | None]:
        """Load camera parameters, returning None for uncalibrated cameras."""

    @property
    def exo_cam_list(self) -> list[PinholeParameters | None]:
        """Get the list of exo camera parameters (None for uncalibrated cameras)."""
        return self._exo_cam_list

    @property
    def exo_video_paths(self) -> list[Path]:
        """Video paths in the order consumed by the multi-reader."""
        return self._video_path_list

    @property
    def exo_video_names(self) -> list[str]:
        """Stable stream names aligned with ``exo_video_paths``.

        Uses camera name if available, otherwise falls back to video path stem.
        """
        video_names: list[str] = []
        for i, video_path in enumerate(self._video_path_list):
            if i < len(self._exo_cam_list) and self._exo_cam_list[i] is not None:
                video_names.append(self._exo_cam_list[i].name)  # type: ignore[union-attr]
            else:
                video_names.append(video_path.stem)
        return video_names

    @property
    @abstractmethod
    def image_plane_distance(self) -> int | float:
        pass
