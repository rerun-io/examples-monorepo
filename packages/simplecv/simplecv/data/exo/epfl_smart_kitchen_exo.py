from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from jaxtyping import Float32
from numpy import ndarray

from simplecv.camera_parameters import PinholeParameters
from simplecv.data.exo.base_exo import BaseExoSequence, ExoData
from simplecv.data.exoego.epfl_smart_kitchen import (
    EPFL_EXO_CAMERA_NAMES,
    _matrix4,
    _pinhole_from_entry,
    load_camera_matrix,
    video_path_for_camera,
)

if TYPE_CHECKING:
    from simplecv.data.exoego.epfl_smart_kitchen import EpflSmartKitchenConfig
else:
    from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig as EpflSmartKitchenConfig


class EpflSmartKitchenExoSequence(BaseExoSequence[EpflSmartKitchenConfig]):
    """Static exocentric RGB streams for EPFL-Smart-Kitchen."""

    def load_video_paths(self) -> list[Path]:
        video_paths: list[Path] = []
        for camera_name in self.config.exo_camera_names:
            if camera_name not in EPFL_EXO_CAMERA_NAMES:
                raise ValueError(f"Unknown EPFL exo camera name: {camera_name}")
            video_path: Path = video_path_for_camera(self.config, camera_name)
            if not video_path.exists():
                raise FileNotFoundError(f"EPFL exo RGB video not found for {camera_name}: {video_path}")
            video_paths.append(video_path)
        return video_paths

    def load_exo_cams(self) -> list[PinholeParameters | None]:
        camera_matrix: dict[str, dict] = load_camera_matrix(self.config)
        cameras: list[PinholeParameters | None] = []
        warned_word2cam: bool = False
        for camera_name in self.config.exo_camera_names:
            if camera_name not in camera_matrix:
                raise KeyError(f"EPFL camera matrix is missing {camera_name!r}")
            entry: dict = camera_matrix[camera_name]
            cam_T_world_raw: object | None = entry.get("world2cam", entry.get("word2cam"))
            if cam_T_world_raw is None:
                raise KeyError(f"EPFL camera matrix entry {camera_name!r} is missing world2cam")
            if "world2cam" not in entry and "word2cam" in entry and not warned_word2cam:
                warnings.warn(
                    "EPFL camera_matrix.json uses the upstream 'word2cam' key; treating it as world2cam.",
                    stacklevel=2,
                )
                warned_word2cam = True
            cam_T_world: Float32[ndarray, "4 4"] = _matrix4(cam_T_world_raw, field_name=f"{camera_name}.world2cam")
            video_path: Path = video_path_for_camera(self.config, camera_name)
            cameras.append(
                _pinhole_from_entry(
                    camera_name=camera_name,
                    entry=entry,
                    video_path=video_path,
                    cam_T_world=cam_T_world,
                )
            )
        return cameras

    def __getitem__(self, idx: int) -> ExoData:
        bgr_list = self.exo_video_readers[idx]
        return ExoData(
            cam_params_list=[cam for cam in self.exo_cam_list if cam is not None],
            bgr_list=bgr_list,
            xyz=None,
            uv_dict=None,
        )

    @property
    def image_plane_distance(self) -> int | float:
        return 0.25
