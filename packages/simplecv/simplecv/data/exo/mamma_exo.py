from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from simplecv.camera_parameters import PinholeParameters
from simplecv.data.exo.base_exo import BaseExoSequence, ExoData
from simplecv.data.exoego.mamma import MammaSequenceLayout, pinhole_from_camera_npz, resolve_sequence_layout

if TYPE_CHECKING:
    from simplecv.data.exoego.mamma import MammaConfig
else:
    from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig as MammaConfig


class MammaExoSequence(BaseExoSequence[MammaConfig]):
    """Static exocentric RGB streams for MAMMA (IOI rig or iPhone captures)."""

    @cached_property
    def _layout(self) -> MammaSequenceLayout:
        """Calibration dir, video dir, and camera names — resolved once per sequence."""
        return resolve_sequence_layout(self.config)

    def load_video_paths(self) -> list[Path]:
        video_paths: list[Path] = []
        for camera_name in self._layout.camera_names:
            video_path: Path = self._layout.video_dir / f"{camera_name}.mp4"
            if not video_path.exists():
                raise FileNotFoundError(f"MAMMA exo video not found for {camera_name}: {video_path}")
            video_paths.append(video_path)
        return video_paths

    def load_exo_cams(self) -> list[PinholeParameters | None]:
        cameras: list[PinholeParameters | None] = []
        for camera_name in self._layout.camera_names:
            npz_path: Path = self._layout.calib_dir / f"{camera_name}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"MAMMA calibration NPZ not found for {camera_name}: {npz_path}")
            cameras.append(pinhole_from_camera_npz(npz_path, self._layout.video_dir / f"{camera_name}.mp4"))
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
        return 0.5
