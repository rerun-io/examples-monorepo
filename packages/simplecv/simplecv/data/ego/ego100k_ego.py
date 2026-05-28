from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from datasets import Features, Value, load_dataset
from huggingface_hub import hf_hub_download
from jaxtyping import Float32
from numpy import ndarray

from simplecv.camera_parameters import Extrinsics, Fisheye62Parameters, Intrinsics, KannalaBrandtDistortion
from simplecv.data.ego.base_ego import BaseEgoSequence, CamNameType, EgoData
from simplecv.video_io import VideoReader

if TYPE_CHECKING:
    from simplecv.data.exoego.ego100k import Egocentric100KConfig
else:
    # Avoid circular import at runtime; beartype needs a concrete symbol.
    from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig as Egocentric100KConfig

features = Features(
    {
        "mp4": Value("binary"),
        "json": {
            "clip_number": Value("int64"),
            "codec": Value("string"),
            "duration_sec": Value("float64"),
            "factory_id": Value("string"),
            "fps": Value("float64"),
            "height": Value("int64"),
            "size_bytes": Value("int64"),
            "width": Value("int64"),
            "worker_id": Value("string"),
        },
        "__key__": Value("string"),
        "__url__": Value("string"),
    }
)


def _load_intrinsics_from_hub(cfg: "Egocentric100KConfig") -> Fisheye62Parameters:
    """Download and parse the worker intrinsics.json into a fisheye camera."""
    intrinsics_path: Path = Path(
        hf_hub_download(
            repo_id=cfg.repo_id,
            filename=f"{cfg.factory_id}/{cfg.worker_id}/intrinsics.json",
            repo_type="dataset",
        )
    )
    intrinsics_json: dict[str, float | int | str]
    with intrinsics_path.open() as f:
        intrinsics_json = json.load(f)

    height: int = int(intrinsics_json["image_height"])
    width: int = int(intrinsics_json["image_width"])
    intrinsics = Intrinsics(
        camera_conventions="RDF",
        fl_x=float(intrinsics_json["fx"]),
        fl_y=float(intrinsics_json["fy"]),
        cx=float(intrinsics_json["cx"]),
        cy=float(intrinsics_json["cy"]),
        height=height,
        width=width,
    )
    cam_R_world: Float32[ndarray, "3 3"] = np.eye(3, dtype=np.float32)
    cam_t_world: Float32[ndarray, "3"] = np.zeros(3, dtype=np.float32)
    extrinsics = Extrinsics(
        cam_R_world=cam_R_world,
        cam_t_world=cam_t_world,
    )
    distortion = KannalaBrandtDistortion(
        k1=float(intrinsics_json.get("k1", 0.0)),
        k2=float(intrinsics_json.get("k2", 0.0)),
        k3=float(intrinsics_json.get("k3", 0.0)),
        k4=float(intrinsics_json.get("k4", 0.0)),
    )

    return Fisheye62Parameters(
        name="ego_fisheye",
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        distortion=distortion,
    )


class Ego100KEgoSequence(BaseEgoSequence["Egocentric100KConfig"]):
    """Egocentric view loader for Egocentric-100K (single camera, streaming)."""

    def __init__(self, cfg: "Egocentric100KConfig") -> None:
        self._temp_video_paths: list[Path] = []
        super().__init__(cfg)

    def __len__(self) -> int:
        return len(self.ego_video_readers)

    def __getitem__(self, idx: int) -> EgoData:
        cam_params_list = [cam_list[idx] for cam_list in self._ego_cam_dict.values()]
        return EgoData(
            cam_params_list=cam_params_list,
            bgr_list=self.ego_video_readers[idx],
        )

    def load_video_paths(self) -> list[Path]:
        """Stream the first clip for the configured worker and materialize it locally.

        We rely on ``datasets.load_dataset`` in streaming mode so the tar shard
        is read lazily from the Hub and only the first matching sample is
        downloaded. The video bytes are written to a temporary ``.mp4`` so the
        existing ``MultiVideoReader`` stack can consume it without further
        changes.
        """
        tmp_dir: Path = self.config.tmp_dir or Path(tempfile.gettempdir())
        tmp_dir.mkdir(parents=True, exist_ok=True)

        data_files: str = f"{self.config.factory_id}/{self.config.worker_id}/*.tar"
        iterable = load_dataset(
            self.config.repo_id,
            data_files=data_files,
            streaming=True,
            features=features,
        )
        stream = iterable["train"] if isinstance(iterable, dict) else iterable
        try:
            sample = next(iter(stream))
        except StopIteration as exc:
            raise RuntimeError(
                f"No samples found for pattern {data_files} in {self.config.repo_id}"
            ) from exc

        video_bytes_obj = sample["mp4"]
        if not isinstance(video_bytes_obj, (bytes, bytearray)):
            raise TypeError(f"Expected mp4 bytes, received {type(video_bytes_obj)}")
        video_bytes: bytes = bytes(video_bytes_obj)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=tmp_dir) as tmp_file:
            tmp_file.write(video_bytes)
            video_path: Path = Path(tmp_file.name)

        self._temp_video_paths.append(video_path)
        return [video_path]

    def load_ego_cams(self) -> dict[str, list[Fisheye62Parameters]]:
        camera_params: Fisheye62Parameters = _load_intrinsics_from_hub(self.config)
        cam_name: str = self.config.worker_id or "ego_cam"
        cam_dict: dict[str, list[Fisheye62Parameters]] = {cam_name: [camera_params]}
        return cam_dict

    def align_cams_and_videos(
        self,
        video_path_list: list[Path],
        ego_cam_dict: dict[CamNameType, list[Fisheye62Parameters]],
    ) -> tuple[dict[CamNameType, list[Fisheye62Parameters]], dict[CamNameType, Path]]:
        """Align the single camera with the streamed video and repeat params per frame."""
        assert video_path_list, "No ego video path provided."
        assert ego_cam_dict, "No ego camera provided."

        video_path: Path = video_path_list[0]
        cam_name: CamNameType = next(iter(ego_cam_dict.keys()))
        cam_params: list[Fisheye62Parameters] = ego_cam_dict[cam_name]
        assert cam_params, "Expected at least one camera parameter entry."

        video_reader = VideoReader(video_path)
        video_len: int = len(video_reader)
        if len(cam_params) < video_len:
            cam_params = cam_params + [cam_params[-1]] * (video_len - len(cam_params))
        elif len(cam_params) > video_len:
            cam_params = cam_params[:video_len]

        aligned_cam_dict: dict[CamNameType, list[Fisheye62Parameters]] = {cam_name: cam_params}
        aligned_video_map: dict[CamNameType, Path] = {cam_name: video_path}
        return aligned_cam_dict, aligned_video_map

    @property
    def image_plane_distance(self) -> int | float:
        """Set a small positive plane distance for projection helpers."""
        return 0.05
