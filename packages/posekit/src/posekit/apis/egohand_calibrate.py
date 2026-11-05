"""Regenerate EgoHand activation ranges from the held-out Rerun catalog split."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from einops import rearrange
from jaxtyping import Float64, UInt8
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry
from simplecv.catalog_video import CatalogVideo, read_catalog_videos
from simplecv.catalog_video_codec import wrap_mp4
from simplecv.rrd_query_utils import first_valid_value
from torch import Tensor
from torchcodec.decoders import VideoDecoder

from posekit.apis.hands_bench import EXAMPLES
from posekit.cameras import CalibratedCamera
from posekit.models.egohand_quantization import EgoHandCalibration, load_calibration, measure_calibration, write_calibration


@dataclass(frozen=True, slots=True)
class Config:
    """Explicit calibration output; source samples come from an auditable split record."""

    output: Path
    """.rrd destination, supplied explicitly so the committed calibration is preserved."""
    catalog_url: str = "rerun+http://127.0.0.1:51235"
    """Catalog containing the calibration segment and its factory lens."""
    dataset: str = "dataforge-selfcap"
    """Registered dataset name."""
    pinhole: str = "/world/rig_04/cam_01/pinhole"
    """OAK-left calibrated camera entity; its video is the /video child."""
    split: Path | None = None
    """Calibration record whose segment, frame indices and excluded mount-fit indices are reused."""
    evaluation_video: Path = EXAMPLES / "selfcap_oak_left.mp4"
    """Every decoded benchmark frame is excluded by SHA256, regardless of source index."""
    assets: Path | None = None
    """Optional local EgoHand asset tree; otherwise use the pinned releases."""


@torch.inference_mode()
def main(config: Config) -> None:
    """Read source calibration and video from the catalog, measure BF16 ranges, and save an .rrd."""
    split: EgoHandCalibration = load_calibration(config.split)
    dataset: DatasetEntry = CatalogClient(config.catalog_url).get_dataset(name=config.dataset)
    statics: pa.Table = dataset.filter_segments(split.segment).filter_contents([config.pinhole]).reader(index=None).to_arrow_table()
    intrinsic: Float64[ndarray, "3 3"] = np.asarray(first_valid_value(statics[f"{config.pinhole}:Pinhole:image_from_camera"]), np.float64).reshape(3, 3).T
    distortion: Float64[ndarray, "14"] = np.asarray(first_valid_value(statics[f"{config.pinhole}:simplecv.components.DistortionCoefficients"]), np.float64)
    model: list[str] = first_valid_value(statics[f"{config.pinhole}:simplecv.components.DistortionModel"])
    if model != ["brown_conrady"] or distortion.shape != (14,):
        raise ValueError(f"{config.pinhole}: EgoHand's committed calibration split requires the OAK 14-term rational factory lens.")
    resolution: list[float] = first_valid_value(statics[f"{config.pinhole}:Pinhole:resolution"])
    camera = CalibratedCamera(intrinsic, distortion, int(resolution[0]), int(resolution[1]))
    video: CatalogVideo = read_catalog_videos(dataset, split.segment, [f"{config.pinhole}/video"], "video_time")[0]
    fps: float = 1e9 / float(np.median(np.diff(video.t_ns)))
    decoder = VideoDecoder(wrap_mp4(video.samples, video.keyframes, round(fps), codec=video.codec), device="cuda", seek_mode="approximate")
    frames: UInt8[Tensor, "f h w 3"] = rearrange(decoder.get_frames_at(split.frame_indices).data, "f c h w -> f h w c").contiguous()
    evaluation = VideoDecoder(config.evaluation_video, device="cuda", dimension_order="NHWC")
    forbidden: list[str] = [hashlib.sha256(frame.cpu().numpy().tobytes()).hexdigest() for frame in evaluation[:]]
    measured: EgoHandCalibration = measure_calibration(
        zip(split.frame_indices, frames, strict=True), camera, segment=split.segment,
        excluded_mount_indices=split.excluded_mount_indices, excluded_frame_sha256=forbidden, assets=config.assets,
    )
    write_calibration(config.output, measured)
    print(f"[egohand-calibrate] wrote {measured.n_crops} crops, {len(measured.input_amax)} linear ranges and {len(measured.attention_amax)} attention ranges to {config.output}")
