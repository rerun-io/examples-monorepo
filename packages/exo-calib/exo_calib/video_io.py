"""NVDEC video streams over the catalog's exo cameras.

Split from ``catalog_io`` so CLIs that never decode video (report, blueprint)
do not pay the torch + torchcodec import cost (~1.2 s warm).
"""

from dataclasses import dataclass

import torch
from jaxtyping import Shaped, UInt8
from numpy import ndarray
from rerun.catalog import DatasetEntry
from simplecv.catalog_video_reader import ChunkedVideoDecoder
from torch import Tensor

from exo_calib.layer_io import TIMELINE


@dataclass(slots=True)
class ExoVideoStreams:
    """Per-camera NVDEC decoders over one segment's exo videos."""

    names: tuple[str, ...]
    """Camera entity names, index-aligned with ``decoders``."""
    times_ns: list[Shaped[ndarray, " n_samples"]]
    """Per-camera sample timestamps (timedelta64[ns], timeline order)."""
    decoders: list[ChunkedVideoDecoder]
    """One chunked decoder per camera, over a reader the cameras share."""

    def frame_rgb(self, cam_idx: int, sample_idx: int) -> UInt8[ndarray, "h w 3"]:
        """Decode one frame to a uint8 RGB HWC numpy array."""
        frame: UInt8[Tensor, "3 h w"] = self.decoders[cam_idx].get_frame_at(sample_idx)
        return frame.permute(1, 2, 0).contiguous().cpu().numpy()


def open_exo_streams(dataset: DatasetEntry, segment_id: str, names: tuple[str, ...]) -> ExoVideoStreams:
    """Open one decoder per exo camera, on NVDEC whenever CUDA is available.

    One catalog query fetches a chunk of every camera; each chunk is muxed in
    its stream's codec (``VideoStream:codec``: AV1, H.264 or H.265).

    Args:
        dataset: Catalog dataset entry.
        segment_id: Segment whose packets are fetched.
        names: Camera entity names under ``/world``.

    Returns:
        Decoders and per-camera sample timestamps.
    """
    from simplecv.catalog_video_reader import open_decoders, select_decode_device

    videos: list[str] = [f"world/{name}/pinhole/video" for name in names]
    decoders: list[ChunkedVideoDecoder] = list(open_decoders(dataset, segment_id, videos, TIMELINE, torch.device(select_decode_device())))
    return ExoVideoStreams(names=names, times_ns=[decoder.index.times for decoder in decoders], decoders=decoders)
