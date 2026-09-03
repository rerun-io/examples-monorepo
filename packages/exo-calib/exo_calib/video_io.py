"""NVDEC video streams over the catalog's exo cameras.

Split from ``catalog_io`` so CLIs that never decode video (report, blueprint)
do not pay the torch + torchcodec import cost (~1.2 s warm).
"""

from dataclasses import dataclass

import torch
from jaxtyping import Shaped, UInt8
from numpy import ndarray
from rerun.catalog import DatasetEntry
from torch import Tensor
from torchcodec.decoders import VideoDecoder

from exo_calib.layer_io import TIMELINE

MP4_NOMINAL_FPS: int = 30
"""Frame rate written into the MP4 wrapper around the catalog packets. Frames are addressed by sample
index and timestamps come from the catalog, so this only labels the container; it is not a capture property."""


@dataclass(slots=True)
class ExoVideoStreams:
    """Per-camera NVDEC decoders over one segment's exo videos."""

    names: tuple[str, ...]
    """Camera entity names, index-aligned with ``decoders``."""
    times_ns: list[Shaped[ndarray, " n_samples"]]
    """Per-camera sample timestamps (timedelta64[ns], timeline order)."""
    decoders: list[VideoDecoder]
    """One whole-segment torchcodec GPU decoder per camera."""

    def frame_rgb(self, cam_idx: int, sample_idx: int) -> UInt8[ndarray, "h w 3"]:
        """Decode one frame to a uint8 RGB HWC numpy array."""
        frame: UInt8[Tensor, "3 h w"] = self.decoders[cam_idx].get_frame_at(sample_idx).data
        return frame.permute(1, 2, 0).contiguous().cpu().numpy()


def open_exo_streams(dataset: DatasetEntry, segment_id: str, names: tuple[str, ...]) -> ExoVideoStreams:
    """Open one NVDEC decoder per exo camera from catalog video packets.

    Each stream is fetched with one catalog query and muxed in its own codec
    (``VideoStream:codec``: AV1, H.264 or H.265) by simplecv's segment decoder.

    Args:
        dataset: Catalog dataset entry.
        segment_id: Segment whose packets are fetched.
        names: Camera entity names under ``/world``.

    Returns:
        Decoders and per-camera sample timestamps.
    """
    from simplecv.rerun_dataloader import open_segment_decoder

    times_per_camera: list[Shaped[ndarray, " n_samples"]] = []
    decoders: list[VideoDecoder] = []
    for name in names:
        times, _, _, decoder = open_segment_decoder(dataset, segment_id, f"world/{name}/pinhole/video", TIMELINE, torch.device("cuda"), MP4_NOMINAL_FPS)
        times_per_camera.append(times)
        decoders.append(decoder)
    return ExoVideoStreams(names=names, times_ns=times_per_camera, decoders=decoders)
