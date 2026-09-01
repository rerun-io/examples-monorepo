"""NVDEC video streams over the catalog's exo cameras.

Split from ``catalog_io`` so CLIs that never decode video (report, blueprint,
sweep, html_report) do not pay the torch + torchcodec import cost (~1.2 s warm).
"""

from dataclasses import dataclass

import torch
from jaxtyping import Shaped
from numpy import ndarray
from rerun.catalog import DatasetEntry
from torchcodec.decoders import VideoDecoder

from exo_calib.catalog_io import EXO_CAMERA_NAMES, TIMELINE


@dataclass(slots=True)
class ExoVideoStreams:
    """Per-camera NVDEC decoders over one segment's exo videos."""

    names: tuple[str, ...]
    """Camera entity names, index-aligned with ``decoders``."""
    times_ns: list[Shaped[ndarray, " n_samples"]]
    """Per-camera sample timestamps (timedelta64[ns], timeline order)."""
    decoders: list[VideoDecoder]
    """One whole-segment torchcodec GPU decoder per camera."""

    def frame_rgb_hw3(self, cam_idx: int, sample_idx: int) -> ndarray:
        """Decode one frame to a uint8 RGB HWC numpy array."""
        frame_chw: torch.Tensor = self.decoders[cam_idx][sample_idx]  # pyrefly: ignore[bad-index]  # TorchCodec accepts Python int indices.
        return frame_chw.permute(1, 2, 0).contiguous().cpu().numpy()


def open_exo_streams(
    dataset: DatasetEntry,
    segment_id: str,
    device: str = "cuda",
    fps: int = 30,
    names: tuple[str, ...] = EXO_CAMERA_NAMES,
) -> ExoVideoStreams:
    """Open one NVDEC decoder per exo camera from catalog video packets.

    Args:
        dataset: Catalog dataset entry.
        segment_id: Segment whose packets are fetched.
        device: Decode device (``cuda`` uses NVDEC).
        fps: Nominal frame rate written into the wrapping MP4 track.
        names: Camera entity names under ``/world``.

    Returns:
        Decoders and per-camera sample timestamps.
    """
    from simplecv.rerun_dataloader import open_segment_decoder

    times_list: list[Shaped[ndarray, " n_samples"]] = []
    decoder_list: list[VideoDecoder] = []
    for name in names:
        times, _samples, _keyframes, decoder = open_segment_decoder(
            dataset, segment_id, f"world/{name}/pinhole/video", TIMELINE, torch.device(device), fps
        )
        times_list.append(times)
        decoder_list.append(decoder)
    return ExoVideoStreams(names=names, times_ns=times_list, decoders=decoder_list)
