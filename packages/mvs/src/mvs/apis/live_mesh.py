"""ARKitScenes catalog frames through the Rerun dataloader, decoded on the GPU.

The Rerun iterable dataset drives iteration (sampling grid, segment handling,
prefetch); video frames come from torchcodec/NVDEC. Upstream's decoder
(rerun-io/reality PR #2893) wraps each sample's keyframe window into its own
in-memory MP4 and is slower than dav1d — no decode-session reuse across calls.
``SegmentNvdecDecoder`` instead bulk-fetches a segment's packets once, wraps
them into a single MP4 (same ``add_mux_stream`` technique), and serves every
sample from one GPU decoder. It skips keyframe anchoring, so the dataloader
ships each grid slot as a point read instead of re-cutting GOP windows.
Rerun logging is temporarily removed while the pipeline stages come together.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from io import BytesIO

import av
import numpy as np
import pyarrow as pa
import torch
from jaxtyping import UInt8
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry, DatasetView
from rerun.experimental.dataloader import (
    ColumnDecoder,
    DataSource,
    Field,
    FixedRateSampling,
    NoShuffle,
    RerunIterableDataset,
)
from simplecv.rerun_log_utils import RerunTyroConfig
from torch import Tensor
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm

CLOUD_CATALOG_URL: str = "rerun://api.latest.cloud.rerun.io:443"
# ARKitScenes rig schema (see arkitscenes-download's ingest/paths.py).
VIDEO_WIDE: str = "world/rig_00/cam_00/pinhole/video"
TIMELINE: str = "video_time"
# Catalog segments store the wide camera as 60 fps AV1.
NATIVE_FPS: float = 60.0
FETCH_SIZE: int = 1024
"""Samples per catalog query in the iterable dataset. The NVDEC decoder ignores the
shipped payloads, so fewer round-trips win: 256 -> 10.3 s, 1024 -> 3.5 s, 2048 flat."""


@dataclass(frozen=True)
class CatalogDataConfig:
    """ARKitScenes Rerun-catalog input configuration."""

    catalog_url: str = CLOUD_CATALOG_URL
    """Catalog server URL."""
    dataset_name: str = "arkitscenes"
    """Catalog dataset entry name."""
    segments: tuple[str, ...] = ("42899799",)
    """Segment ids to process sequentially; empty runs every registered segment."""


@dataclass
class Config:
    """Configuration for the live meshing tool."""

    rr_config: RerunTyroConfig
    """Rerun viewer/save/connect behavior."""
    data: CatalogDataConfig = field(default_factory=CatalogDataConfig)
    """Where the catalog segments come from."""
    device: str = "cuda"
    """Device the decoded frames land on."""


def open_segment_decoder(
    dataset: DatasetEntry, segment_id: str, entity: str, device: torch.device
) -> tuple[ndarray, list[bytes], list[bool], VideoDecoder]:
    """Fetch one segment's video packets (one query), wrap them, and open a GPU decoder.

    Returns the sample timestamps (timedelta64[ns], timeline order), the raw AV1
    samples with their keyframe flags (relayable as a Rerun VideoStream), and the
    decoder over the whole segment.
    """
    view: DatasetView = dataset.filter_segments(segment_id).filter_contents(entity)
    table = (
        view.reader(index=TIMELINE)
        .select(TIMELINE, f"/{entity}:VideoStream:sample", f"/{entity}:VideoStream:is_keyframe")
        .sort(TIMELINE)
        .to_arrow_table()
    )
    times: ndarray = np.array([t.value for t in table[0]], dtype="timedelta64[ns]")
    blobs = table[1].combine_chunks().flatten()
    data = memoryview(blobs.flatten().buffers()[1])
    offsets: list[int] = blobs.offsets.to_pylist()
    samples: list[bytes] = [bytes(data[start:end]) for start, end in zip(offsets[:-1], offsets[1:], strict=True)]
    keyframes: list[bool] = [bool(flag) for flag in table[2].combine_chunks().flatten().to_pylist()]
    decoder: VideoDecoder = VideoDecoder(
        wrap_mp4(samples, keyframes), device=device, seek_mode="exact", num_ffmpeg_threads=0
    )
    return times, samples, keyframes, decoder


def wrap_mp4(samples: list[bytes], keyframes: list[bool], fps: int = int(NATIVE_FPS)) -> bytes:
    """Mux pre-encoded AV1 samples into an in-memory MP4 with positional pts (no re-encode).

    ``add_mux_stream`` muxes without instantiating an encoder. The nominal 16x16
    stream dimensions are irrelevant: decoders read the real dimensions from the
    bitstream (parameter sets travel in-band).
    """
    buffer: BytesIO = BytesIO()
    # Pin the mp4 track timescale to fps: the muxer otherwise picks its own (15360)
    # without rescaling our positional pts, and the track then claims a ~0.1s
    # duration. Index-seeking decoders never notice; timestamp readers do.
    with av.open(buffer, "w", format="mp4", options={"video_track_timescale": str(fps)}) as container:
        stream = container.add_mux_stream("av1", rate=fps, width=16, height=16)
        stream.time_base = Fraction(1, fps)
        for sample_index, (sample, is_keyframe) in enumerate(zip(samples, keyframes, strict=True)):
            packet: av.Packet = av.Packet(sample)
            packet.pts = packet.dts = sample_index
            packet.duration = 1
            packet.time_base = stream.time_base
            packet.stream = stream
            packet.is_keyframe = is_keyframe
            container.mux(packet)
    return buffer.getvalue()


class SegmentNvdecDecoder(ColumnDecoder):
    """Serve the dataloader's per-sample decode calls from one whole-segment GPU decoder.

    The base-class defaults (no ``prior_keyframe_path``, no ``context_range``) make the
    dataloader ship each grid slot as a point read; the shipped bytes are ignored and the
    frame comes from the cached segment-wide NVDEC decoder instead.
    """

    def __init__(self, dataset: DatasetEntry, entity: str, device: torch.device) -> None:
        self._dataset = dataset
        self._entity = entity
        self._device = device
        self._segment_id: str | None = None
        self._decoder: VideoDecoder | None = None
        self.times: ndarray = np.empty(0, dtype="timedelta64[ns]")
        """The current segment's sample timestamps (timedelta64[ns], timeline order)."""
        self.samples: list[bytes] | None = None
        """The current segment's raw AV1 samples, relayable as a Rerun VideoStream."""
        self.keyframes: list[bool] | None = None
        """Keyframe flag per raw sample."""

    def decode(
        self, raw: pa.ChunkedArray, index_value: int | np.datetime64 | np.timedelta64, segment_id: str
    ) -> Tensor | None:
        """Return the segment's latest frame at or before *index_value* as a uint8 RGB CHW device tensor."""
        del raw  # decoded from the segment-wide cache, not the shipped point read
        if segment_id != self._segment_id:
            self.times, self.samples, self.keyframes, self._decoder = open_segment_decoder(
                self._dataset, segment_id, self._entity, self._device
            )
            self._segment_id = segment_id
        frame_index: int = int(np.searchsorted(self.times, index_value, side="right")) - 1
        if frame_index < 0:
            return None
        assert self._decoder is not None
        frame_chw: UInt8[Tensor, "3 h w"] = self._decoder.get_frame_at(frame_index).data
        return frame_chw


def main(config: Config) -> None:
    """Iterate every requested segment through the dataloader, decoding frames on the GPU."""
    client = CatalogClient(config.data.catalog_url)
    dataset: DatasetEntry = client.get_dataset(name=config.data.dataset_name)
    segment_ids: list[str] = list(config.data.segments) or dataset.segment_ids()
    source: DataSource = DataSource(dataset=dataset, segments=segment_ids)
    fields: dict[str, Field] = {
        "video": Field(
            f"/{VIDEO_WIDE}:VideoStream:sample", decode=SegmentNvdecDecoder(dataset, VIDEO_WIDE, torch.device(config.device))
        ),
    }
    samples: RerunIterableDataset = RerunIterableDataset(
        source,
        index=TIMELINE,
        fields=fields,
        timeline_sampling=FixedRateSampling(rate_hz=NATIVE_FPS),
        shuffle_strategy=NoShuffle(),  # pyrefly: ignore  # unexpected-keyword — pyrefly resolves rerun stubs from an older env; mvs runs the 0.36 prerelease API
        fetch_size=FETCH_SIZE,
    )
    frames: int = 0
    for sample in tqdm(samples, unit="frame"):
        if sample["video"] is not None:
            frames += 1
    print(f"decoded {frames} frames on {config.device} across {len(segment_ids)} segment(s)")
