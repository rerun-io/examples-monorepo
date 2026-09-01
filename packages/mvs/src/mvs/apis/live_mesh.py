"""ARKitScenes catalog frames through the Rerun dataloader, decoded on the GPU.

The Rerun iterable dataset drives iteration (sampling grid, segment handling,
prefetch); video frames come from torchcodec/NVDEC through simplecv's
``SegmentNvdecDecoder``, which serves every sample from one whole-segment GPU
decoder instead of re-cutting GOP windows.
Rerun logging is temporarily removed while the pipeline stages come together.
"""

from dataclasses import dataclass, field

import torch
from rerun.catalog import CatalogClient, DatasetEntry
from rerun.experimental.dataloader import (
    DataSource,
    Field,
    FixedRateSampling,
    NoShuffle,
    RerunIterableDataset,
)
from simplecv.rerun_dataloader import SegmentNvdecDecoder
from simplecv.rerun_log_utils import RerunTyroConfig
from tqdm import tqdm

CLOUD_CATALOG_URL: str = "rerun://api.latest.cloud.rerun.io:443"
# ARKitScenes rig schema (see arkitscenes-download's ingest/paths.py).
VIDEO_WIDE: str = "world/rig_00/cam_00/pinhole/video"
TIMELINE: str = "video_time"
# Catalog segments store the wide camera as 60 fps AV1.
NATIVE_FPS: float = 60.0
FETCH_BLOCK_SIZE: int = 1024
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


def main(config: Config) -> None:
    """Iterate every requested segment through the dataloader, decoding frames on the GPU."""
    client = CatalogClient(config.data.catalog_url)
    dataset: DatasetEntry = client.get_dataset(name=config.data.dataset_name)
    segment_ids: list[str] = list(config.data.segments) or dataset.segment_ids()
    source: DataSource = DataSource(dataset=dataset, segments=segment_ids)
    fields: dict[str, Field] = {
        "video": Field(
            f"/{VIDEO_WIDE}:VideoStream:sample",
            decode=SegmentNvdecDecoder(dataset, VIDEO_WIDE, TIMELINE, torch.device(config.device), int(NATIVE_FPS)),
        ),
    }
    samples: RerunIterableDataset = RerunIterableDataset(
        source,
        index=TIMELINE,
        fields=fields,
        timeline_sampling=FixedRateSampling(rate_hz=NATIVE_FPS),
        shuffle_strategy=NoShuffle(),
        fetch_block_size=FETCH_BLOCK_SIZE,
    )
    frames: int = 0
    for sample in tqdm(samples, unit="frame"):
        if sample["video"] is not None:
            frames += 1
    print(f"decoded {frames} frames on {config.device} across {len(segment_ids)} segment(s)")
