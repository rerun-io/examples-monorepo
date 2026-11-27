"""Test support for catalog video: synthetic encoded video, and the whole-column read a reader is checked against.

Import only from tests: the whole-column read holds every sample of a stream, the very thing the reader avoids.
"""

from collections.abc import Sequence
from fractions import Fraction

import av
import numpy as np
import pyarrow as pa
from jaxtyping import UInt8
from numpy import ndarray
from rerun.catalog import DatasetEntry

from simplecv.catalog_video import CatalogVideo, catalog_video_from_table


def encode_h264(frames: int, gop: int, seed: int = 0, *, width: int = 96, height: int = 64) -> list[tuple[bytes, bool]]:
    """Moving noise (every frame distinct), closed GOPs of ``gop`` frames, no B-frames: one ``(sample, is_keyframe)`` a frame."""
    codec = av.CodecContext.create("libx264", "w")
    codec.width, codec.height, codec.pix_fmt = width, height, "yuv420p"
    codec.time_base = Fraction(1, 30)
    codec.options = {"g": str(gop), "keyint_min": str(gop), "bf": "0", "sc_threshold": "0"}
    base: UInt8[ndarray, "h w 3"] = np.random.default_rng(seed).integers(0, 255, (height, width, 3), dtype=np.uint8)
    samples: list[tuple[bytes, bool]] = []
    for index in range(frames):
        frame = av.VideoFrame.from_ndarray(np.roll(base, 3 * index, axis=1), format="rgb24")
        frame.pts = index
        samples += [(bytes(packet), bool(packet.is_keyframe)) for packet in codec.encode(frame)]
    samples += [(bytes(packet), bool(packet.is_keyframe)) for packet in codec.encode(None)]
    return samples


def whole_column(dataset: DatasetEntry, segment: str, entities: Sequence[str], timeline: str) -> tuple[CatalogVideo, ...]:
    """Every stream's whole column in one query: the reference, independent of the reader under test."""
    paths: list[str] = [f"/{entity.strip('/')}" for entity in entities]
    columns: list[str] = [f"{path}:VideoStream:{component}" for path in paths for component in ("sample", "is_keyframe", "codec")]
    table: pa.Table = dataset.filter_segments(segment).filter_contents(paths).reader(index=timeline).select(timeline, *columns).to_arrow_table()
    return tuple(catalog_video_from_table(table, path, timeline, f"{segment} {path}") for path in paths)
