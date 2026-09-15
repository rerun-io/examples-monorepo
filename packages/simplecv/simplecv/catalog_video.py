"""Fetch encoded camera streams together for persistent segment decoders."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
from jaxtyping import Int64, Shaped, UInt8
from numpy import ndarray
from rerun.catalog import DatasetEntry

from simplecv.catalog_video_codec import CatalogCodecName, catalog_codec_name


@dataclass(frozen=True, slots=True)
class CatalogVideo:
    """One camera's encoded packets and original catalog timing."""

    times: Shaped[ndarray, " n_frames"]
    """Original catalog index, preserving timestamp, duration or sequence dtype."""
    samples: list[UInt8[ndarray, " n_bytes"]]
    """Packet views into Arrow storage, without per-packet byte copies."""
    keyframes: list[bool]
    """One keyframe flag per packet, including false for sparse null entries."""
    codec: CatalogCodecName
    """Codec carried by the recording."""

    @property
    def t_ns(self) -> Int64[ndarray, " n_frames"]:
        """Integer index values (nanoseconds for temporal indices)."""
        return self.times.view(np.int64)


def catalog_keyframes(column: pa.ChunkedArray) -> list[bool]:
    """Read packet flags, treating both sparse nulls and explicit false as false."""
    return [bool(value and value[0]) for value in column.combine_chunks().to_pylist()]


def read_catalog_videos(dataset: DatasetEntry, segment_id: str, entities: Sequence[str], timeline: str) -> tuple[CatalogVideo, ...]:
    """Read all requested cameras in one materialization, preserving their individual sample times.

    Holds compressed data for one segment in RAM. No sorting or decoding is
    performed; unordered or duplicate timestamps fail before packet muxing.
    """
    paths: list[str] = [f"/{entity.strip('/')}" for entity in entities]
    columns: list[str] = [f"{entity}:VideoStream:{component}" for entity in paths for component in ("sample", "is_keyframe", "codec")]
    table: pa.Table = dataset.filter_segments(segment_id).filter_contents(paths).reader(index=timeline).select(timeline, *columns).to_arrow_table()
    videos: list[CatalogVideo] = []
    for entity in paths:
        # Rows from the other cameras can share timestamps but have null packets.
        # Filter the camera's projected columns before flattening nested buffers.
        camera: pa.Table = table.select([timeline, *[f"{entity}:VideoStream:{component}" for component in ("sample", "is_keyframe", "codec")]])
        camera = camera.filter(camera[1].is_valid())
        times: Shaped[ndarray, " n_frames"] = camera[0].combine_chunks().to_numpy(zero_copy_only=False)
        if not len(times) or not np.all(np.diff(times.view(np.int64)) > 0):
            raise ValueError(f"{segment_id} {entity}: expected frames with strictly increasing timestamps")
        blobs: pa.LargeListArray = camera[1].combine_chunks().cast(pa.list_(pa.large_list(pa.uint8()))).flatten()
        data: UInt8[ndarray, " n_bytes"] = blobs.values.to_numpy(zero_copy_only=True)
        offsets: Int64[ndarray, " n_offsets"] = blobs.offsets.to_numpy(zero_copy_only=True)
        samples: list[UInt8[ndarray, " n_bytes"]] = [data[start:stop] for start, stop in zip(offsets[:-1], offsets[1:], strict=True)]
        codecs: pa.Array = camera[3].combine_chunks().drop_null().flatten()
        if not len(codecs):
            raise ValueError(f"{segment_id} {entity}: video codec is missing")
        flags: list[bool] = catalog_keyframes(camera[2])
        videos.append(CatalogVideo(times, samples, flags, catalog_codec_name(int(codecs[0].as_py()))))
    return tuple(videos)
