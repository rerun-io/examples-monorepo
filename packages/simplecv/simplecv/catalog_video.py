"""Fetch encoded camera streams together for persistent segment decoders."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
from jaxtyping import Bool, Int64, Shaped, UInt8
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
    flags: pa.ListArray = column.combine_chunks()
    offsets: Int64[ndarray, " n_offsets"] = np.asarray(flags.offsets, dtype=np.int64)
    # A row's flag is the first element of its list; null and empty rows have no elements.
    has_flag: Bool[ndarray, " n_rows"] = offsets[1:] > offsets[:-1]
    result: Bool[ndarray, " n_rows"] = np.zeros(len(flags), dtype=bool)
    if has_flag.any():
        values: Bool[ndarray, " n_values"] = np.asarray(flags.values.to_numpy(zero_copy_only=False), dtype=bool)
        result[has_flag] = values[offsets[:-1][has_flag]]
    return result.tolist()


def packet_views(column: pa.ChunkedArray) -> list[UInt8[ndarray, " n_bytes"]]:
    """One zero-copy view per encoded packet in a ``VideoStream:sample`` column.

    Arrow has no ``list<u8>`` to binary cast, so the child buffer is sliced by the
    list offsets. ``large_list`` keeps 64-bit offsets: one camera of a multi-hour
    session exceeds ``int32``'s 2 GiB. The views keep the column's storage alive
    while they live, and ``av.Packet`` copies from them, so nothing is duplicated.
    """
    blobs: pa.LargeListArray = column.combine_chunks().cast(pa.list_(pa.large_list(pa.uint8()))).flatten()
    data: UInt8[ndarray, " n_bytes"] = blobs.values.to_numpy(zero_copy_only=True)
    offsets: Int64[ndarray, " n_offsets"] = blobs.offsets.to_numpy(zero_copy_only=True)
    return [data[start:stop] for start, stop in zip(offsets[:-1], offsets[1:], strict=True)]


def catalog_codec(column: pa.ChunkedArray, where: str) -> CatalogCodecName:
    """The codec of a ``VideoStream:codec`` column: logged once as a static, so row zero of the non-null values.

    Raises:
        ValueError: If the stream carries no codec, naming ``where`` it was read.
    """
    codecs: pa.Array = column.combine_chunks().drop_null().flatten()
    if len(codecs) == 0:
        raise ValueError(f"{where}: the video stream carries no codec, so its samples cannot be decoded")
    return catalog_codec_name(int(codecs[0].as_py()))


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
        sample_column: str = f"{entity}:VideoStream:sample"
        # Rows from the other cameras can share timestamps but have null packets.
        # Filter the camera's projected columns before flattening nested buffers.
        camera: pa.Table = table.select([timeline, sample_column, f"{entity}:VideoStream:is_keyframe", f"{entity}:VideoStream:codec"])
        camera = camera.filter(camera[sample_column].is_valid())
        times: Shaped[ndarray, " n_frames"] = camera[timeline].combine_chunks().to_numpy(zero_copy_only=False)
        if not len(times) or not np.all(np.diff(times.view(np.int64)) > 0):
            raise ValueError(f"{segment_id} {entity}: expected frames with strictly increasing timestamps")
        codec: CatalogCodecName = catalog_codec(camera[f"{entity}:VideoStream:codec"], f"{segment_id} {entity}")
        videos.append(CatalogVideo(times, packet_views(camera[sample_column]), catalog_keyframes(camera[f"{entity}:VideoStream:is_keyframe"]), codec))
    return tuple(videos)
