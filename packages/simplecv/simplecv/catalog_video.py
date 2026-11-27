"""Fetch encoded camera streams together for persistent segment decoders."""

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
from jaxtyping import Bool, Int64, Shaped, UInt8
from numpy import ndarray

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

    Arrow has no ``list<u8>`` to binary cast, so each chunk's child buffer is sliced
    by its own list offsets. Chunk by chunk, nothing is copied and no offset
    overflows: combining the chunks first copied every packet (27 s of a 9.9 GB
    two-lens read) and, without a cast to ``large_list``, overflowed the 32-bit
    offsets past 2 GiB (a 78-minute RoboCap session). The views keep the column's
    storage alive while they live, and ``av.Packet`` copies from them.
    """
    views: list[UInt8[ndarray, " n_bytes"]] = []
    for chunk in column.chunks:
        blobs: pa.ListArray | pa.LargeListArray = chunk.flatten()
        # ``values`` is the whole child buffer; ``offsets`` index into it, also for a sliced chunk.
        data: UInt8[ndarray, " n_bytes"] = blobs.values.to_numpy(zero_copy_only=True)
        offsets: Int64[ndarray, " n_offsets"] = blobs.offsets.to_numpy(zero_copy_only=True).astype(np.int64)
        views += [data[start:stop] for start, stop in zip(offsets[:-1].tolist(), offsets[1:].tolist(), strict=True)]
    return views


def catalog_codec(column: pa.ChunkedArray, where: str) -> CatalogCodecName:
    """The codec of a ``VideoStream:codec`` column: logged once as a static, so row zero of the non-null values.

    Raises:
        ValueError: If the stream carries no codec, naming ``where`` it was read.
    """
    codecs: pa.Array = column.combine_chunks().drop_null().flatten()
    if len(codecs) == 0:
        raise ValueError(f"{where}: the video stream carries no codec, so its samples cannot be decoded")
    return catalog_codec_name(int(codecs[0].as_py()))


def catalog_video_from_table(table: pa.Table, entity: str, timeline: str, where: str) -> CatalogVideo:
    """One camera's packets from a table that holds its ``VideoStream`` columns on ``timeline``.

    Raises:
        ValueError: If the camera has no frames with strictly increasing timestamps, or no codec (naming ``where``).
    """
    sample_column: str = f"{entity}:VideoStream:sample"
    # Rows from the other cameras can share timestamps but have null packets.
    # Filter the camera's projected columns before flattening nested buffers; the filter
    # copies every packet (~10x the read on a 17.8 min 1920 px lens), so only when needed.
    camera: pa.Table = table.select([timeline, sample_column, f"{entity}:VideoStream:is_keyframe", f"{entity}:VideoStream:codec"])
    if camera[sample_column].null_count:
        camera = camera.filter(camera[sample_column].is_valid())
    times: Shaped[ndarray, " n_frames"] = camera[timeline].combine_chunks().to_numpy(zero_copy_only=False)
    if not len(times) or not np.all(np.diff(times.view(np.int64)) > 0):
        raise ValueError(f"{where}: expected frames with strictly increasing timestamps")
    codec: CatalogCodecName = catalog_codec(camera[f"{entity}:VideoStream:codec"], where)
    return CatalogVideo(times, packet_views(camera[sample_column]), catalog_keyframes(camera[f"{entity}:VideoStream:is_keyframe"]), codec)
