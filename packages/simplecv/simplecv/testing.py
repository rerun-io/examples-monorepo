"""Test support for catalog video: synthetic encoded video, and the whole-column read a reader is checked against.

Import only from tests: the whole-column read holds every sample of a stream, the very thing the reader avoids.
"""

from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import UInt8
from numpy import ndarray
from pyarrow import ChunkedArray
from rerun.catalog import DatasetEntry

from simplecv.catalog_video import CatalogVideo, catalog_video_from_table
from simplecv.rrd_query_utils import RRDQuerySession, first_valid_value


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


def read_video_stream_from_rrd(
    rrd_path: str, video_entity: str, timeline: str
) -> tuple[rr.VideoCodec, ChunkedArray, ChunkedArray]:
    """Read a ``rr.VideoStream`` entity back from an ``.rrd`` recording.

    Args:
        rrd_path: Path to the recording on disk.
        video_entity: Entity path where ``rr.VideoStream`` was logged.
        timeline: Timeline used as the sample index when the stream was logged.

    Returns:
        ``(codec, times, samples)``. ``codec`` is the static codec component.
        ``times`` is the per-sample timeline column (nanoseconds). ``samples``
        is the per-sample encoded byte column (Annex B for H.264/H.265,
        OBU/IVF-style for AV1/VP9).
    """
    normalized_entity: str = video_entity.lstrip("/")
    query_session = RRDQuerySession(Path(rrd_path))

    codec_table = query_session.read_arrow(
        contents=normalized_entity,
        selectors=[f"{normalized_entity}:VideoStream:codec"],
        index=None,
    )
    if codec_table.num_rows == 0:
        codec_table = query_session.read_arrow(
            contents=normalized_entity,
            selectors=[f"{normalized_entity}:VideoStream:codec"],
            index=timeline,
        )
        codec_column = codec_table.column(1) if codec_table.num_columns > 1 else codec_table.column(0)
    else:
        codec_column = codec_table.column(0)

    if codec_table.num_rows == 0:
        raise ValueError(f"There's no video stream codec specified at {video_entity} for timeline {timeline}.")

    codec_value_raw = first_valid_value(
        codec_column,
        component_name=f"{normalized_entity}:VideoStream:codec",
    )
    codec_value: int = int(np.asarray(codec_value_raw).reshape(-1)[0])
    codec: rr.VideoCodec = rr.VideoCodec(codec_value)

    timestamps_and_samples = query_session.read_arrow(
        contents=normalized_entity,
        selectors=[f"{normalized_entity}:VideoStream:sample"],
        index=timeline,
    )
    if timestamps_and_samples.num_rows == 0:
        raise ValueError(f"No video samples found at {video_entity} for timeline {timeline}.")

    times: ChunkedArray = timestamps_and_samples.column(0)
    samples: ChunkedArray = timestamps_and_samples.column(1)

    return codec, times, samples
