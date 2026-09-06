"""What every dataforge test module needs: the GPU-encoder gate, rrd read-back, and synthetic frames.

The read-back helpers go through the *public* reader (``RrdReader`` → ``ChunkStore``
→ a datafusion view over one index), so a test asserts what a consumer sees rather
than what the writer intended. The frame builders are shared because their only
job is to make consecutive frames differ; the sizes stay with the modules, which
each have their own reason for theirs.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
from jaxtyping import UInt8
from numpy import ndarray

from dataforge import schema
from dataforge.logging_toolkit import require_av1_nvenc, resolve_ffmpeg

NOISE_CEILING: int = 96
"""Upper bound of the per-pixel noise, low enough that gradient + noise cannot wrap."""


@pytest.fixture(scope="session")
def nvenc_ffmpeg() -> Path:
    """The resolved ffmpeg, or a skip when this machine cannot encode AV1 on the GPU."""
    ffmpeg: Path = resolve_ffmpeg()
    try:
        require_av1_nvenc(ffmpeg)
    except RuntimeError as error:
        pytest.skip(f"no av1_nvenc: {error}")
    return ffmpeg


def gray_frame(index: int, *, width: int, height: int, noisy: bool = False) -> UInt8[ndarray, "height width"]:
    """A horizontal gradient with a square that moves with ``index``, so no two frames match.

    Args:
        index: Frame number; it drives the square's position and its tint.
        width: Frame width in pixels.
        height: Frame height in pixels.
        noisy: Add sensor-like noise. A flat gradient compresses to a couple of
            kilobytes, and msd's split-archive fixture needs PNG frames that
            really do span volumes.

    Returns:
        One grayscale frame.
    """
    ceiling: int = 255 - NOISE_CEILING if noisy else 255
    frame: UInt8[ndarray, "height width"] = np.tile(np.linspace(0, ceiling, width, dtype=np.uint8), (height, 1))
    if noisy:
        frame = frame + np.random.default_rng(index).integers(0, NOISE_CEILING, (height, width), dtype=np.uint8)
    left: int = (index * 4) % (width - 16)
    frame[8:24, left : left + 16] = np.uint8(255 - (index * 7) % 256)
    return frame


def png_frame(index: int, *, width: int, height: int, noisy: bool = False) -> bytes:
    """One ``gray_frame`` as encoded PNG bytes — the ``image2pipe`` input form."""
    success, buffer = cv2.imencode(".png", gray_frame(index, width=width, height=height, noisy=noisy))
    assert success
    return buffer.tobytes()


def read_back(rrd: Path) -> rr.experimental.ChunkStore:
    """Load a saved rrd the way a consumer does: reader → store → queryable views.

    The stream is materialized because ``from_chunks`` declares ``Sequence[Chunk]``;
    these recordings are a few dozen rows, so the list costs nothing.
    """
    return rr.experimental.ChunkStore.from_chunks(list(rr.experimental.RrdReader(rrd).stream()))


def column_rows(store: rr.experimental.ChunkStore, column: str) -> pa.Table:
    """Non-null rows of one component column, index-sorted."""
    table: pa.Table = store.reader(index=schema.TIMELINE).to_arrow_table().sort_by(schema.TIMELINE)
    return table.select([schema.TIMELINE, column]).drop_null()
