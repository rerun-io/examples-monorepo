"""The mask layer of a segment: per camera, the segmentation mask the frontend applies to each frame.

The layer contract (shared with colsfm's keypoint masks): per camera ``C``, a temporal ``SegmentationImage`` (uint8,
nonzero = no features there) at ``C/pinhole/<name>``, at the resolution it was computed at, on that camera's
``video_time`` stamps, with an optional static ``Transform3D(scale=[W_img / W_mask, H_img / H_mask, 1])`` on the same
entity (none = scale 1). Masks may be sparser than the video: each frame takes the latest mask at or before it, and a
frame before a camera's first mask is unmasked. The core reads the mask through its scale
(``slam_rs._core.Vio.track``); this module only checks that the scale puts the mask on the image.

``FrameMasks`` holds the layer's stamps; each frameset iterator reads the pixels through its own ``MaskCursor``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pyarrow as pa
from datafusion import col, lit
from jaxtyping import Int64, UInt8
from numpy import ndarray
from rerun.catalog import DatasetEntry
from simplecv.rrd_query_utils import first_valid_value

from slam_rs.rig import TIMELINE

SCALE_TOLERANCE_PX: float = 0.5
"""How far ``mask size x scale`` may land from the image size: a 512 mask over a 1920 image has scale 3.75 exactly, but a
writer that rounds the scale lands within a fraction of a pixel. The lookup then uses the sizes, not the logged scale."""

MASK_WINDOW_NS: int = 2_000_000_000
"""How far past a frame one read of a camera's masks reaches: a cursor holds about this much of each camera's
uncompressed masks (60 at 30 fps), whatever the video window."""

STAMP_WINDOW_NS: int = 10_000_000_000
"""The span of one stamps read. The catalog sends whole chunks, pixels included, even when only the index and the format
are selected: one read of clip 007's 1920 px people layer held 12.5 GB, 10 s windows hold 0.65 GB at most."""


class FrameMasks:
    """One segment's mask layer for the fed cameras: entities and stamps; ``MaskCursor`` reads the pixels."""

    def __init__(self, dataset: DatasetEntry, segment_id: str, entities: Sequence[str], image_sizes: Sequence[tuple[int, int]]) -> None:
        """Read each camera's mask stamps, sizes and scale, and check them against the camera's image.

        Args:
            dataset: Dataset that holds the mask layer.
            segment_id: The segment.
            entities: Mask entity per fed camera, in frameset order.
            image_sizes: ``(width, height)`` of each camera's video frames, in the same order.

        Raises:
            ValueError: When a camera has no mask, or a mask times its scale is not its camera's image size.
        """
        self.dataset: DatasetEntry = dataset
        self.segment_id: str = segment_id
        self.entities: tuple[str, ...] = tuple(entities)
        self.stamps: list[Int64[ndarray, " n"]] = []
        """Per camera, every mask's ``video_time`` in ns, ascending."""
        statics: pa.Table = dataset.filter_segments([segment_id]).filter_contents(list(entities)).reader(index=None).to_arrow_table()
        columns: set[str] = {str(column) for column in dataset.schema().column_names()}
        ranges: pa.Table = dataset.filter_segments([segment_id]).get_index_ranges().to_arrow_table()
        first_ns: int = int(ranges[f"{TIMELINE}:start"].cast(pa.int64())[0].as_py())
        last_ns: int = int(ranges[f"{TIMELINE}:end"].cast(pa.int64())[0].as_py())
        for entity, (width, height) in zip(entities, image_sizes, strict=True):
            scale_column: str = f"{entity}:Transform3D:scale"
            logged: list[float] | None = first_valid_value(statics[scale_column], allow_none=True) if scale_column in statics.column_names else None
            scale: list[float] = [1.0, 1.0] if logged is None else [float(value) for value in logged[:2]]
            format_column: str = f"{entity}:SegmentationImage:format"
            if format_column not in columns:
                raise ValueError(f"{segment_id}: no SegmentationImage at {entity}")
            # Stamps and sizes only; the server still sends the pixels, so one time window at a time.
            table: pa.Table = pa.concat_tables(
                dataset.filter_segments([segment_id])
                .filter_contents([entity])
                .reader(index=TIMELINE)
                .select(TIMELINE, format_column)
                .filter(
                    (col(TIMELINE) >= lit(pa.scalar(start_ns, pa.duration("ns"))))
                    & (col(TIMELINE) < lit(pa.scalar(start_ns + STAMP_WINDOW_NS, pa.duration("ns"))))
                )
                .to_arrow_table()
                for start_ns in range(first_ns, last_ns + 1, STAMP_WINDOW_NS)
            )
            table = table.filter(table[format_column].is_valid())
            if table.num_rows == 0:
                raise ValueError(f"{segment_id}: no SegmentationImage at {entity}")
            formats: list[dict[str, int]] = [row[0] for row in table[format_column].to_pylist()]
            for size in {(fmt["width"], fmt["height"]) for fmt in formats}:
                covered: tuple[float, float] = (size[0] * scale[0], size[1] * scale[1])
                if abs(covered[0] - width) >= SCALE_TOLERANCE_PX or abs(covered[1] - height) >= SCALE_TOLERANCE_PX:
                    raise ValueError(
                        f"{segment_id}: {entity} is {size[0]}x{size[1]} at scale {scale[0]:g}x{scale[1]:g}, which does not cover "
                        f"the {width}x{height} image"
                    )
            self.stamps.append(np.sort(table[TIMELINE].combine_chunks().cast(pa.int64()).to_numpy()))

    def read(self, camera: int, first_ns: int, last_ns: int) -> tuple[Int64[ndarray, " n"], list[UInt8[ndarray, "mh mw"]]]:
        """Read the masks one camera's frames in ``[first_ns, last_ns]`` (``video_time``) can take.

        That is every mask in the range, plus the latest one before it, which the range's first frames hold.

        Args:
            camera: Index of the camera, in frameset order.
            first_ns: The range's first frame.
            last_ns: The range's last frame.

        Returns:
            The masks' stamps, ascending, and the masks as views into the read table's buffers.
        """
        entity: str = self.entities[camera]
        buffer_column: str = f"{entity}:SegmentationImage:buffer"
        format_column: str = f"{entity}:SegmentationImage:format"
        before: int = int(np.searchsorted(self.stamps[camera], first_ns, side="right")) - 1
        start_ns: int = int(self.stamps[camera][before]) if before >= 0 else first_ns
        table: pa.Table = (
            self.dataset.filter_segments([self.segment_id])
            .filter_contents([entity])
            .reader(index=TIMELINE)
            .select(TIMELINE, buffer_column, format_column)
            .filter(  # a plain comparison on the index column is applied by the server; a cast is not
                (col(TIMELINE) >= lit(pa.scalar(start_ns, pa.duration("ns")))) & (col(TIMELINE) <= lit(pa.scalar(last_ns, pa.duration("ns"))))
            )
            .to_arrow_table()
        )
        if table[format_column].null_count:  # Table.filter copies every mask byte
            table = table.filter(table[format_column].is_valid())
        stamped: list[tuple[int, UInt8[ndarray, "mh mw"]]] = []
        # Views into the table's own buffers, batch by batch (a table's columns may chunk at different rows): no copy.
        for batch in table.to_batches():
            images: pa.Array = batch.column(buffer_column).flatten()  # one image per row -> list<uint8> per row
            stamps_ns: list[int] = batch.column(TIMELINE).cast(pa.int64()).to_pylist()
            for row, (stamp, fmt) in enumerate(zip(stamps_ns, batch.column(format_column).to_pylist(), strict=True)):
                pixels: UInt8[ndarray, " n"] = images[row].values.to_numpy(zero_copy_only=True)
                stamped.append((int(stamp), pixels.reshape(fmt[0]["height"], fmt[0]["width"])))
        stamped.sort(key=lambda item: item[0])
        return np.array([stamp for stamp, _ in stamped], dtype=np.int64), [mask for _, mask in stamped]


class MaskCursor:
    """One frameset iterator's masks: per camera, a short window read ahead of its frames and read again once they
    leave it. Each iterator owns its cursor, so two iterators over one feed never replace each other's masks."""

    def __init__(self, masks: FrameMasks, window_ns: int = MASK_WINDOW_NS) -> None:
        """Start with nothing read.

        Args:
            masks: The segment's mask layer.
            window_ns: How far past a frame one read reaches.
        """
        self.masks: FrameMasks = masks
        self.window_ns: int = window_ns
        cameras: int = len(masks.entities)
        self.ranges: list[tuple[int, int]] = [(1, 0)] * cameras
        """Per camera, the ``[first_ns, last_ns]`` frame range its window was read for; empty until the first read."""
        self.window_stamps: list[Int64[ndarray, " n"]] = [np.empty(0, dtype=np.int64)] * cameras
        self.window_masks: list[list[UInt8[ndarray, "mh mw"]]] = [[] for _ in range(cameras)]

    def at(self, stamps: Sequence[int]) -> list[UInt8[ndarray, "mh mw"] | None]:
        """Each camera's latest mask at or before its frame.

        Args:
            stamps: Each camera's frame ``video_time``, in frameset order.

        Returns:
            One C-contiguous mask per camera, or None for a camera with no mask yet.
        """
        held: list[UInt8[ndarray, "mh mw"] | None] = []
        for camera, t_ns in enumerate(stamps):
            first_ns, last_ns = self.ranges[camera]
            if not first_ns <= t_ns <= last_ns:
                self.ranges[camera] = (t_ns, t_ns + self.window_ns)
                self.window_stamps[camera], self.window_masks[camera] = self.masks.read(camera, t_ns, t_ns + self.window_ns)
            index: int = int(np.searchsorted(self.window_stamps[camera], t_ns, side="right")) - 1
            held.append(self.window_masks[camera][index] if index >= 0 else None)
        return held
