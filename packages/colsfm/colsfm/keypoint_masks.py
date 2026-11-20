"""Per-image keypoint masks: pixels where no keypoint may be stored (people, the operator, the rig itself).

A mask is a boolean image at the keyframe's own resolution, True where keypoints are dropped. Extraction applies it
after the score gate and the top-k (`colsfm.features_native.store_feature_batch`), before anything reaches the
database, so matching, loop closure and mapping never see a masked keypoint. The kept keypoints keep their order and
their descriptors.

Masks come from the catalog: a catalog input names, per camera stream, a `SegmentationImage` entity on the same
stamps as the video (`colsfm.frame_manifest.StreamKeyframes.mask_entity`), nonzero = drop. This is the monorepo's mask
layer contract, shared with slam-rs (`slam_rs.frame_masks`): the mask is stored at the resolution it was computed at,
with a static `Transform3D(scale=[W_img / W_mask, H_img / H_mask, 1])` on the same entity (none = scale 1). Each image
pixel takes the mask pixel that covers its centre, `(2c + 1) * W_mask // (2 * W_img)` per axis, which is how the viewer
draws the scaled mask. `CatalogMasks` reads each stream's masks once, on first use, keeps them bit-packed (one bit per
pixel) and hands each one out at its image's resolution.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import numpy as np
import pyarrow as pa
from jaxtyping import Bool, Float32, UInt8
from numpy import ndarray

from colsfm.frame_manifest import FrameSourceManifest

KeypointMask = Bool[ndarray, "height width"]
"""True where a keypoint is dropped, at the image's own resolution."""


@runtime_checkable
class MaskSource(Protocol):
    """Keypoint masks by image name."""

    def mask(self, name: str, width: int, height: int) -> KeypointMask | None:
        """The mask of one image, or None when the image is not masked.

        Args:
            name: Image name, as `frames_meta.json` and the database hold it.
            width: The image's width in pixels.
            height: The image's height in pixels.

        Returns:
            `[height, width]` booleans, True = drop.
        """
        ...


def keypoints_outside_mask(keypoints_xy: Float32[ndarray, "n 2"], mask: KeypointMask) -> Bool[ndarray, " n"]:
    """Which keypoints survive a mask: those whose nearest pixel is not masked.

    Keypoints are in OpenCV pixel convention (pixel centres on integers), so the pixel is the rounded coordinate,
    clipped into the image.

    Args:
        keypoints_xy: Pixel coordinates.
        mask: True = drop.

    Returns:
        True for each keypoint to keep.
    """
    height, width = mask.shape
    columns: ndarray = np.clip(np.rint(keypoints_xy[:, 0]).astype(np.int64), 0, width - 1)
    rows: ndarray = np.clip(np.rint(keypoints_xy[:, 1]).astype(np.int64), 0, height - 1)
    return ~mask[rows, columns]


def pack_mask_table(table: pa.Table, timeline: str, entity: str) -> dict[int, tuple[int, int, UInt8[ndarray, " n"]]]:
    """Bit-pack every mask of one `SegmentationImage` entity, keyed by its stamp; rows without a mask (null or cleared) are skipped.

    Walks the table's record batches through the list offsets, so no mask byte is copied before `np.packbits`. The
    loader used to `filter` the table to the rows holding a mask first: an Arrow Take over a list-of-list column of
    every mask in the segment, which on clip 007 (six 3,203-mask face streams) was most of the extraction stage.

    Args:
        table: The entity's rows on `timeline`, as the catalog reader returns them.
        timeline: The index column.
        entity: The mask entity path, without the leading slash.

    Returns:
        Stamp (ns) -> (mask width, mask height, `np.packbits` of the flat `!= 0` mask).

    Raises:
        ValueError: When a row holds more than one mask, a mask has no format, or a mask does not hold one byte per pixel.
    """
    packed: dict[int, tuple[int, int, UInt8[ndarray, " n"]]] = {}
    for batch in table.to_batches():
        buffers: pa.ListArray = batch.column(f"/{entity}:SegmentationImage:buffer")
        valid: Bool[ndarray, " rows"] = buffers.is_valid().to_numpy(zero_copy_only=False)
        if not valid.any():
            continue
        stamps: ndarray = batch.column(timeline).to_numpy(zero_copy_only=False).view(np.int64)
        formats: list[list[dict[str, int]] | None] = batch.column(f"/{entity}:SegmentationImage:format").to_pylist()
        # `offsets` and `values` both address the unsliced child arrays, so they compose for a sliced batch as well.
        instance_offsets: ndarray = buffers.offsets.to_numpy()
        instances: pa.ListArray = buffers.values
        byte_offsets: ndarray = instances.offsets.to_numpy()
        data: UInt8[ndarray, " n"] = instances.values.to_numpy(zero_copy_only=False)
        for row in np.flatnonzero(valid):
            stamp: int = int(stamps[row])
            instance: int = int(instance_offsets[row])
            # A cleared row (`[]`) is valid but empty: its start equals the next row's, so it has no mask of its own.
            num_instances: int = int(instance_offsets[row + 1]) - instance
            if num_instances == 0:
                continue
            if num_instances != 1:
                raise ValueError(f"/{entity} at t_ns {stamp}: {num_instances} mask instances, expected 1")
            mask_format: list[dict[str, int]] | None = formats[row]
            if not mask_format or len(mask_format) != 1:
                raise ValueError(f"/{entity} at t_ns {stamp}: a mask with no format (or {len(mask_format or [])}), expected 1")
            pixels: UInt8[ndarray, " n"] = data[byte_offsets[instance] : byte_offsets[instance + 1]]
            mask_width, mask_height = int(mask_format[0]["width"]), int(mask_format[0]["height"])
            if len(pixels) != mask_width * mask_height:
                raise ValueError(f"/{entity} at t_ns {stamp}: a {mask_width}x{mask_height} mask holds {len(pixels)} bytes, not one per pixel")
            packed[stamp] = (mask_width, mask_height, np.packbits(pixels != 0))
    return packed


class CatalogMasks:
    """The masks a catalog manifest names, read from the catalog one stream at a time and held bit-packed."""

    def __init__(self, manifest: FrameSourceManifest) -> None:
        """Bind to a manifest; nothing is read until a mask is asked for.

        Args:
            manifest: A frame-source manifest whose streams name a `mask_entity`.
        """
        self.manifest: FrameSourceManifest = manifest
        self.location: dict[str, tuple[int, int]] = {
            name: (stream_index, t_ns)
            for stream_index, stream in enumerate(manifest.streams)
            if stream.mask_entity is not None
            for name, t_ns in zip(stream.names, stream.t_ns, strict=True)
        }
        self.packed: dict[int, dict[int, tuple[int, int, UInt8[ndarray, " n"]]]] = {}
        """Per stream index: sample t_ns -> (mask width, mask height, `np.packbits` of the flat mask)."""
        self.scales: dict[int, tuple[float, float]] = {}
        """Per stream index: the layer's (x, y) scale from mask pixels to image pixels."""

    def _load(self, stream_index: int) -> dict[int, tuple[int, int, UInt8[ndarray, " n"]]]:
        if stream_index not in self.packed:
            # Imported here: only a masked catalog input needs the client.
            from rerun.catalog import CatalogClient
            from simplecv.rrd_query_utils import first_valid_value

            entity: str | None = self.manifest.streams[stream_index].mask_entity
            assert entity is not None
            dataset = CatalogClient(self.manifest.catalog_url).get_dataset(name=self.manifest.dataset)
            layer = dataset.filter_segments(self.manifest.segment).filter_contents([f"/{entity}"])
            statics: pa.Table = layer.reader(index=None).to_arrow_table()
            scale_column: str = f"/{entity}:Transform3D:scale"
            logged: list[float] | None = first_valid_value(statics[scale_column], allow_none=True) if scale_column in statics.column_names else None
            self.scales[stream_index] = (1.0, 1.0) if logged is None else (float(logged[0]), float(logged[1]))
            table: pa.Table = layer.reader(index=self.manifest.timeline).to_arrow_table()
            if f"/{entity}:SegmentationImage:buffer" not in table.column_names:
                raise ValueError(f"{self.manifest.segment}: no SegmentationImage at /{entity}")
            packed: dict[int, tuple[int, int, UInt8[ndarray, " n"]]] = pack_mask_table(table, self.manifest.timeline, entity)
            self.packed[stream_index] = packed
            print(f"[colsfm.masks] /{entity}: {len(self.packed[stream_index])} masks")
        return self.packed[stream_index]

    def mask(self, name: str, width: int, height: int) -> KeypointMask | None:
        """The image's mask, or None when its stream names no mask entity.

        Args:
            name: Image name from the manifest.
            width: The image's width.
            height: The image's height.

        Returns:
            `[height, width]` booleans, True = drop: the stored mask read through its scale.

        Raises:
            ValueError: When the stream is masked but has no mask at this image's stamp (a missing mask is an input
                error, not an unmasked image), or when the mask at its scale does not cover the image to within half a
                pixel.
        """
        if name not in self.location:
            return None
        stream_index, t_ns = self.location[name]
        packed: Mapping[int, tuple[int, int, UInt8[ndarray, " n"]]] = self._load(stream_index)
        if t_ns not in packed:
            raise ValueError(f"{name}: no mask at t_ns {t_ns} in /{self.manifest.streams[stream_index].mask_entity}")
        mask_width, mask_height, bits = packed[t_ns]
        scale_x, scale_y = self.scales[stream_index]
        if abs(mask_width * scale_x - width) >= 0.5 or abs(mask_height * scale_y - height) >= 0.5:
            raise ValueError(
                f"{name}: the {mask_width}x{mask_height} mask at scale {scale_x:g}x{scale_y:g} does not cover the {width}x{height} image"
            )
        native: Bool[ndarray, "mask_height mask_width"] = np.unpackbits(bits, count=mask_width * mask_height).astype(bool).reshape(mask_height, mask_width)
        if (mask_width, mask_height) == (width, height):
            return native
        rows: ndarray = (2 * np.arange(height) + 1) * mask_height // (2 * height)
        columns: ndarray = (2 * np.arange(width) + 1) * mask_width // (2 * width)
        return native[np.ix_(rows, columns)]


class DictMasks:
    """Masks held in memory by name (tests, and callers that computed their masks already)."""

    def __init__(self, masks: Mapping[str, KeypointMask]) -> None:
        """Hold the masks.

        Args:
            masks: Image name -> mask; names not present are unmasked.
        """
        self.masks: Mapping[str, KeypointMask] = masks

    def mask(self, name: str, width: int, height: int) -> KeypointMask | None:
        """The image's mask, or None.

        Args:
            name: Image name.
            width: The image's width.
            height: The image's height.

        Returns:
            The mask.

        Raises:
            ValueError: When the held mask is not `[height, width]`.
        """
        mask: KeypointMask | None = self.masks.get(name)
        if mask is not None and mask.shape != (height, width):
            raise ValueError(f"{name}: mask is {mask.shape[1]}x{mask.shape[0]}, the image {width}x{height}")
        return mask


def open_mask_source(manifest: FrameSourceManifest | None) -> MaskSource | None:
    """The mask source a catalog input names, or None when no stream is masked.

    Args:
        manifest: The input's frame-source manifest, None for a folder input.

    Returns:
        A `CatalogMasks`, or None.
    """
    if manifest is None or all(stream.mask_entity is None for stream in manifest.streams):
        return None
    return CatalogMasks(manifest)
