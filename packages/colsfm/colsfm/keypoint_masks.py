"""Per-image keypoint masks: pixels where no keypoint may be stored (people, the operator, the rig itself).

A mask is a boolean image at the keyframe's own resolution, True where keypoints are dropped. Extraction applies it
after the score gate and the top-k (`colsfm.features_native.store_feature_batch`), before anything reaches the
database, so matching, loop closure and mapping never see a masked keypoint. The kept keypoints keep their order and
their descriptors.

Masks come from the catalog: a catalog input names, per camera stream, a `SegmentationImage` entity on the same
stamps as the video (`colsfm.frame_manifest.StreamKeyframes.mask_entity`), nonzero = drop. `CatalogMasks` reads each
stream's masks once, on first use, and keeps them bit-packed (one bit per pixel).
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
        self.packed: dict[int, dict[int, tuple[int, UInt8[ndarray, " n"]]]] = {}
        """Per stream index: sample t_ns -> (pixel count, `np.packbits` of the flat mask)."""

    def _load(self, stream_index: int) -> dict[int, tuple[int, UInt8[ndarray, " n"]]]:
        if stream_index not in self.packed:
            # Imported here: only a masked catalog input needs the client.
            from rerun.catalog import CatalogClient

            entity: str | None = self.manifest.streams[stream_index].mask_entity
            assert entity is not None
            dataset = CatalogClient(self.manifest.catalog_url).get_dataset(name=self.manifest.dataset)
            table: pa.Table = (
                dataset.filter_segments(self.manifest.segment).filter_contents([f"/{entity}"]).reader(index=self.manifest.timeline).to_arrow_table()
            )
            column: str = f"/{entity}:SegmentationImage:buffer"
            if column not in table.column_names:
                raise ValueError(f"{self.manifest.segment}: no SegmentationImage at /{entity}")
            table = table.filter(table[column].is_valid())
            t_ns: ndarray = table[self.manifest.timeline].combine_chunks().to_numpy(zero_copy_only=False).view(np.int64)
            buffers: pa.Array = table[column].combine_chunks().flatten()  # one instance per row -> list<uint8> per row
            packed: dict[int, tuple[int, UInt8[ndarray, " n"]]] = {}
            for row, stamp in enumerate(t_ns):
                pixels: UInt8[ndarray, " n"] = buffers[row].values.to_numpy(zero_copy_only=False)
                packed[int(stamp)] = (len(pixels), np.packbits(pixels != 0))
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
            `[height, width]` booleans, True = drop.

        Raises:
            ValueError: When the stream is masked but has no mask at this image's stamp: a missing mask is an input
                error, not an unmasked image.
        """
        if name not in self.location:
            return None
        stream_index, t_ns = self.location[name]
        packed: Mapping[int, tuple[int, UInt8[ndarray, " n"]]] = self._load(stream_index)
        if t_ns not in packed:
            raise ValueError(f"{name}: no mask at t_ns {t_ns} in /{self.manifest.streams[stream_index].mask_entity}")
        num_pixels, bits = packed[t_ns]
        if num_pixels != width * height:
            raise ValueError(f"{name}: the mask has {num_pixels} pixels, the image is {width}x{height}")
        return np.unpackbits(bits, count=num_pixels).astype(bool).reshape(height, width)


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
