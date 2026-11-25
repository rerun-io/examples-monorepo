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

Every input also gets a mask from its camera model, without a flag: `valid_angle_mask` drops the pixels whose ray the
model cannot unproject back to itself (`valid_angle`), and `with_valid_angle_masks` joins it to the catalog's masks.
Only a lens that images rays at or past 90 degrees, or a fisheye polynomial that folds inside the image, has such
pixels; every other camera keeps every keypoint, bit for bit.
"""

from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

import numpy as np
import pyarrow as pa
import pycolmap
from jaxtyping import Bool, Float32, Float64, UInt8
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


def keypoints_outside_mask(keypoints_xy: Float32[ndarray, "n 2"], mask: KeypointMask, pixel_centre: float = 0.0) -> Bool[ndarray, " n"]:
    """Which keypoints survive a mask: those whose pixel is not masked.

    The pixel is the coordinate less `pixel_centre`, rounded, clipped into the image: OpenCV keypoints (the `raco`
    backend) put pixel centres on integers (0.0), COLMAP keypoints (the `pycolmap` backend) at +0.5, so a content mask
    reads them at 0.5.

    Args:
        keypoints_xy: Pixel coordinates.
        mask: True = drop.
        pixel_centre: The coordinate of pixel 0's centre in the keypoints' convention.

    Returns:
        True for each keypoint to keep.
    """
    height, width = mask.shape
    columns: ndarray = np.clip(np.rint(keypoints_xy[:, 0] - pixel_centre).astype(np.int64), 0, width - 1)
    rows: ndarray = np.clip(np.rint(keypoints_xy[:, 1] - pixel_centre).astype(np.int64), 0, height - 1)
    return ~mask[rows, columns]


def pack_mask_table(table: pa.Table, timeline: str, entity: str) -> dict[int, tuple[int, int, UInt8[ndarray, " n"]]]:
    """Bit-pack every mask of one `SegmentationImage` entity, keyed by its stamp; rows without a mask (null or cleared) are skipped.

    Walks the table's record batches through the list offsets, so no mask byte is copied before `np.packbits`
    (a `filter` first copies every mask in the segment).

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


VALID_ANGLE_MARGIN_DEG: Final[float] = 1.0
"""How far short of 90 degrees an `OPENCV_FISHEYE` keypoint must stay off the optical axis.

COLMAP projects nothing at or behind z = 0 (`HasProjectableDepth`), and its fisheye unprojection
(`NormalFromFisheye`) returns a z = 1 point, so a pixel at theta > 90 degrees comes back folded into the front
hemisphere with no error: on TartanAir's 195 degree lens (k = 0) a 95 degree pixel unprojects to 58.9 degrees. At 90
degrees the z = 1 point is at infinity. One degree short keeps it finite (tan 89 = 57); on that lens, an 89 degree
keypoint cut halved the trajectory error against ground truth (34 to 17 mm) with the default bundle adjuster."""


def valid_angle(camera: pycolmap.Camera) -> float | None:
    """The widest angle off the optical axis whose pixel unprojects back to its own ray, or None when the model has no limit.

    `OPENCV_FISHEYE` (Kannala-Brandt, theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)) can image
    any angle, so it gets the smaller of 90 degrees less `VALID_ANGLE_MARGIN_DEG` and the polynomial's first turning
    point (past it two rays share a pixel; RoboCap's lens reaches it inside the image), found analytically.
    `PINHOLE` and `FULL_OPENCV` are perspective: every pixel holds a ray in front of the camera. `FULL_OPENCV`'s rational
    pole is `colsfm.point_filters.before_rational_pole`'s guard, applied to the map's points.

    Args:
        camera: The camera.

    Returns:
        The angle in radians, or None.
    """
    if camera.model != pycolmap.CameraModelId.OPENCV_FISHEYE:
        return None
    k1, k2, k3, k4 = (float(value) for value in camera.params[4:8])
    # d theta_d / d theta = 1 + 3 k1 x + 5 k2 x^2 + 7 k3 x^3 + 9 k4 x^4 with x = theta^2.
    turning: list[float] = [
        float(root.real) for root in np.roots([9 * k4, 7 * k3, 5 * k2, 3 * k1, 1.0]) if abs(root.imag) < 1e-12 and root.real > 0.0
    ]
    horizon: float = float(np.radians(90.0 - VALID_ANGLE_MARGIN_DEG))
    return min(horizon, float(np.sqrt(min(turning)))) if turning else horizon


def valid_angle_mask(camera: pycolmap.Camera) -> KeypointMask | None:
    """The pixels past the camera's `valid_angle`, or None when no pixel of its image is.

    The boundary is the forward projection of the limit, which is exact: `OPENCV_FISHEYE` distorts radially in
    normalised coordinates, so the limit's rays land on an ellipse with semi-axes fx theta_d and fy theta_d around the
    principal point. It is sampled at integer pixel coordinates, the nearest of which `keypoints_outside_mask` reads.

    Args:
        camera: The camera.

    Returns:
        `[height, width]` booleans, True = drop, or None.
    """
    limit: float | None = valid_angle(camera)
    if limit is None:
        return None
    width, height = int(camera.width), int(camera.height)
    limit_rays: Float64[ndarray, "2 3"] = np.asarray([[np.sin(limit), 0.0, np.cos(limit)], [0.0, np.sin(limit), np.cos(limit)]])
    edge: Float64[ndarray, "2 2"] = np.asarray(camera.img_from_cam(limit_rays), dtype=np.float64)
    centre_x, centre_y = float(camera.principal_point_x), float(camera.principal_point_y)
    columns: Float64[ndarray, " width"] = ((np.arange(width) - centre_x) / (edge[0, 0] - centre_x)) ** 2
    rows: Float64[ndarray, " height"] = ((np.arange(height) - centre_y) / (edge[1, 1] - centre_y)) ** 2
    outside: KeypointMask = rows[:, None] + columns[None, :] > 1.0
    return outside if outside.any() else None


class ValidAngleMasks:
    """Another mask source with each camera's `valid_angle_mask` joined in (a pixel either drops is dropped)."""

    def __init__(self, angle_masks: Mapping[str, KeypointMask], masks: MaskSource | None) -> None:
        """Hold both.

        Args:
            angle_masks: Image name -> its camera's `valid_angle_mask`; names not present have none.
            masks: The input's own masks, or None.
        """
        self.angle_masks: Mapping[str, KeypointMask] = angle_masks
        self.masks: MaskSource | None = masks

    def mask(self, name: str, width: int, height: int) -> KeypointMask | None:
        """The image's joined mask, or None when neither source masks it.

        Args:
            name: Image name.
            width: The image's width.
            height: The image's height.

        Returns:
            `[height, width]` booleans, True = drop.
        """
        own: KeypointMask | None = None if self.masks is None else self.masks.mask(name, width, height)
        angle: KeypointMask | None = self.angle_masks.get(name)
        if own is None:
            return angle
        if angle is None:
            return own
        return own | angle


def with_valid_angle_masks(database_path: Path, masks: MaskSource | None) -> MaskSource | None:
    """The input's masks with every camera's `valid_angle_mask` joined in; `masks` itself when no camera has one.

    Args:
        database_path: The run's database; its camera and image rows name each image's camera.
        masks: The input's own masks, or None.

    Returns:
        A `ValidAngleMasks`, or `masks` unchanged.
    """
    with pycolmap.Database.open(database_path) as database:
        by_camera: dict[int, KeypointMask | None] = {int(camera.camera_id): valid_angle_mask(camera) for camera in database.read_all_cameras()}
        angle_masks: dict[str, KeypointMask] = {
            image.name: mask for image in database.read_all_images() if (mask := by_camera[int(image.camera_id)]) is not None
        }
    if not angle_masks:
        return masks
    limited: list[str] = sorted({name.split("/")[0] for name in angle_masks})
    print(f"[colsfm.masks] valid-angle mask on {len(angle_masks)} images ({', '.join(limited)})")
    return ValidAngleMasks(angle_masks, masks)


def drop_masked_keypoints(database_path: Path, masks: MaskSource) -> None:
    """Remove the masked keypoints, and their descriptors, of every image already in the database.

    For an extractor that writes its own rows (the `pycolmap` backend); the `raco` backend applies masks before it
    stores anything (`colsfm.features_native.store_feature_batch`). Kept rows keep their order and every column.
    COLMAP keypoints put pixel centres at +0.5, so a content mask reads each keypoint in the pixel it falls in; a
    `ValidAngleMasks`' camera mask is sampled at integer COLMAP coordinates and reads it at the nearest one.

    Args:
        database_path: A database whose keypoints and descriptors are written.
        masks: The masks to apply.
    """
    angle_masks, content = (masks.angle_masks, masks.masks) if isinstance(masks, ValidAngleMasks) else ({}, masks)
    # pycolmap updates keypoints but cannot replace a descriptor row (its write is an INSERT), so both rows are
    # rewritten through sqlite once the pycolmap handle is closed; `cols` and the descriptor `type` stay as stored.
    rewrites: list[tuple[int, Float32[ndarray, "kept columns"], UInt8[ndarray, "kept bytes"]]] = []
    with pycolmap.Database.open(database_path) as database:
        cameras: dict[int, pycolmap.Camera] = {int(camera.camera_id): camera for camera in database.read_all_cameras()}
        for image in database.read_all_images():
            camera: pycolmap.Camera = cameras[int(image.camera_id)]
            content_mask: KeypointMask | None = None if content is None else content.mask(image.name, int(camera.width), int(camera.height))
            angle_mask: KeypointMask | None = angle_masks.get(image.name)
            if (content_mask is None and angle_mask is None) or not database.exists_keypoints(image.image_id):
                continue
            keypoints: Float32[ndarray, "n columns"] = database.read_keypoints(image.image_id)
            kept: Bool[ndarray, " n"] = np.ones(len(keypoints), dtype=bool)
            if content_mask is not None:
                kept &= keypoints_outside_mask(keypoints[:, :2], content_mask, pixel_centre=0.5)
            if angle_mask is not None:
                kept &= keypoints_outside_mask(keypoints[:, :2], angle_mask)
            if not kept.all():
                rewrites.append((int(image.image_id), keypoints[kept], database.read_descriptors(image.image_id).data[kept]))
    with contextlib.closing(sqlite3.connect(database_path)) as connection, connection:
        for image_id, keypoints, descriptors in rewrites:
            connection.execute("UPDATE keypoints SET rows = ?, data = ? WHERE image_id = ?", (len(keypoints), keypoints.tobytes(), image_id))
            connection.execute("UPDATE descriptors SET rows = ?, data = ? WHERE image_id = ?", (len(descriptors), descriptors.tobytes(), image_id))
