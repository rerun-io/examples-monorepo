"""The COLMAP database that carries cuSFM's frames into pycolmap.

`create_database` writes the four tables COLMAP's own feature extractor and
matcher expect to find already populated — cameras, rig, frames, images — using
cuSFM's identifiers verbatim (`camera_params_id` -> `camera_id`,
`KeyframeMetaData.id` -> `image_id`, `synced_sample_id` -> `frame_id`), the same
mapping `colsfm.reconstruction` uses.

Three things about COLMAP 4.2 shape this module:

* **Everything is `write_*`, not `add_*`.** pycolmap 0.x's `Database.add_camera`
  and friends do not exist (pycolmap-capabilities.md §4).
* **`Camera.has_prior_focal_length` defaults to False**, and that silently
  selects the uncalibrated (fundamental-matrix) path in two-view geometry: a
  PINHOLE pair comes back `UNCALIBRATED`, an `OPENCV_FISHEYE` pair comes back
  `DEGENERATE` with zero inliers (§5). `colsfm.cameras.colmap_camera` sets it
  True on every camera it builds, which is every camera that reaches this module.
* **`extract_features` reuses a pre-existing image row** rather than inserting a
  second one: COLMAP's `ImageReader` looks the name up first, keeps the
  `camera_id`, `image_id` and `frame_id` already stored, and leaves the rig and
  frame tables untouched. That is what lets the pipeline own the identifiers and
  still use COLMAP's own ALIKED extractor. Measured on Galileo; the log line
  `Focal Length: 852.92px (Prior)` is the visible confirmation.

Keypoints written by COLMAP are Nx6 (xy plus a 2x2 affine block), so
`read_keypoints` slices the two columns the rest of the pipeline wants.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Final, TypeAlias

import numpy as np
import pycolmap
from jaxtyping import Float, Float32, Int64
from numpy import ndarray

from colsfm.cameras import colmap_cameras
from colsfm.frames_meta import FramesMeta, KeyframeMeta, RigFrame
from colsfm.reconstruction import RIG_ID, build_rig, camera_sensor_id

ImagePair: TypeAlias = tuple[int, int]
"""An undirected image pair as `(min(image_id), max(image_id))`."""

KeypointsXY: TypeAlias = Float32[ndarray, "num_keypoints 2"]
"""Keypoint pixel coordinates in the original image, COLMAP's float32 storage type."""

Keypoints: TypeAlias = Float[ndarray, "num_keypoints 2"]
"""The same coordinates at whichever float width the caller asked `read_keypoints_batch` for.

COLMAP stores them as float32 and `colsfm.matching` works in that width, but pycolmap's
generalized absolute-pose estimators take float64, so the batch reader is parameterised on
the dtype rather than forcing every caller through a cast it has to remember."""

Descriptors: TypeAlias = Float32[ndarray, "num_features descriptor_dim"]
"""One image's feature descriptors, as `FeatureDescriptors.to_float()` reinterprets them."""

KeypointDtype: TypeAlias = type[np.float32 | np.float64]
"""The two float widths `read_keypoints_batch` will return keypoint positions at."""

KEYPOINT_XY_COLUMNS: Final[int] = 2
"""COLMAP's keypoint rows are Nx2, Nx4 or Nx6; only the leading xy pair is a position."""

MatchIndices: TypeAlias = Int64[ndarray, "num_matches 2"]
"""One row per match: the keypoint index in the pair's first image, then in its second."""


@contextlib.contextmanager
def database_transaction(database: pycolmap.Database) -> Iterator[None]:
    """Commit everything the block writes as one transaction.

    SQLite wraps each unbatched statement in its own transaction, so a batch of
    keypoint and descriptor writes used to reach the file one column at a time: an
    interrupted extraction could leave an image with keypoints and no descriptors,
    which every reader downstream takes as a contradiction rather than as damage.
    Inside this block they land together.

    `pycolmap.DatabaseTransaction` is an RAII object with no `close()`: COLMAP
    commits it in its destructor, so the block ends by dropping the last reference
    to it. That means it commits on the way out of an exception too -- it makes the
    write coherent, it is not a rollback.

    Args:
        database: An open COLMAP database.

    Yields:
        Nothing; the block runs inside the transaction.
    """
    transaction: pycolmap.DatabaseTransaction = pycolmap.DatabaseTransaction(database)
    try:
        yield
    finally:
        del transaction


def create_database(database_path: Path, frames_meta: FramesMeta, *, overwrite: bool = False) -> None:
    """Write cameras, the rig, the frames and the images of a collection.

    Nothing else is written: keypoints, descriptors, matches and two-view
    geometries are filled in later by `colsfm.features` and `colsfm.matching`.

    Args:
        database_path: Destination `.db` file; parent directories are created.
        frames_meta: The collection to import, already filtered to the keyframes
            that should be reconstructed (see `colsfm.keyframe_selection`).
        overwrite: Delete an existing database first instead of failing.

    Raises:
        FileExistsError: When the file exists and `overwrite` is False.
    """
    if database_path.exists():
        if not overwrite:
            raise FileExistsError(f"Database {database_path} already exists; pass overwrite=True to replace it")
        database_path.unlink()
    database_path.parent.mkdir(parents=True, exist_ok=True)

    cameras: dict[int, pycolmap.Camera] = colmap_cameras(frames_meta)
    keyframe_by_id: dict[int, KeyframeMeta] = frames_meta.keyframe_by_id()
    rig_frames: tuple[RigFrame, ...] = frames_meta.rig_frames()

    with pycolmap.Database.open(database_path) as database:
        for camera in cameras.values():
            database.write_camera(camera, True)
        database.write_rig(build_rig(frames_meta), True)
        for rig_frame in rig_frames:
            database.write_frame(_colmap_frame(rig_frame, keyframe_by_id), True)
            for keyframe_id in rig_frame.keyframe_ids:
                database.write_image(_colmap_image(keyframe_by_id[keyframe_id]), True)


def _colmap_frame(rig_frame: RigFrame, keyframe_by_id: Mapping[int, KeyframeMeta]) -> pycolmap.Frame:
    """Build the COLMAP frame for one rig instant.

    Args:
        rig_frame: The rig instant, from `FramesMeta.rig_frames`.
        keyframe_by_id: Keyframes of the collection, keyed by keyframe id.

    Returns:
        A frame carrying one `data_t` per member image. The pose stays unset —
        database frames carry the grouping only (pycolmap-capabilities.md §4).
    """
    frame: pycolmap.Frame = pycolmap.Frame()
    frame.frame_id = rig_frame.synced_sample_id
    frame.rig_id = RIG_ID
    for keyframe_id in rig_frame.keyframe_ids:
        sensor: pycolmap.sensor_t = camera_sensor_id(keyframe_by_id[keyframe_id].camera_params_id)
        frame.add_data_id(pycolmap.data_t(sensor, keyframe_id))
    return frame


def _colmap_image(keyframe: KeyframeMeta) -> pycolmap.Image:
    """Build the COLMAP image row for one keyframe.

    Args:
        keyframe: The keyframe to import.

    Returns:
        An image whose `image_id`, `camera_id` and `frame_id` are cuSFM's own
        ids and whose `name` is `image_name` relative to the raw image root.
    """
    image: pycolmap.Image = pycolmap.Image(name=keyframe.image_name, camera_id=keyframe.camera_params_id)
    image.image_id = keyframe.keyframe_id
    image.frame_id = keyframe.synced_sample_id
    return image


def image_ids_by_name(database_path: Path) -> dict[str, int]:
    """Index the database's images by name.

    COLMAP's pair lists and `verify_matches` address images by name, so this is
    what turns keyframe-id pairs into something pycolmap will accept.

    Args:
        database_path: An existing database.

    Returns:
        `image_id` keyed by `name`.
    """
    with pycolmap.Database.open(database_path) as database:
        return {image.name: image.image_id for image in database.read_all_images()}


def read_keypoints(database_path: Path, image_id: int) -> KeypointsXY:
    """Read one image's keypoints as pixel coordinates.

    Args:
        database_path: An existing database.
        image_id: The image to read.

    Returns:
        Float32 `[num_keypoints, 2]` xy pixel coordinates. COLMAP's ALIKED
        extractor stores Nx6 rows; the trailing affine block is dropped.
    """
    with pycolmap.Database.open(database_path) as database:
        stored: Float32[ndarray, "num_keypoints num_columns"] = database.read_keypoints(image_id)
    return np.ascontiguousarray(stored[:, :KEYPOINT_XY_COLUMNS])


def read_keypoints_batch(
    database_path: Path, image_ids: Iterable[int], dtype: KeypointDtype = np.float32
) -> dict[int, Keypoints]:
    """Read many images' keypoint positions through one database handle.

    The same "leading xy columns" rule as `read_keypoints`, spelled once: COLMAP's
    `write_keypoints` stores whatever column count it is handed (Nx2, Nx4 and Nx6 are all
    legal, `docs/spec/pycolmap-capabilities.md` §4), so anything past the first two is an
    affine block, not a position.

    Args:
        database_path: An existing database.
        image_ids: Images to read, in any order.
        dtype: Float width to return. `float32` is COLMAP's storage width and what the
            matcher works in; `float64` is what pycolmap's generalized absolute-pose
            estimators take, so `colsfm.loop_pose` asks for it.

    Returns:
        Keypoint pixels per image id; an image with no keypoint row is left out.

    Raises:
        FileNotFoundError: When the database does not exist.
    """
    if not database_path.is_file():
        raise FileNotFoundError(f"No COLMAP database at {database_path}")
    keypoints: dict[int, Keypoints] = {}
    with pycolmap.Database.open(database_path) as database:
        for image_id in image_ids:
            if not database.exists_keypoints(image_id):
                continue
            stored: Float32[ndarray, "num_keypoints num_columns"] = database.read_keypoints(image_id)
            keypoints[image_id] = np.ascontiguousarray(stored[:, :KEYPOINT_XY_COLUMNS], dtype=dtype)
    return keypoints


def read_descriptors_from_database(database_path: Path, image_ids: Iterable[int] | None = None) -> dict[int, Descriptors]:
    """Read float32 feature descriptors out of a COLMAP database.

    COLMAP 4.2 stores descriptors as an opaque uint8 blob, so an Nx128 float32 ALIKED
    block comes back as Nx512 uint8 and must be reinterpreted with
    `FeatureDescriptors.to_float()` — a reinterpret, not a cast
    (`docs/spec/pycolmap-capabilities.md` §4).

    Args:
        database_path: An existing database.
        image_ids: Images to read, or None for every image in the database.

    Returns:
        Descriptors per image id; an image with no descriptor row is left out.

    Raises:
        FileNotFoundError: When the database does not exist.
    """
    if not database_path.is_file():
        raise FileNotFoundError(f"No COLMAP database at {database_path}")
    descriptors_by_image: dict[int, Descriptors] = {}
    with pycolmap.Database.open(database_path) as database:
        wanted: list[int] = (
            [int(image.image_id) for image in database.read_all_images()]
            if image_ids is None
            else [int(image_id) for image_id in image_ids]
        )
        for image_id in wanted:
            if not database.exists_descriptors(image_id):
                continue
            stored: pycolmap.FeatureDescriptors = database.read_descriptors(image_id)
            descriptors_by_image[image_id] = np.ascontiguousarray(stored.to_float().data, dtype=np.float32)
    return descriptors_by_image


def keypoint_counts(database_path: Path, image_ids: Iterable[int] | None = None) -> dict[int, int]:
    """Count the keypoints stored per image.

    Args:
        database_path: An existing database.
        image_ids: Images to count; every image in the database by default.

    Returns:
        Keypoint count keyed by `image_id`; 0 for an image with no keypoint row.

    Note:
        `num_keypoints_for_image` reads the stored row count out of SQLite rather
        than materialising the blob, which matters on RoboCap where the keypoints
        are 222 MB and this is only ever printed.
    """
    with pycolmap.Database.open(database_path) as database:
        wanted: list[int] = (
            [image.image_id for image in database.read_all_images()] if image_ids is None else list(image_ids)
        )
        return {image_id: int(database.num_keypoints_for_image(image_id)) for image_id in wanted}


def read_two_view_geometry(database_path: Path, image_id1: int, image_id2: int) -> pycolmap.TwoViewGeometry:
    """Read the verified two-view geometry of one pair.

    Args:
        database_path: An existing database.
        image_id1: First image of the pair.
        image_id2: Second image of the pair.

    Returns:
        The stored geometry; an empty one when the pair was never verified.
    """
    with pycolmap.Database.open(database_path) as database:
        return database.read_two_view_geometry(image_id1, image_id2)


def _counts_by_pair(pair_ids: Sequence[int], counts: Sequence[int]) -> dict[ImagePair, int]:
    """Turn one of pycolmap's bulk `(pair_ids, counts)` readings into a pair-keyed map.

    COLMAP stores a pair under a single packed 64-bit id; the bulk readers return
    every stored row, so a pair that has no row is simply absent and the caller
    supplies the 0.

    Args:
        pair_ids: Packed pair ids, as the bulk reader returned them.
        counts: The matching counts, same length and order.

    Returns:
        Count keyed by `(min(image_id), max(image_id))`.
    """
    return {pycolmap.pair_id_to_image_pair(pair_id): int(count) for pair_id, count in zip(pair_ids, counts, strict=True)}


def pairs_with_matches(database_path: Path, pairs: Sequence[ImagePair]) -> set[ImagePair]:
    """Which of these pairs the database has already matched, whatever the count.

    Presence, not count. A pair the matcher ran and found nothing for is *finished
    work*: matching it again costs a matcher call and produces the same nothing.
    The loop stage used to ask `raw_match_counts` and treat `== 0` as "not matched",
    so a completed zero-match pair was indistinguishable from missing work.

    `exists_matches` rather than `read_num_matches`: COLMAP's bulk count query is
    `WHERE rows > 0`, so it cannot see the very rows this function exists to find.
    One open handle and one indexed lookup per pair; 14 443 pairs cost milliseconds.

    Args:
        database_path: An existing database.
        pairs: Image pairs to look up, in any order.

    Returns:
        The subset of `pairs` that has a matches row, zero-match rows included.
    """
    with pycolmap.Database.open(database_path) as database:
        return {pair for pair in pairs if database.exists_matches(pair[0], pair[1])}


def pair_inlier_counts(database_path: Path, pairs: Sequence[ImagePair]) -> dict[ImagePair, int]:
    """Count the verified inlier matches of many pairs in one pass.

    `read_two_view_geometry_num_inliers` reads every stored row's count without
    materialising a single match blob, so the cost no longer grows with the
    matches per pair.

    Args:
        database_path: An existing database.
        pairs: Image pairs to look up, in any order.

    Returns:
        Inlier count keyed by the pair as given; 0 for a pair with no geometry.
    """
    with pycolmap.Database.open(database_path) as database:
        counts: dict[ImagePair, int] = _counts_by_pair(*database.read_two_view_geometry_num_inliers())
    return {pair: counts.get(pair, 0) for pair in pairs}


def delete_two_view_geometries(database_path: Path, pairs: Sequence[ImagePair]) -> None:
    """Drop the two-view geometry rows of the given pairs.

    `pycolmap.verify_matches` skips any pair that already has a row, and COLMAP's
    matcher writes an `UNDEFINED` placeholder row whenever
    `skip_geometric_verification` is set — so the placeholders have to go before
    verification will do anything.

    Args:
        database_path: An existing database.
        pairs: Image pairs whose geometries should be removed.
    """
    # One transaction, as `create_database` and the feature writers use: SQLite wraps
    # each unbatched delete in its own otherwise, which is one commit per pair.
    with pycolmap.Database.open(database_path) as database, database_transaction(database):
        for image_id1, image_id2 in pairs:
            database.delete_two_view_geometry(image_id1, image_id2)


def raw_match_counts(database_path: Path, pairs: Sequence[ImagePair]) -> dict[ImagePair, int]:
    """Count the unverified matches of many pairs in one pass.

    `read_num_matches` reads every stored row's count in one query rather than
    fetching and measuring each pair's match blob.

    Args:
        database_path: An existing database.
        pairs: Image pairs to look up, in any order.

    Returns:
        Raw match count keyed by the pair as given; 0 for an unmatched pair.
    """
    with pycolmap.Database.open(database_path) as database:
        counts: dict[ImagePair, int] = _counts_by_pair(*database.read_num_matches())
    return {pair: counts.get(pair, 0) for pair in pairs}
