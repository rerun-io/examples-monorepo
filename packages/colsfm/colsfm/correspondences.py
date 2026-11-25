"""Reading the database's verified two-view geometries into a triangulator.

One owner for the step between "the matcher wrote geometries" and "the mapper can
triangulate": the `CorrespondenceGraph` COLMAP's `IncrementalTriangulator` walks,
the `ObservationManager` the filters count through, and the check that the
database and the reconstruction agree on which images exist at all.

Separate from `colsfm.mapping` because it is the mapper's *input*, and it is read
by the extrinsic refinement and by the audit tools on their own terms -- none of
which want the round loop that follows it.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
from jaxtyping import UInt32
from numpy import ndarray


@dataclass(slots=True)
class Correspondences:
    """The correspondence graph a triangulator needs, kept alive alongside its owner.

    `IncrementalTriangulator` holds raw references to the graph and the
    reconstruction, so the `DatabaseCache` that owns the graph has to outlive
    both. Keeping all three in one object is what guarantees that.
    """

    database_cache: pycolmap.DatabaseCache
    """Owns the correspondence graph; must outlive every triangulator built on it."""
    graph: pycolmap.CorrespondenceGraph
    """Keypoint correspondences from the database's verified two-view geometries."""
    observation_manager: pycolmap.ObservationManager
    """Bookkeeping for observations and the filters, bound to the reconstruction."""
    num_images: int
    """Images the database contributed correspondences for."""
    num_image_pairs: int
    """Verified image pairs that survived `min_num_matches`."""


def load_correspondences(
    reconstruction: pycolmap.Reconstruction,
    database_path: Path,
    min_num_matches: int = 0,
    *,
    inlier_overrides: Mapping[tuple[int, int], UInt32[ndarray, "n_inliers 2"]] | None = None,
) -> Correspondences:
    """Load keypoints and verified two-view geometries into the reconstruction.

    The database's keypoints become each image's `points2D` and its verified
    two-view geometries become the correspondence graph the triangulator walks.
    Nothing is re-estimated: the images already carry poses, and COLMAP's
    `DatabaseCache` is used only for the keypoints and the graph.

    Database image ids must equal the reconstruction's, because the graph is
    keyed by them. Both come from cuSFM's keyframe ids, so they agree by
    construction; a mismatch is a bug in whoever wrote the database and is
    reported rather than silently mis-triangulated.

    An image that already holds the database's keypoints (a model a previous
    `run_mapping` left behind, as `--no-regularised-extrinsics` re-maps it) keeps
    them and the tracks on them: pycolmap only sets `points2D` on an empty image.

    Args:
        reconstruction: A reconstruction with registered, pose-carrying frames.
        database_path: COLMAP database written by the feature and matching stages.
        min_num_matches: Drop image pairs with fewer inlier matches; cuSFM's
            `min_num_matches_per_pair`.
        inlier_overrides: Per pair `(min id, max id)`, the subset of its stored inlier matches the graph carries instead
            (`MappingOptions.inlier_overrides`); a pair left below `min_num_matches` leaves the graph.

    Returns:
        The graph, the observation manager and the cache that owns them.

    Raises:
        FileNotFoundError: When the database does not exist.
        ValueError: When a database image name maps to a different id than the
            reconstruction's, or when the database holds none of its images.
    """
    if not database_path.is_file():
        raise FileNotFoundError(f"No COLMAP database at {database_path}")
    cache_options: pycolmap.DatabaseCacheOptions = pycolmap.DatabaseCacheOptions()
    cache_options.min_num_matches = min_num_matches
    cache_options.ignore_watermarks = False
    cache_options.image_names = {image.name for image in reconstruction.images.values()}
    with pycolmap.Database.open(database_path) as database:
        database_cache: pycolmap.DatabaseCache = pycolmap.DatabaseCache.create(database, cache_options)
        # The rows `DatabaseCache` itself read (`ReadTwoViewGeometries`, in row order), for the rebuild below; with
        # nothing overridden the cache's own graph is used, so neither the read nor the rebuild runs.
        geometry_rows: tuple[list[int], list[pycolmap.TwoViewGeometry]] | None = database.read_two_view_geometries() if inlier_overrides else None

    image_id_by_name: dict[str, int] = {image.name: image_id for image_id, image in reconstruction.images.items()}
    mismatched: list[str] = []
    num_loaded: int = 0
    for image_id, cached_image in database_cache.images.items():
        expected_image_id: int | None = image_id_by_name.get(cached_image.name)
        if expected_image_id is None:
            continue
        if expected_image_id != image_id:
            mismatched.append(f"{cached_image.name}: database id {image_id}, reconstruction id {expected_image_id}")
            continue
        image: pycolmap.Image = reconstruction.image(image_id)
        if image.num_points2D() == 0:
            image.points2D = cached_image.points2D
        elif image.num_points2D() != cached_image.num_points2D():
            mismatched.append(f"{cached_image.name}: {image.num_points2D()} keypoints in the model, {cached_image.num_points2D()} in the database")
            continue
        num_loaded += 1
    if mismatched:
        raise ValueError(f"Database and reconstruction disagree on images: {mismatched[:5]}")
    if num_loaded == 0:
        raise ValueError(f"{database_path} holds none of the reconstruction's {reconstruction.num_images()} images")

    graph: pycolmap.CorrespondenceGraph = (
        database_cache.correspondence_graph
        if geometry_rows is None
        else graph_with_inlier_overrides(database_cache, geometry_rows, inlier_overrides=inlier_overrides, min_num_matches=min_num_matches)
    )
    observation_manager: pycolmap.ObservationManager = pycolmap.ObservationManager(reconstruction, graph)
    return Correspondences(
        database_cache=database_cache,
        graph=graph,
        observation_manager=observation_manager,
        num_images=num_loaded,
        num_image_pairs=graph.num_image_pairs(),
    )


def graph_with_inlier_overrides(
    database_cache: pycolmap.DatabaseCache,
    geometry_rows: tuple[Sequence[int], Sequence[pycolmap.TwoViewGeometry]],
    *,
    inlier_overrides: Mapping[tuple[int, int], UInt32[ndarray, "n_inliers 2"]] | None = None,
    min_num_matches: int = 0,
) -> pycolmap.CorrespondenceGraph:
    """The cache's correspondence graph with some pairs' inliers replaced, rebuilt from the database's own rows.

    `DatabaseCache` has no match filter, and replacing a pair's inliers here is what lets a stage trim the pairs it
    matched without rewriting database rows (COLMAP runs a full `VACUUM` when it closes a connection that deleted
    anything). `load_correspondences` calls it only when a pair is overridden; without one it returns the cache's graph,
    correspondence for correspondence (`test_correspondence_exclusion`).

    It replays what `DatabaseCache::Load` did: the same rows (`ReadTwoViewGeometries`), in their order, each added with
    its stored geometry, so every correspondence list comes out in the same order, many-to-one matches included. Which
    rows the cache accepted (configuration, `min_num_matches`, the watermark rule, the selected images) stays COLMAP's
    decision: a row is replayed only when the cache's graph holds its pair.

    Args:
        database_cache: The loaded cache.
        geometry_rows: `Database.read_two_view_geometries()` of the database the cache was loaded from: pair ids and
            geometries in row order.
        inlier_overrides: Per pair `(min id, max id)`, the inlier matches (columns in that id order) to add instead of the
            stored ones; a match-level filter without database writes.
        min_num_matches: The cache's own floor, applied again to an overridden pair.

    Returns:
        A finalized graph over the cache's images.
    """
    cached: pycolmap.CorrespondenceGraph = database_cache.correspondence_graph
    graph: pycolmap.CorrespondenceGraph = pycolmap.CorrespondenceGraph()
    for image_id, image in database_cache.images.items():
        if cached.exists_image(image_id):
            graph.add_image(image_id, len(image.points2D))
    in_cache: set[int] = set(cached.image_pairs())
    for pair_id, geometry in zip(*geometry_rows, strict=True):
        if int(pair_id) not in in_cache:
            continue
        image_id1, image_id2 = pycolmap.pair_id_to_image_pair(int(pair_id))
        pair: tuple[int, int] = (min(image_id1, image_id2), max(image_id1, image_id2))
        override: UInt32[ndarray, "n_inliers 2"] | None = None if inlier_overrides is None else inlier_overrides.get(pair)
        if override is not None:
            if len(override) == 0 or len(override) < min_num_matches:
                continue
            geometry = copy.copy(geometry)
            # The override is written low id first; a row stored the other way round keeps its own column order.
            columns: UInt32[ndarray, "n_inliers 2"] = override if image_id1 < image_id2 else override[:, ::-1]
            geometry.inlier_matches = np.ascontiguousarray(columns, dtype=np.uint32)
        graph.add_two_view_geometry(image_id1, image_id2, geometry)
    graph.finalize()
    return graph


def registered_image_ids(reconstruction: pycolmap.Reconstruction) -> list[int]:
    """Image ids belonging to a registered frame, ascending.

    Args:
        reconstruction: The model.

    Returns:
        Sorted image ids.
    """
    image_ids: list[int] = []
    for frame_id in reconstruction.reg_frame_ids():
        for data_id in reconstruction.frame(frame_id).data_ids:
            if data_id.sensor_id.type == pycolmap.SensorType.CAMERA:
                image_ids.append(data_id.id)
    return sorted(image_ids)
