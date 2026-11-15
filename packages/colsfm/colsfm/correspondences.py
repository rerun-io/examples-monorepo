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

from dataclasses import dataclass
from pathlib import Path

import pycolmap


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

    Args:
        reconstruction: A reconstruction with registered, pose-carrying frames.
        database_path: COLMAP database written by the feature and matching stages.
        min_num_matches: Drop image pairs with fewer inlier matches; cuSFM's
            `min_num_matches_per_pair`.

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
        reconstruction.image(image_id).points2D = cached_image.points2D
        num_loaded += 1
    if mismatched:
        raise ValueError(f"Database and reconstruction disagree on image ids: {mismatched[:5]}")
    if num_loaded == 0:
        raise ValueError(f"{database_path} holds none of the reconstruction's {reconstruction.num_images()} images")

    graph: pycolmap.CorrespondenceGraph = database_cache.correspondence_graph
    observation_manager: pycolmap.ObservationManager = pycolmap.ObservationManager(reconstruction, graph)
    return Correspondences(
        database_cache=database_cache,
        graph=graph,
        observation_manager=observation_manager,
        num_images=num_loaded,
        num_image_pairs=graph.num_image_pairs(),
    )


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


