"""The guards that run after a bundle adjustment, which cuSFM has no equivalent of.

A converged solve can still leave a point outside the world or inside a camera:
`filter_degenerate_points` removes what the depth and bounds tests reject, and
`filter_projection_failures` removes observations whose projection is not finite
or lands absurdly far off the sensor. Both docstrings carry the measurements that
justify their thresholds.

Their own module because they are a *policy about points*, applied between the
solve and the next round, and they are exercised directly by tests that build a
degenerate model on purpose rather than through a whole mapping run.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pycolmap
from jaxtyping import Bool, Float64
from numpy import ndarray

from colsfm.config import VisionMappingConfig


@dataclass(frozen=True, slots=True)
class _FilterCounts:
    """Observations removed by one round's filters, split by cause."""

    reprojection_and_angle: int = 0
    """Dropped by `filter_all_points3D` (reprojection, negative depth, triangulation angle)."""
    negative_depth: int = 0
    """Dropped by the explicit cheirality pass."""
    short_track: int = 0
    """Dropped for having fewer than `min_correspondences` observations."""
    degenerate: int = 0
    """Dropped by the pre-solve guard: the world-Z cap, a non-finite coordinate or a
    coordinate whose magnitude exceeds `depth_threshold`."""

    def total(self) -> int:
        """Total observations removed.

        Returns:
            The sum of every cause.
        """
        return self.reprojection_and_angle + self.negative_depth + self.short_track + self.degenerate


def filter_degenerate_points(
    observation_manager: pycolmap.ObservationManager,
    reconstruction: pycolmap.Reconstruction,
    depth_threshold: float,
) -> int:
    """Delete every point whose world position is non-finite or out of bounds.

    A point goes when any coordinate is NaN or infinite, or when the largest
    coordinate magnitude exceeds `depth_threshold`. That is deliberately stricter
    than cuSFM, whose only spatial filter is a one-sided cap on the world Z axis
    (keypoints_mapper_main.md §5.5); a bound on all three axes and both signs
    subsumes it, and cuSFM's asymmetry looks like an oversight rather than a
    design. On Galileo the difference is nothing — the guard removes 0 points in
    all five rounds.

    Two things make the guard necessary, and neither is caught by COLMAP's own
    filters:

    1. **NaN is invisible to every threshold test.** They all ask
       `error > threshold`, and any comparison against NaN is false, so a NaN
       point survives filtering and then poisons the whole normal equation.
    2. **A converged solve can still throw a point to infinity.** Measured on 200
       RoboCap rig frames: round 0's bundle adjustment reports CONVERGENCE and
       halves its cost while pushing 5 of 31 017 points out to 4941 m. Under the
       fisheye model those points project next to the projection singularity, so
       their reprojection error comes out at 4.5e153 px — which is where the
       round's reported mean of 1.4e149 px came from. The pre-solve filter alone
       cannot help: the model going *into* that solve was clean (max error
       22.4 px). The guard therefore runs after the solve as well as before it.

    Args:
        observation_manager: Bookkeeping bound to the reconstruction.
        reconstruction: The model to filter, edited in place.
        depth_threshold: `vision_mapping_config.depth_threshold`, in metres.

    Returns:
        Observations removed, i.e. the summed track length of the deleted points.
    """
    # One walk of the map, not an id list and then a lookup per id: the map is
    # 210 000 entries on RoboCap and the second pass bought nothing.
    points: list[tuple[int, pycolmap.Point3D]] = list(reconstruction.points3D.items())
    if not points:
        return 0
    point3D_ids: list[int] = [point3D_id for point3D_id, _ in points]
    points_xyz: Float64[ndarray, "num_points 3"] = np.array(
        [point.xyz for _, point in points], dtype=np.float64
    )
    # `np.abs(nan) > threshold` is False, so the finiteness term has to carry the NaNs.
    out_of_bounds: Bool[ndarray, "num_points"] = ~np.isfinite(points_xyz).all(axis=1) | (
        np.abs(points_xyz).max(axis=1) > depth_threshold
    )
    removed: int = 0
    for point3D_id in np.asarray(point3D_ids)[out_of_bounds].tolist():
        removed += reconstruction.point3D(point3D_id).track.length()
        observation_manager.delete_point3D(point3D_id)
    return removed


def projection_sanity_bound_px(reconstruction: pycolmap.Reconstruction) -> float:
    """The largest reprojection error that can still be a measurement.

    Args:
        reconstruction: The model, for its cameras' pixel dimensions.

    Returns:
        The longest image side over every camera, or 0.0 for a model without
        cameras. An observation further than a whole image from its point is not
        a bad measurement, it is a failed projection.
    """
    sides: list[float] = [float(max(camera.width, camera.height)) for camera in reconstruction.cameras.values()]
    return max(sides) if sides else 0.0


def filter_projection_failures(
    observation_manager: pycolmap.ObservationManager, reconstruction: pycolmap.Reconstruction
) -> int:
    """Drop observations a solve has made unprojectable, at a numeric-sanity bound.

    Not a quality filter — the round's own gate (25 down to 5 px) is far tighter
    and owns that job. This only catches projection *failures*, at
    `projection_sanity_bound_px`, and it exists because bundle adjustment can
    produce them out of a clean model:

    On RoboCap's 4528 fisheye images, 4 points of 209 905 end up within
    5-90 mm of a camera centre — one of them at **negative** depth. At that
    distance `OPENCV_FISHEYE` projects them thousands of image widths away and
    their reprojection error comes out at 4.5e153 px, which is where the
    `1e152 px` mean and the CHOLMOD "not positive definite" warnings in the first
    three rounds came from. The model going *into* each of those solves was clean
    (max 23.5 px), so no pre-solve filter can prevent it.

    Deleting them now rather than in the next round changes no result — the next
    gate is at most 25 px, so anything above a whole image width was already
    doomed — but it keeps the round's reported metric meaningful and keeps the
    next solve from linearising the same singularity.

    Args:
        observation_manager: Bookkeeping bound to the reconstruction.
        reconstruction: The model to filter, edited in place.

    Returns:
        Observations removed.
    """
    bound_px: float = projection_sanity_bound_px(reconstruction)
    if bound_px <= 0.0:
        return 0
    # `filter_all_points3D` covers reprojection error, cheirality and triangulation
    # angle over the whole map; a zero angle threshold switches the last of the three
    # off, leaving the two that describe a failed projection. `_filter_points` already
    # calls it this way -- naming every point in a set was the same request, built by
    # copying 210 000 ids out of the map first.
    return observation_manager.filter_all_points3D(bound_px, 0.0)


def _filter_points(
    observation_manager: pycolmap.ObservationManager,
    reconstruction: pycolmap.Reconstruction,
    mapping_config: VisionMappingConfig,
    max_pixel_error: float,
) -> _FilterCounts:
    """Apply cuSFM's point filters for one round, before its bundle adjustment (§5.4-§5.5).

    The reprojection gate decays per round; the triangulation angle is the max
    over observing pairs, matching cuSFM's `HasLargeEnoughTriangulateAngle`; the
    depth test is a **world Z** cap, one axis and one-sided, not a camera depth.
    `filter_degenerate_points` adds the non-finite and absurd-coordinate guard
    cuSFM has no equivalent of; see its docstring.

    Args:
        observation_manager: Bookkeeping bound to the reconstruction.
        reconstruction: The model to filter.
        mapping_config: The mapping configuration.
        max_pixel_error: The round's gate, in pixels.

    Returns:
        Observations removed, split by cause.
    """
    reprojection_and_angle: int = observation_manager.filter_all_points3D(
        max_pixel_error, mapping_config.min_triangulation_deg
    )
    negative_depth: int = observation_manager.filter_observations_with_negative_depth()
    short_track: int = observation_manager.filter_points3D_with_short_tracks(mapping_config.min_correspondences)
    degenerate: int = filter_degenerate_points(observation_manager, reconstruction, mapping_config.depth_threshold)
    return _FilterCounts(
        reprojection_and_angle=reprojection_and_angle,
        negative_depth=negative_depth,
        short_track=short_track,
        degenerate=degenerate,
    )


