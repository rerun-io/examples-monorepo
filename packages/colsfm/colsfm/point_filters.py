"""The guards that run after a bundle adjustment, which cuSFM has no equivalent of.

A converged solve can still leave a point outside the world or inside a camera:
`filter_degenerate_points` removes what the depth and bounds tests reject,
`filter_projection_failures` removes observations whose projection is not finite
or lands absurdly far off the sensor, and `filter_past_rational_pole` removes
observations at or past a `FULL_OPENCV` camera's rational distortion pole. The
docstrings carry the measurements that justify them.

Their own module because they are a *policy about points*, applied between the
solve and the next round, and they are exercised directly by tests that build a
degenerate model on purpose rather than through a whole mapping run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pycolmap
from jaxtyping import Bool, Float64, Int64
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
    """Dropped by the pre-solve guard: a non-finite coordinate, or a point farther than
    `depth_threshold` from every camera that observes it."""

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
    """Delete every point whose world position is non-finite or farther than `depth_threshold` from every camera that
    observes it.

    The bound is relative to the cameras, not a box around the world origin: cuSFM's only spatial filter is a one-sided
    cap on world Z (keypoints_mapper_main.md §5.5), and a cap on the world coordinates deletes a whole walk once it
    leaves the box (Casa Loma 006 left ±300 m after 210 s, and 84 % of its images lost every point). A point is
    measured from its nearest observing camera centre, so what it bounds is the depth a camera could have measured it
    at, wherever the walk goes. On Galileo the guard removes 0 points in all five rounds.

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
        depth_threshold: `vision_mapping_config.depth_threshold`, in metres: the largest distance from a point to its
            nearest observing camera centre.

    Returns:
        Observations removed, i.e. the summed track length of the deleted points.
    """
    # One walk of the map, then each registered image's observation list (as `filter_past_rational_pole` does):
    # `reconstruction.point3D(id)` copies the point and its whole track.
    points: list[tuple[int, pycolmap.Point3D]] = list(reconstruction.points3D.items())
    if not points:
        return 0
    point3D_ids: Int64[ndarray, " num_points"] = np.asarray([point3D_id for point3D_id, _ in points], dtype=np.int64)
    points_xyz: Float64[ndarray, "num_points 3"] = np.array([point.xyz for _, point in points], dtype=np.float64)
    row_of: dict[int, int] = {int(point3D_id): row for row, point3D_id in enumerate(point3D_ids.tolist())}
    # A point no registered camera observes keeps +inf and goes.
    nearest_m: Float64[ndarray, " num_points"] = np.full(len(points), np.inf, dtype=np.float64)
    for image_id in reconstruction.reg_image_ids():
        image: pycolmap.Image = reconstruction.image(image_id)
        rows: Int64[ndarray, " n"] = np.fromiter((row_of[int(point2D.point3D_id)] for point2D in image.get_observation_points2D()), dtype=np.int64)
        if not rows.size:
            continue
        offset: Float64[ndarray, "n 3"] = points_xyz[rows] - np.asarray(image.projection_center(), dtype=np.float64)
        np.minimum.at(nearest_m, rows, np.sqrt(np.einsum("ni,ni->n", offset, offset)))
    # `nan <= threshold` is False, so a non-finite point or centre goes too.
    out_of_bounds: Bool[ndarray, " num_points"] = ~np.isfinite(points_xyz).all(axis=1) | ~(nearest_m <= depth_threshold)
    removed: int = 0
    for point3D_id in point3D_ids[out_of_bounds].tolist():
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


def filter_projection_failures(observation_manager: pycolmap.ObservationManager, reconstruction: pycolmap.Reconstruction) -> int:
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


def rational_pole_angle(camera: pycolmap.Camera) -> float:
    """The angle off the optical axis where a `FULL_OPENCV` camera's rational distortion reaches its first pole.

    `FULL_OPENCV`'s radial factor `(1 + k1 r^2 + k2 r^4 + k3 r^6) / (1 + k4 r^2 + k5 r^4 + k6 r^6)` is infinite
    at the denominator's first positive root in `r^2`, found here analytically: when the numerator nearly cancels it
    the pole is a spike too narrow to sample. MSD MGO07's cam 2 has one at r = 1.32422 (52.9 degrees, inside the image
    at v = 592 of 640), where the projection jumps to -1.9e10 px within 1e-7 of r; a central-difference Jacobian across
    it made an extrinsics solve fail on its first evaluation.

    This is a guard against that one singularity, not a general validity test of the camera model. Every other model,
    and a `FULL_OPENCV` camera without a positive root, gets the whole front half-space: radial folds before the pole
    and folds of the tangential terms between image axes (`p1 = p2 = 0.1` folds along the negative diagonal at
    r = 1.17851) are out of scope; the extrinsic solve's finite-evaluation backstop catches only non-finite results.

    Args:
        camera: The camera.

    Returns:
        The angle in radians; pi/2 when the model has no rational pole (COLMAP projects nothing at or behind z = 0).
    """
    if camera.model != pycolmap.CameraModelId.FULL_OPENCV:
        return 0.5 * np.pi
    k4, k5, k6 = (float(value) for value in camera.params[9:12])
    poles: list[float] = [float(root.real) for root in np.roots([k6, k5, k4, 1.0]) if abs(root.imag) < 1e-12 and root.real > 0.0]
    return float(np.arctan(np.sqrt(min(poles)))) if poles else 0.5 * np.pi


def before_rational_pole(points_in_camera: Float64[ndarray, "n 3"], pole_angle: float) -> Bool[ndarray, " n"]:
    """Which camera-frame points lie in front of the camera and inside its `rational_pole_angle`.

    Args:
        points_in_camera: Points in the camera frame, metres.
        pole_angle: The camera's `rational_pole_angle`.

    Returns:
        True for points in front of the camera and short of the pole.
    """
    off_axis: Float64[ndarray, " n"] = np.arctan2(np.hypot(points_in_camera[:, 0], points_in_camera[:, 1]), points_in_camera[:, 2])
    return (points_in_camera[:, 2] > 0.0) & (off_axis < pole_angle)


def filter_past_rational_pole(observation_manager: pycolmap.ObservationManager, reconstruction: pycolmap.Reconstruction) -> int:
    """Delete every observation whose point lies at or past its camera's rational pole (`rational_pole_angle`).

    Runs at the bundle-adjustment boundary (`colsfm.mapping.solve_bundle_adjustment`), before the adjuster is built and
    again after the solve, so the rounds, the polish and extrinsic refinement never hand the solver such an observation
    and never read one back; the extrinsic gather applies the same `before_rational_pole` test to its own folded points.
    Cameras without a pole are skipped (only a point behind them would fail, and `filter_projection_failures` already
    removes those).

    Args:
        observation_manager: Bookkeeping bound to the reconstruction.
        reconstruction: The model to filter, edited in place.

    Returns:
        Observations removed: the reconstruction's observation count before minus after, so a deletion that takes a
        whole two-observation track counts both.
    """
    pole_angles: dict[int, float] = {camera_id: rational_pole_angle(camera) for camera_id, camera in reconstruction.cameras.items()}
    limited: set[int] = {camera_id for camera_id, angle in pole_angles.items() if angle < 0.5 * np.pi}
    if not limited:
        return 0
    before: int = reconstruction.compute_num_observations()
    # One pass over the points, then each image's observation list: `reconstruction.point3D(id)` copies the point and
    # its whole track, which per observation made this 83 s per call on MGO07 (1.4M observations, load ~65).
    xyz_by_id: dict[int, ndarray] = {int(point3D_id): point.xyz for point3D_id, point in reconstruction.points3D.items()}
    for image_id in reconstruction.reg_image_ids():
        image: pycolmap.Image = reconstruction.image(image_id)
        if image.camera_id not in limited:
            continue
        point2D_idxs: list[int] = list(image.get_observation_point2D_idxs())
        if not point2D_idxs:
            continue
        world: Float64[ndarray, "n 3"] = np.asarray(
            [xyz_by_id[int(point2D.point3D_id)] for point2D in image.get_observation_points2D()], dtype=np.float64
        )
        cam_from_world: pycolmap.Rigid3d = image.cam_from_world()
        # einsum, not `@`: a [n, 3] x [3, 3] product through MKL's thread pool spun 170 CPU-seconds per call on a busy host.
        in_camera: Float64[ndarray, "n 3"] = np.einsum(
            "ij,nj->ni", np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64), world
        ) + np.asarray(cam_from_world.translation, dtype=np.float64)
        for row in np.flatnonzero(~before_rational_pole(in_camera, pole_angles[image.camera_id])):
            point2D_idx: int = int(point2D_idxs[int(row)])
            # A deletion that leaves a track below two observations deletes the point and every other observation
            # of it, this image's included, so each row is checked again before its own deletion.
            if image.point2D(point2D_idx).has_point3D():
                observation_manager.delete_observation(image_id, point2D_idx)
    return before - reconstruction.compute_num_observations()


def _filter_points(
    observation_manager: pycolmap.ObservationManager,
    reconstruction: pycolmap.Reconstruction,
    mapping_config: VisionMappingConfig,
    max_pixel_error: float,
) -> _FilterCounts:
    """Apply cuSFM's point filters for one round, before its bundle adjustment (§5.4-§5.5).

    The reprojection gate decays per round; the triangulation angle is the max
    over observing pairs, matching cuSFM's `HasLargeEnoughTriangulateAngle`.
    `filter_degenerate_points` is the depth test: a bound on the distance to the
    nearest observing camera, where cuSFM caps world Z; see its docstring.

    Args:
        observation_manager: Bookkeeping bound to the reconstruction.
        reconstruction: The model to filter.
        mapping_config: The mapping configuration.
        max_pixel_error: The round's gate, in pixels.

    Returns:
        Observations removed, split by cause.
    """
    reprojection_and_angle: int = observation_manager.filter_all_points3D(max_pixel_error, mapping_config.min_triangulation_deg)
    negative_depth: int = observation_manager.filter_observations_with_negative_depth()
    short_track: int = observation_manager.filter_points3D_with_short_tracks(mapping_config.min_correspondences)
    degenerate: int = filter_degenerate_points(observation_manager, reconstruction, mapping_config.depth_threshold)
    return _FilterCounts(
        reprojection_and_angle=reprojection_and_angle,
        negative_depth=negative_depth,
        short_track=short_track,
        degenerate=degenerate,
    )
