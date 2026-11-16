"""The guards that run after a bundle adjustment, which cuSFM has no equivalent of.

A converged solve can still leave a point outside the world or inside a camera:
`filter_degenerate_points` removes what the depth and bounds tests reject,
`filter_projection_failures` removes observations whose projection is not finite
or lands absurdly far off the sensor, and `filter_outside_camera_domain` removes
observations a camera model cannot represent (behind it, or past the angle where
its distortion folds back). The docstrings carry the measurements that justify them.

Their own module because they are a *policy about points*, applied between the
solve and the next round, and they are exercised directly by tests that build a
degenerate model on purpose rather than through a whole mapping run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

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
    outside_domain: int = 0
    """Dropped for lying outside the camera model's valid domain (`filter_outside_camera_domain`)."""

    def total(self) -> int:
        """Total observations removed.

        Returns:
            The sum of every cause.
        """
        return self.reprojection_and_angle + self.negative_depth + self.short_track + self.degenerate + self.outside_domain


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
    points_xyz: Float64[ndarray, "num_points 3"] = np.array([point.xyz for _, point in points], dtype=np.float64)
    # `np.abs(nan) > threshold` is False, so the finiteness term has to carry the NaNs.
    out_of_bounds: Bool[ndarray, "num_points"] = ~np.isfinite(points_xyz).all(axis=1) | (np.abs(points_xyz).max(axis=1) > depth_threshold)
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


DOMAIN_ANGLE_SAMPLES: Final[int] = 2048
"""Viewing angles sampled between the optical axis and 90 degrees by `camera_domain_angle`."""


def camera_domain_angle(camera: pycolmap.Camera) -> float:
    """The largest angle off the optical axis over which a camera model is a valid projection.

    A projection is valid while the image radius grows with the viewing angle, up to the
    first pole of a rational model. `FULL_OPENCV`'s factor
    `(1 + k1 r^2 + k2 r^4 + k3 r^6) / (1 + k4 r^2 + k5 r^4 + k6 r^6)` has a pole at the
    denominator's first positive root, found analytically: when the numerator nearly cancels
    it the pole is a spike too narrow to sample. MSD MGO07's cam 2 has one at r = 1.32422
    (52.9 degrees, inside the image at v = 592 of 640), where the projection jumps to
    -1.9e10 px within 1e-7 of r; a central-difference Jacobian across it made an extrinsics
    solve fail on its first evaluation. Monotonicity is read off COLMAP's own `img_from_cam`
    along the four image axes, so every model gets the same rule; a pinhole camera's domain
    is the whole front half-space.

    Args:
        camera: The camera.

    Returns:
        The angle in radians, at most pi/2 (COLMAP projects nothing at or behind z = 0).
    """
    angles: Float64[ndarray, " n"] = np.linspace(0.0, 0.5 * np.pi, DOMAIN_ANGLE_SAMPLES, endpoint=False)
    centre: Float64[ndarray, "2"] = np.array([camera.principal_point_x, camera.principal_point_y], dtype=np.float64)
    limit: float = 0.5 * np.pi
    if camera.model == pycolmap.CameraModelId.FULL_OPENCV:
        k4, k5, k6 = (float(value) for value in camera.params[9:12])
        poles: list[float] = [float(root.real) for root in np.roots([k6, k5, k4, 1.0]) if abs(root.imag) < 1e-12 and root.real > 0.0]
        if poles:
            limit = float(np.arctan(np.sqrt(min(poles))))
    angles = angles[angles < limit]
    for axis_x, axis_y in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
        rays: Float64[ndarray, "n 3"] = np.stack([np.sin(angles) * axis_x, np.sin(angles) * axis_y, np.cos(angles)], axis=1)
        radius: Float64[ndarray, " n"] = np.linalg.norm(np.asarray(camera.img_from_cam(rays), dtype=np.float64) - centre, axis=1)
        valid: Bool[ndarray, " n"] = np.isfinite(radius)
        valid[1:] &= np.diff(radius) > 0.0
        invalid: ndarray = np.flatnonzero(~valid)
        if invalid.size:
            limit = min(limit, float(angles[max(int(invalid[0]) - 1, 0)]))
    return limit


def in_camera_domain(points_in_camera: Float64[ndarray, "n 3"], domain_angle: float) -> Bool[ndarray, " n"]:
    """Which camera-frame points the camera model can project (`camera_domain_angle`).

    Args:
        points_in_camera: Points in the camera frame, metres.
        domain_angle: The camera's `camera_domain_angle`.

    Returns:
        True for points in front of the camera and inside the valid angle.
    """
    off_axis: Float64[ndarray, " n"] = np.arctan2(np.hypot(points_in_camera[:, 0], points_in_camera[:, 1]), points_in_camera[:, 2])
    return (points_in_camera[:, 2] > 0.0) & (off_axis < domain_angle)


def filter_outside_camera_domain(observation_manager: pycolmap.ObservationManager, reconstruction: pycolmap.Reconstruction) -> int:
    """Delete every observation whose point lies outside its camera model's valid domain.

    The one domain rule of every solve: each mapping round runs it before its solve
    (`_filter_points`), the bundle-adjustment boundary (`colsfm.mapping.solve_bundle_adjustment`)
    after every solve, so the rounds, the polish and extrinsic refinement all read a model
    without such observations; the extrinsic gather applies the same `in_camera_domain`
    test to its own folded points.
    Cameras whose domain is the whole front half-space are skipped (only a point behind
    them would fail, and `filter_projection_failures` already removes those).

    Args:
        observation_manager: Bookkeeping bound to the reconstruction.
        reconstruction: The model to filter, edited in place.

    Returns:
        Observations removed.
    """
    domains: dict[int, float] = {camera_id: camera_domain_angle(camera) for camera_id, camera in reconstruction.cameras.items()}
    limited: set[int] = {camera_id for camera_id, angle in domains.items() if angle < 0.5 * np.pi}
    if not limited:
        return 0
    # One pass over the points, then each image's observation list: `reconstruction.point3D(id)` copies the point and
    # its whole track, which per observation made this 83 s per call on MGO07 (1.4M observations, load ~65).
    xyz_by_id: dict[int, ndarray] = {int(point3D_id): point.xyz for point3D_id, point in reconstruction.points3D.items()}
    removed: int = 0
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
        for row in np.flatnonzero(~in_camera_domain(in_camera, domains[image.camera_id])):
            point2D_idx: int = int(point2D_idxs[int(row)])
            # A deletion that leaves a track below two observations deletes the point and every other observation
            # of it, this image's included, so each row is checked again before its own deletion.
            if image.point2D(point2D_idx).has_point3D():
                observation_manager.delete_observation(image_id, point2D_idx)
                removed += 1
    return removed


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
    reprojection_and_angle: int = observation_manager.filter_all_points3D(max_pixel_error, mapping_config.min_triangulation_deg)
    negative_depth: int = observation_manager.filter_observations_with_negative_depth()
    short_track: int = observation_manager.filter_points3D_with_short_tracks(mapping_config.min_correspondences)
    degenerate: int = filter_degenerate_points(observation_manager, reconstruction, mapping_config.depth_threshold)
    outside_domain: int = filter_outside_camera_domain(observation_manager, reconstruction)
    return _FilterCounts(
        reprojection_and_angle=reprojection_and_angle,
        negative_depth=negative_depth,
        short_track=short_track,
        degenerate=degenerate,
        outside_domain=outside_domain,
    )
