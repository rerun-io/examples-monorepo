"""The observation graph of an extrinsic refinement: who saw what, and where.

Split out of `colsfm.extrinsic_refinement` because this half of the problem answers a
different question from the rest of the stage. The refinement alternates a pyceres solve
with pycolmap's bundle adjustment; bundle adjustment moves rig poses and 3D points but
adds and removes no observation, so which image saw which point never changes and is
indexed once (`build_observation_index`), while only the geometry is folded in again per
round (`camera_observations`). This module owns that separation, and nothing about the
Ceres problem the observations feed.

## What the blob's relative-block count means, measured

`co_observed_camera_pairs` reproduces the blob's `camera rig extrinsic constraint` set.
Its **777 relative blocks** on Galileo are *not* `4 declared stereo pairs x N frames`:
Galileo's 226 keyframes fall into 29 rig frames holding 8, 8, ..., 6 and 4 cameras, and

```
sum over rig frames of C(cameras in that frame, 2) = 27*28 + 15 + 6 = 777
```

exactly. So the constraint is added for **every unordered pair of cameras that fired in
the same rig frame**, once per rig frame. Because a rig's extrinsics are shared across
frames, all the blocks of one pair carry an identical residual, which is why this module
returns pairs with multiplicities rather than a block per frame; `colsfm.extrinsic_costs`
turns the multiplicity back into the blob's cost.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import pycolmap
from jaxtyping import Float64, Int64
from numpy import ndarray

from colsfm.geometry import Vector3
from colsfm.reconstruction import RigReference

PointsInVehicle: TypeAlias = Float64[ndarray, "n_obs 3"]
"""Observed 3D points expressed in the vehicle (FLU) frame of their own rig instant."""

PixelObservations: TypeAlias = Float64[ndarray, "n_obs 2"]
"""Measured keypoint positions in pixels."""

CameraPairKey: TypeAlias = tuple[int, int]
"""Two `camera_params_id`s, ascending; one inter-camera extrinsic constraint."""


@dataclass(frozen=True, slots=True)
class CameraObservations:
    """Every observation of one camera, with the frozen geometry already folded in.

    Rig poses and 3D points are constant during solve (A), so each observation reduces
    to the point expressed in its own rig instant's vehicle frame. Projecting it needs
    only `vehicle_T_cam`, which is the parameter.
    """

    camera_params_id: int
    """The camera these observations belong to."""
    camera: pycolmap.Camera
    """Its intrinsics, used batched through `Camera.img_from_cam`."""
    points_in_vehicle: PointsInVehicle
    """`vehicle_T_world(frame) * X` per observation, metres."""
    observed_px: PixelObservations
    """The measured keypoint of each observation, pixels."""


@dataclass(frozen=True, slots=True)
class ImageObservationIndex:
    """Which points one image observes, and where in that image it saw them.

    The two array fields are *rows*, not ids: turning an observation's point id into a
    row of the sorted point array is a `searchsorted` per image, and neither the ids
    nor the sort changes across the alternation, so it is done once here rather than
    once per round per image.
    """

    frame_id: int
    """The rig instant this image belongs to; picks the pose to fold in."""
    rows: Int64[ndarray, " n_obs"]
    """Row of each observation's point in `ObservationIndex.point3D_ids`."""
    observed_px: PixelObservations
    """The measured keypoint of each of those observations, pixels; same order as `rows`."""


@dataclass(frozen=True, slots=True)
class CameraObservationIndex:
    """One camera's share of the observation graph, without any geometry folded in.

    This is the half of `CameraObservations` that a round of the alternation cannot
    change: `solve_bundle_adjustment` moves poses and points but adds and removes no
    observation, so the images, the point ids and the pixels are the same every round
    and are built once. Only the geometry has to be gathered again.
    """

    camera_params_id: int
    """The camera these observations belong to."""
    camera: pycolmap.Camera
    """Its intrinsics, used batched through `Camera.img_from_cam`."""
    images: tuple[ImageObservationIndex, ...]
    """Its images that observe at least one point, in ascending image id order."""


@dataclass(frozen=True, slots=True)
class ObservationIndex:
    """The whole observation graph of one model, resolved to array rows.

    Everything a round of the alternation cannot change, held together because the
    per-image rows only mean anything against this exact point ordering.
    """

    point3D_ids: Int64[ndarray, " n_points"]
    """Every point id `points3D` holds, ascending; the rows the image indices refer to."""
    stored_point3D_ids: Int64[ndarray, " n_points"]
    """The same ids in `points3D`'s own iteration order, as they stood at build time."""
    order: Int64[ndarray, " n_points"]
    """The permutation from `stored_point3D_ids` to `point3D_ids`; `argsort`, done once."""
    by_camera: dict[int, CameraObservationIndex]
    """One entry per `camera_params_id` that owns at least one observation."""

    def positions(self, reconstruction: pycolmap.Reconstruction) -> Float64[ndarray, "n_points 3"]:
        """This round's world positions, in `point3D_ids` order.

        The one thing that does change per round, and the only pass over `points3D`
        the round pays for: the sort that puts them in `point3D_ids` order was done
        when the index was built.

        Args:
            reconstruction: The model as the last solve left it.

        Returns:
            Float64 `[n_points, 3]` positions, row `i` being `point3D_ids[i]`'s.

        Raises:
            ValueError: When the model no longer holds the points this index was built
                on. Every row in it would then name a different point, so this is a
                loud failure rather than a quietly wrong reprojection. Bundle
                adjustment inside the alternation filters nothing, so it does not
                arise; a caller reusing an index across a filter would hit it.
        """
        stored_ids, stored_xyz = _stored_points(reconstruction)
        if stored_ids.shape != self.stored_point3D_ids.shape or not np.array_equal(stored_ids, self.stored_point3D_ids):
            raise ValueError(
                f"this observation index was built on {len(self.stored_point3D_ids)} points and the model now holds "
                f"{len(stored_ids)} different ones; rebuild it with `build_observation_index`"
            )
        return stored_xyz[self.order]


def _stored_points(
    reconstruction: pycolmap.Reconstruction,
) -> tuple[Int64[ndarray, " n_points"], Float64[ndarray, "n_points 3"]]:
    """Read every point id and position out in one pass, in `points3D`'s own order.

    Args:
        reconstruction: The model to read.

    Returns:
        The ids and their world positions, in the order `points3D` iterates them.
    """
    point3D_ids: list[int] = []
    positions: list[Vector3] = []
    for point3D_id, point in reconstruction.points3D.items():
        point3D_ids.append(int(point3D_id))
        positions.append(point.xyz)
    if not point3D_ids:
        return np.empty(0, dtype=np.int64), np.empty((0, 3), dtype=np.float64)
    return np.asarray(point3D_ids, dtype=np.int64), np.asarray(positions, dtype=np.float64)


def build_observation_index(reconstruction: pycolmap.Reconstruction) -> ObservationIndex:
    """Index every observation by camera and image, once, for the whole alternation.

    The point ids are sorted here and every observation is resolved to a row of that
    sorted array here, so a round has only to read the positions and gather them.

    Args:
        reconstruction: A posed, triangulated model.

    Returns:
        The index: the observed point ids and one `CameraObservationIndex` per
        `camera_params_id` that owns at least one observation.
    """
    stored_ids, _ = _stored_points(reconstruction)
    order: Int64[ndarray, " n_points"] = np.argsort(stored_ids)
    sorted_ids: Int64[ndarray, " n_points"] = stored_ids[order]
    images_by_camera: dict[int, list[ImageObservationIndex]] = {}
    for image_id in sorted(reconstruction.images):
        image: pycolmap.Image = reconstruction.image(image_id)
        point3D_ids: list[int] = []
        observed: list[Vector3] = []
        for point2D in image.points2D:
            if not point2D.has_point3D():
                continue
            point3D_ids.append(int(point2D.point3D_id))
            observed.append(point2D.xy)
        if not point3D_ids:
            continue
        images_by_camera.setdefault(image.camera_id, []).append(
            ImageObservationIndex(
                frame_id=image.frame_id,
                rows=np.searchsorted(sorted_ids, np.asarray(point3D_ids, dtype=np.int64)).astype(np.int64),
                observed_px=np.asarray(observed, dtype=np.float64),
            )
        )
    return ObservationIndex(
        point3D_ids=sorted_ids,
        stored_point3D_ids=stored_ids,
        order=order,
        by_camera={
            camera_params_id: CameraObservationIndex(
                camera_params_id=camera_params_id,
                camera=reconstruction.camera(camera_params_id),
                images=tuple(images),
            )
            for camera_params_id, images in sorted(images_by_camera.items())
        },
    )


def camera_observations(
    reconstruction: pycolmap.Reconstruction,
    rig_reference: RigReference,
    index: ObservationIndex | None = None,
) -> dict[int, CameraObservations]:
    """Fold the frozen rig poses and points into per-camera observation arrays.

    Only this half is per-round work, and it is now only that: one pass over `points3D`
    for the positions, then one fancy index and one rigid transform per image. The
    per-image `searchsorted`, the presence mask and the sort of the point ids do not
    change across the alternation — bundle adjustment inside it moves poses and points
    but adds and removes no observation — so they belong to `build_observation_index`,
    which runs once.

    Args:
        reconstruction: A posed, triangulated model built on a camera-referenced rig.
        rig_reference: How that rig was built, from `colsfm.reconstruction.rig_reference`.
        index: A prebuilt observation index for this model, or None to build one.

    Returns:
        One `CameraObservations` per `camera_params_id` that owns at least one
        observation, keyed by that id.
    """
    resolved_index: ObservationIndex = build_observation_index(reconstruction) if index is None else index
    vehicle_T_world_by_frame_id: dict[int, pycolmap.Rigid3d] = {
        frame_id: world_T_vehicle.inverse()
        for frame_id, world_T_vehicle in rig_reference.world_T_vehicle_by_frame_id(reconstruction).items()
    }
    points_xyz: Float64[ndarray, "n_points 3"] = resolved_index.positions(reconstruction)

    observations: dict[int, CameraObservations] = {}
    for camera_params_id, camera_index in sorted(resolved_index.by_camera.items()):
        points_in_vehicle: list[PointsInVehicle] = []
        pixels: list[PixelObservations] = []
        for image_index in camera_index.images:
            vehicle_T_world: pycolmap.Rigid3d | None = vehicle_T_world_by_frame_id.get(image_index.frame_id)
            if vehicle_T_world is None:
                continue
            world_points: PointsInVehicle = points_xyz[image_index.rows]
            points_in_vehicle.append(
                world_points @ np.asarray(vehicle_T_world.rotation.matrix(), dtype=np.float64).T
                + np.asarray(vehicle_T_world.translation, dtype=np.float64)
            )
            pixels.append(image_index.observed_px)
        if not points_in_vehicle:
            continue
        observations[camera_params_id] = CameraObservations(
            camera_params_id=camera_params_id,
            camera=camera_index.camera,
            points_in_vehicle=np.ascontiguousarray(np.concatenate(points_in_vehicle, axis=0)),
            observed_px=np.ascontiguousarray(np.concatenate(pixels, axis=0)),
        )
    return observations


def co_observed_camera_pairs(reconstruction: pycolmap.Reconstruction) -> dict[CameraPairKey, int]:
    """Count the rig frames in which each unordered pair of cameras fired together.

    This is the blob's `camera rig extrinsic constraint` set: one block per pair per rig
    frame. On Galileo the counts sum to 777, matching
    `bundle_adjustment_solver.cc:1138`.

    Args:
        reconstruction: A model whose frames group the images of one rig instant.

    Returns:
        `{(lower camera_params_id, higher camera_params_id): number of rig frames}`.
    """
    cameras_by_frame: dict[int, set[int]] = {}
    for image_id in sorted(reconstruction.images):
        image: pycolmap.Image = reconstruction.image(image_id)
        cameras_by_frame.setdefault(image.frame_id, set()).add(image.camera_id)
    pair_counts: dict[CameraPairKey, int] = {}
    for cameras in cameras_by_frame.values():
        for lower, higher in itertools.combinations(sorted(cameras), 2):
            pair_counts[(lower, higher)] = pair_counts.get((lower, higher), 0) + 1
    return dict(sorted(pair_counts.items()))

