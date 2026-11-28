"""The bundle-adjustment problem, options and result that cross into the Mojo solver.

One array contract for every caller: colsfm reads a `pycolmap.Reconstruction` into a `BaProblem`, the tests build
one from numpy, and COLMAP's C++ can fill the same arrays through the C ABI. Host arrays are float64, COLMAP's own
precision; the solver works in float32 on the GPU, as CASPAR's generated f32 tree does.

Conventions, all COLMAP's:

- A pose row is `[qx, qy, qz, qw, tx, ty, tz]`: a rotation quaternion and a translation, `x_out = R(q) x_in + t`.
- `rig_from_world` is one row per frame. `sensor_from_rig` is one row per camera (the identity for a rig's
  reference sensor), so `cam_from_world = sensor_from_rig[camera] * rig_from_world[frame]`.
- `intrinsics` is one row per camera in COLMAP's `camera.params` order for its model, zero-padded to
  `MAX_INTRINSICS`.
- The residual of an observation is `project(cam_from_world * point) - pixel`, unweighted, with no robust loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Final

from jaxtyping import Bool, Float64, Int32
from numpy import ndarray

MAX_INTRINSICS: Final[int] = 12
"""Columns of `BaProblem.intrinsics`: FULL_OPENCV's parameter count, the widest model."""


class CameraModel(IntEnum):
    """The camera models the solver projects; the value is the row's entry in `BaProblem.camera_model`."""

    PINHOLE = 0
    """`[fx, fy, cx, cy]`."""
    SIMPLE_RADIAL = 1
    """`[f, cx, cy, k]`."""
    OPENCV_FISHEYE = 2
    """`[fx, fy, cx, cy, k1, k2, k3, k4]`."""
    FULL_OPENCV = 3
    """`[fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6]`."""


FOCAL_AND_EXTRA_COLUMNS: Final[dict[CameraModel, tuple[int, ...]]] = {
    CameraModel.PINHOLE: (0, 1),
    CameraModel.SIMPLE_RADIAL: (0, 3),
    CameraModel.OPENCV_FISHEYE: (0, 1, 4, 5, 6, 7),
    CameraModel.FULL_OPENCV: (0, 1, 4, 5, 6, 7, 8, 9, 10, 11),
}
"""Columns of an `intrinsics` row that `intrinsics_free` refines (CASPAR's FocalAndExtra node), in tangent order."""


@dataclass(frozen=True, slots=True)
class BaProblem:
    """One bundle adjustment: the state, which parts of it are free, and the observations.

    A fixed node (`*_free` False) keeps its value; its Jacobian blocks are skipped. `intrinsics_free` frees a
    camera's focal lengths and distortion coefficients together (CASPAR's "focal and extra" group); the principal
    point is never refined.
    """

    camera_model: Int32[ndarray, " n_cameras"]
    """`CameraModel` value per camera."""
    intrinsics: Float64[ndarray, "n_cameras 12"]
    """COLMAP `camera.params` per camera, zero-padded."""
    intrinsics_free: Bool[ndarray, " n_cameras"]
    """Whether a camera's focal lengths and distortion coefficients are refined."""
    sensor_from_rig: Float64[ndarray, "n_cameras 7"]
    """Rig extrinsic per camera; the identity `[0, 0, 0, 1, 0, 0, 0]` for a reference sensor."""
    sensor_free: Bool[ndarray, " n_cameras"]
    """Whether a camera's `sensor_from_rig` is refined."""
    rig_from_world: Float64[ndarray, "n_frames 7"]
    """Pose per frame."""
    pose_free: Bool[ndarray, " n_frames"]
    """Whether a frame's pose is refined; the gauge frame is not."""
    points: Float64[ndarray, "n_points 3"]
    """World points."""
    point_free: Bool[ndarray, " n_points"]
    """Whether a point is refined."""
    pixels: Float64[ndarray, "n_observations 2"]
    """Measured pixel per observation."""
    observation_frame: Int32[ndarray, " n_observations"]
    """Row of `rig_from_world` per observation."""
    observation_camera: Int32[ndarray, " n_observations"]
    """Row of the per-camera arrays per observation."""
    observation_point: Int32[ndarray, " n_observations"]
    """Row of `points` per observation."""


@dataclass(frozen=True, slots=True)
class BaOptions:
    """CASPAR's solver parameters under CASPAR's names, with colsfm's values as the defaults.

    The defaults are COLMAP's `CasparBundleAdjustmentOptions` with the reduced camera system on, which is what
    `colsfm.mapping.bundle_adjustment_options` sets.
    """

    solver_iter_max: int = 200
    """Levenberg-Marquardt trials at most."""
    diag_init: float = 1.0
    """Initial damping."""
    diag_min: float = 1e-12
    """Lower bound of the damping."""
    diag_scaling_up: float = 2.0
    """Damping growth on the first rejected step; it doubles on each further rejection."""
    diag_scaling_down: float = 0.333333
    """Lower bound of the damping shrink factor on an accepted step."""
    diag_exit_value: float = 1e3
    """The solve ends when the damping passes this after a rejected step."""
    score_exit_value: float = 0.0
    """The solve ends when the cost falls to this."""
    solver_rel_decrease_min: float = 1.0
    """A step is accepted when its cost is below this fraction of the best cost."""
    reduced_camera_system: bool = True
    """Eliminate the points inside the PCG and stop it on the Q criterion."""
    reduced_pcg_iter_max: int = 500
    """PCG iterations at most, reduced camera system."""
    reduced_pcg_eta: float = 0.1
    """Q-criterion tolerance of the reduced PCG, in (0, 1)."""
    pcg_iter_max: int = 20
    """PCG iterations at most, full system."""
    pcg_rel_error_exit: float = 1e-4
    """Full-system PCG stop: squared residual norm relative to its start."""


@dataclass(frozen=True, slots=True)
class BaIteration:
    """One Levenberg-Marquardt trial, the fields of CASPAR's verbose iteration log."""

    cost_tried: float
    """Cost at the retracted candidate."""
    cost_best: float
    """Best cost after the accept or reject."""
    damping: float
    """Damping the trial was solved with."""
    pcg_iterations: int
    """PCG iterations the trial ran."""
    accepted: bool
    """Whether the candidate became the current state."""


@dataclass(frozen=True, slots=True)
class BaResult:
    """The refined state and how the solve went. Fixed nodes come back unchanged."""

    intrinsics: Float64[ndarray, "n_cameras 12"]
    """Refined intrinsics, same layout as `BaProblem.intrinsics`."""
    sensor_from_rig: Float64[ndarray, "n_cameras 7"]
    """Refined rig extrinsics, quaternions normalised."""
    rig_from_world: Float64[ndarray, "n_frames 7"]
    """Refined poses, quaternions normalised."""
    points: Float64[ndarray, "n_points 3"]
    """Refined points."""
    initial_cost: float
    """Half the sum of squared residuals at the start (CASPAR's score)."""
    final_cost: float
    """Half the sum of squared residuals of the accepted state."""
    iterations: int
    """Levenberg-Marquardt trials run."""
    pcg_iterations: int
    """PCG iterations summed over the trials."""
    exit_reason: str
    """CASPAR's `ExitReason` name: `MAX_ITERATIONS`, `CONVERGED_DIAG_EXIT` or `CONVERGED_SCORE_THRESHOLD`."""
    solve_seconds: float
    """Wall time of the solver loop on the GPU, uploads and downloads excluded."""
    total_seconds: float
    """Wall time of the whole call, host arrays in to host arrays out."""
    trace: tuple[BaIteration, ...] = ()
    """The trials in order. After convergence float32 summation order decides accept or reject, so the length of the
    tail varies from run to run (in CASPAR too); compare solvers on the prefix up to convergence."""
