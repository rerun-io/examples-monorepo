"""Numpy boundary. Every input is checked for dtype and the documented shape.

The public constructor takes warm-fit config numbers, in order: phi,
`dist_weight`, `temporal_weight`, translation unit, limit margin, max iterations,
relative tolerance, absolute tolerance, initial damping. Geometry is already
scaled to the subject (mm). Optional entries 9:14 are init iterations, init relative
tolerance, rotation hypotheses, full fit hypotheses, and finger starts. Camera and wrist translations are metres.
"""

from typing import Literal

import numpy as np
from jaxtyping import Bool, Float32, Float64, Int64

class FitOutput:
    rotation: Float32[np.ndarray, "3 3"]  # (3, 3), world from wrist
    translation: Float32[np.ndarray, "3"]  # (3,), metres
    joint_angles: Float32[np.ndarray, "22"]  # (22,), last two carried unchanged
    e_2d: float
    e_dist: float
    e_temporal: float
    energy: float
    rigid_iterations: list[int]
    chosen_hypotheses: list[int]
    full_iterations: list[int]
    winner: int | None
    iterations: int
    converged: bool
    termination: Literal["tolerance", "stationary", "iterations", "damping", "non_finite", "no_evidence"]

class HandFitter:
    def __init__(
        self,
        axes: Float32[np.ndarray, "20 3"],
        pivots: Float32[np.ndarray, "20 3"],
        rest: Float32[np.ndarray, "21 3"],
        weights: Float32[np.ndarray, "21 3"],
        limits: Float32[np.ndarray, "20 2"],
        indices: Int64[np.ndarray, "21 3"],
        topology: Int64[np.ndarray, "4 22"],
        config: Float64[np.ndarray, "9"] | Float64[np.ndarray, "14"],
    ) -> None:
        """Shapes: (20,3), (20,3), (21,3), (21,3), (20,2), (21,3), (4,22), (9,) or (14,).

        Topology rows: parent, frame index, first child, next sibling.
        """
        ...
    def add_camera(
        self,
        cam_from_rig: Float32[np.ndarray, "4 4"],
        focal: Float32[np.ndarray, "2"],
        principal: Float32[np.ndarray, "2"],
        distortion: Float32[np.ndarray, "8"] | None = None,
    ) -> int:
        """Shapes (4,4), (2,), (2,), (8,) or None for pinhole. Returns camera ID."""
        ...
    def fit(
        self,
        sides: Int64[np.ndarray, "h"],
        rotations: Float32[np.ndarray, "h 3 3"],
        translations: Float32[np.ndarray, "h 3"],
        angles: Float32[np.ndarray, "h 22"],
        camera_indices: Int64[np.ndarray, "h 2"],
        world_from_rig: Float32[np.ndarray, "h 2 4 4"],
        pixels: Float32[np.ndarray, "h 2 21 2"],
        weights: Float32[np.ndarray, "h 2 21"],
        distances: Float32[np.ndarray, "h 2 21"],
        *,
        has_previous: Bool[np.ndarray, "h"] | None = None,
        central_difference: bool = False,
    ) -> list[FitOutput]:
        """Shapes (h,), (h,3,3), (h,3), (h,22), (h,2), (h,2,4,4), (h,2,21,2), (h,2,21), (h,2,21).

        has_previous defaults to all warm; false rows ignore their supplied pose. Side 0 = left, 1 = right.
        Camera ID -1 marks an absent slot. Zero weights mask NaNs in observations.
        The GIL is released for all solves. Results preserve input order.
        """
        ...

__version__: str
