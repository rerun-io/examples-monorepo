"""MAMMA's Vicon radial lens (the release's ``vicon_radial_2``), for numpy arrays and torch tensors.

The lens maps a distorted (image) pixel ``d`` to its undistorted (pinhole) pixel
``u = c + (d - c) (1 + k1 r^2 + k2 r^4 + k3 r^6)``, ``r = |d - c|``, in the camera's native pixel grid. Projection is a
pinhole followed by the inverse of this map, found by fixed-point iteration (``distort``).

On the catalog the lens is three AnyValues cells on the camera node (dataforge ``mamma_layers.log_cameras``):
``camera_model`` = ``vicon_radial_2``, ``vicon_radial_2`` = ``[cx, cy, k1, k2, k3]``, and the native grid as the shared
camera-source keys ``source_width`` / ``source_height``. ``simplecv.catalog_calibration.read_lens`` reads them back.
"""

from dataclasses import dataclass
from typing import TypeAlias, overload

import torch
from jaxtyping import Bool, Float64
from numpy import ndarray
from serde import serde
from torch import Tensor

Pixels: TypeAlias = Float64[ndarray, "n 2"] | Float64[Tensor, "n 2"]
"""Float64 pixels as either kind the lens takes."""

VICON_RADIAL_MODEL: str = "vicon_radial_2"
"""The ``camera_model`` value of a Vicon radial camera, and the name of the cell that holds ``[cx, cy, k1, k2, k3]``."""
LENS_ITERATIONS: int = 50
"""Fixed-point steps of the lens inversion; converged points stop moving after a few."""
LENS_TOLERANCE_PX: float = 1e-9
"""Largest step, in pixels, of a converged lens inversion."""
LENS_RESIDUAL_PX: float = 1e-6
"""Largest error, in pixels, of an inverted point mapped back; a point that misses it did not converge."""


@serde
@dataclass(frozen=True, slots=True)
class ViconRadialLens:
    """The ``vicon_radial_2`` lens in the camera's native pixel grid (see the module docstring).

    ``undistort`` and ``distort`` take float64 pixels as a numpy array or a torch tensor (any device) and return the
    same kind; both go through the one formula below.
    """

    centre: Float64[ndarray, "2"]
    """Distortion centre, native pixels."""
    radial: Float64[ndarray, "3"]
    """k1, k2, k3 per pixel^2, pixel^4, pixel^6."""
    native_wh: tuple[int, int]
    """Width and height of the grid the lens is stated in (the camera node's ``source_width`` / ``source_height``)."""

    @overload
    def undistort(self, distorted: Float64[ndarray, "n 2"]) -> Float64[ndarray, "n 2"]: ...
    @overload
    def undistort(self, distorted: Float64[Tensor, "n 2"]) -> Float64[Tensor, "n 2"]: ...
    def undistort(self, distorted: Pixels) -> Pixels:
        """Distorted (image) pixels to undistorted (pinhole) pixels, native grid: the release's map."""
        centre, radial = self._parameters_like(distorted)
        offset: Pixels = distorted - centre
        return centre + offset * _radial_scale(offset, radial)

    @overload
    def distort(self, undistorted: Float64[ndarray, "n 2"]) -> Float64[ndarray, "n 2"]: ...
    @overload
    def distort(self, undistorted: Float64[Tensor, "n 2"]) -> Float64[Tensor, "n 2"]: ...
    def distort(self, undistorted: Pixels) -> Pixels:
        """Undistorted pixels to image pixels by fixed-point inversion, native grid; a point that does not converge is NaN."""
        # Only operators and methods numpy and torch share, so one body serves both kinds.
        centre, radial = self._parameters_like(undistorted)
        target: Pixels = undistorted - centre
        offset: Pixels = target
        for _ in range(LENS_ITERATIONS):
            step: Pixels = target / _radial_scale(offset, radial) - offset
            offset = offset + step
            moved: Pixels = abs(step)
            finite: Float64[ndarray, " k"] | Float64[Tensor, " k"] = moved[moved < float("inf")]  # NaN points (behind the camera) stop no one
            if len(finite) == 0 or not finite.max() > LENS_TOLERANCE_PX:
                break
        distorted: Pixels = centre + offset
        converged: Bool[ndarray, " n"] | Bool[Tensor, " n"] = (abs(self.undistort(distorted) - undistorted) <= LENS_RESIDUAL_PX).all(-1)
        distorted[~converged] = float("nan")
        return distorted

    def _parameters_like(
        self, points: Pixels
    ) -> tuple[Float64[ndarray, "2"], Float64[ndarray, "3"]] | tuple[Float64[Tensor, "2"], Float64[Tensor, "3"]]:
        """``centre`` and ``radial`` as the kind of ``points`` (a tensor on their device)."""
        if isinstance(points, Tensor):
            return (
                torch.as_tensor(self.centre, dtype=torch.float64, device=points.device),
                torch.as_tensor(self.radial, dtype=torch.float64, device=points.device),
            )
        return self.centre, self.radial


def _radial_scale(offset: Pixels, radial: Float64[ndarray, "3"] | Float64[Tensor, "3"]) -> Float64[ndarray, "n 1"] | Float64[Tensor, "n 1"]:
    """``1 + k1 r^2 + k2 r^4 + k3 r^6`` of each offset from the centre: the lens, for numpy and torch alike."""
    r2: Float64[ndarray, "n 1"] | Float64[Tensor, "n 1"] = (offset**2).sum(-1)[..., None]
    return 1.0 + radial[0] * r2 + radial[1] * r2**2 + radial[2] * r2**3
