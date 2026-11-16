# SPDX-License-Identifier: Apache-2.0
"""The trajectory fit (simplecv's Umeyama) with colsfm's residual report, and nothing that needs pycolmap.

Split out of `colsfm.alignment` so that both sides of the repository can call the
same fit. `colsfm.alignment` is where the fit belongs conceptually -- it sits
beside `RigTrack` and the timestamp join -- but that module reads
`frames_meta.json` through `pycolmap.Rigid3d`, and the `demo` package runs in the
*default* pixi environment, which has no pycolmap at all. So `demo.poses` carried
its own copy of the same twenty lines of SVD, under the name `umeyama_rigid`, and
the two could only drift.

Everything here is plain NumPy. `colsfm.alignment` re-exports it, so no caller of
that module changed; `demo.poses` imports it directly. The one method that does
touch pycolmap, `RigidAlignment.apply_pose`, imports it inside the call, which is
what keeps this module importable where pycolmap is not installed.

The scale is **held at 1.0** by default and the scale a similarity fit would have
chosen is reported separately: a reconstruction that shrank the trajectory must
show up as alignment error rather than disappear into the fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, TypeAlias

import numpy as np
from jaxtyping import Float64, Int64
from numpy import ndarray
from simplecv.ops.umeyama import SimilarityTransform, umeyama_alignment

if TYPE_CHECKING:
    import pycolmap

Vector3: TypeAlias = Float64[ndarray, "3"]
"""A 3-vector: a translation in metres or a rotation axis."""

Matrix3: TypeAlias = Float64[ndarray, "3 3"]
"""A rotation matrix."""

Positions: TypeAlias = Float64[ndarray, "n 3"]
"""A trajectory as one world-frame position per sample, in metres."""

Rotations: TypeAlias = Float64[ndarray, "n 3 3"]
"""World-frame rotation matrices, one per sample."""

Timestamps: TypeAlias = Int64[ndarray, "n"]
"""Capture times in integer microseconds since the Unix epoch."""

DEGENERATE_VARIANCE: Final[float] = 1e-15
"""Source variance (m^2) at or below which a trajectory counts as not moving: the fit holds the scale at 1.0 instead of
estimating it. Far below simplecv's default (1e-9, a 32 µm spread), so a small but moving trajectory keeps its scale."""


@dataclass(frozen=True, slots=True)
class RigidAlignment:
    """A least-squares fit of one trajectory onto another: SE(3), or SIM(3) on request."""

    target_R_source: Matrix3
    """Rotation taking source positions into the target frame."""
    target_t_source: Vector3
    """Translation taking source positions into the target frame, in metres."""
    rmse_meters: float
    """Root-mean-square residual after the fit."""
    max_error_meters: float
    """Largest single residual after the fit."""
    would_be_scale: float
    """Scale a *similarity* fit would have chosen; 1.0 means metric scale survived.

    Reported whether or not the fit took it: for a rigid fit it is the diagnostic
    the benchmark prints, and for a similarity fit it equals `scale`."""
    scale: float = 1.0
    """Scale the fit actually applied — 1.0 for a rigid fit, `would_be_scale` for a
    similarity one. `apply` and `apply_pose` both honour it, so a caller never has
    to remember which kind of fit it holds."""

    def apply(self, positions: Positions) -> Positions:
        """Map source-frame positions into the target frame.

        Args:
            positions: Float64 source positions with shape `[n, 3]`.

        Returns:
            Float64 positions with shape `[n, 3]` in the target frame.
        """
        return self.scale * (positions @ self.target_R_source.T) + self.target_t_source

    def apply_rotations(self, rotations: Rotations) -> Rotations:
        """Map source-frame rotation matrices into the target frame.

        The scale does not touch a rotation, so this is the same for both fits.

        Args:
            rotations: Float64 rotation matrices with shape `[n, 3, 3]`.

        Returns:
            Float64 rotation matrices with shape `[n, 3, 3]` in the target frame.
        """
        return np.einsum("ij,njk->nik", self.target_R_source, rotations)

    def apply_pose(self, world_T_body: pycolmap.Rigid3d) -> pycolmap.Rigid3d:
        """Map a whole source-frame pose into the target frame.

        pycolmap is imported inside the call, and only here: it is the one thing in
        this module that needs it, and the `demo` package runs where it is absent.

        Args:
            world_T_body: A pose in the source world frame.

        Returns:
            The same pose expressed in the target world frame: the rotation
            composed with the fit's, the translation through `apply`.
        """
        from colsfm.geometry import rigid3d_from_matrix

        world_R_body: Matrix3 = np.asarray(world_T_body.rotation.matrix(), dtype=np.float64)
        world_t_body: Positions = np.asarray(world_T_body.translation, dtype=np.float64).reshape(1, 3)
        return rigid3d_from_matrix(self.target_R_source @ world_R_body, self.apply(world_t_body).reshape(3))


def align_rigid(source_xyz: Positions, target_xyz: Positions, *, estimate_scale: bool = False) -> RigidAlignment:
    """Fit `target ~ s * R @ source + t`, with `s` held at 1.0 unless asked otherwise.

    The benchmark holds the scale at 1.0 so that a reconstruction which shrank the
    trajectory shows up as alignment error rather than disappearing into the fit;
    the scale the same fit *would* have chosen comes back as `would_be_scale`
    either way. `estimate_scale=True` is the SIM(3) fit the audits want when the
    question is "is the *shape* right", with the residuals measured after the
    scaling; it changes nothing about the default path.

    Args:
        source_xyz: Float64 positions to move, shape `[n, 3]`.
        target_xyz: Float64 positions to move onto, shape `[n, 3]`.
        estimate_scale: Let the fit absorb a scale change instead of holding it at 1.0.

    A trajectory that does not move (stationary, or rotating in place; source variance at most
    `DEGENERATE_VARIANCE`) is still scored: the fit is rigid with scale 1.0 (simplecv's `degenerate="unit_scale"`
    policy), and `would_be_scale` reports 1.0.

    Returns:
        The fitted transform plus its residual statistics.

    Raises:
        ValueError: When the two trajectories differ in length or hold fewer than three samples
            (`simplecv.ops.umeyama.umeyama_alignment`, which does the fit).
    """
    if source_xyz.shape != target_xyz.shape:
        raise ValueError(f"alignment needs matched trajectories, got {source_xyz.shape} and {target_xyz.shape}")
    # The rotation does not depend on the scale, so the similarity fit gives `would_be_scale` for both kinds.
    similarity: SimilarityTransform = umeyama_alignment(source_xyz, target_xyz, allow_scaling=True, eps=DEGENERATE_VARIANCE, degenerate="unit_scale")
    fit: SimilarityTransform = (
        similarity if estimate_scale else umeyama_alignment(source_xyz, target_xyz, eps=DEGENERATE_VARIANCE, degenerate="unit_scale")
    )
    distances: Float64[ndarray, "n"] = np.linalg.norm(fit.apply(source_xyz) - target_xyz, axis=1)
    return RigidAlignment(
        target_R_source=fit.dst_R_src,
        target_t_source=fit.dst_t_src,
        rmse_meters=float(np.sqrt((distances**2).mean())),
        max_error_meters=float(distances.max()),
        would_be_scale=similarity.scale,
        scale=fit.scale,
    )


def path_length_meters(positions: Positions) -> float:
    """Total distance travelled along a trajectory.

    Args:
        positions: Float64 positions with shape `[n, 3]`.

    Returns:
        The summed segment length in metres; 0.0 for fewer than two samples.
    """
    if len(positions) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
