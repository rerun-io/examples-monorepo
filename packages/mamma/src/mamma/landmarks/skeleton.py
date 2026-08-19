"""Approximate 2D SMPL-X skeleton from the 512 dense surface landmarks.

Each of the 512 landmarks is a fixed SMPL-X surface vertex, so every joint can
be estimated as a fixed weighted average of nearby landmarks. Weights are
k-nearest-neighbor gaussians built in rest pose (joint -> surface distances).
Restricting SMPL-X's own ``J_regressor`` to the 512 sampled vertices was
measured and rejected: most rows keep <5% of their mass (pelvis 0.2%), giving
12.5 cm mean body-joint error on posed bodies vs 3.3 cm for the kNN weights.
No SMPL-X fitting happens here — this is a linear map, exact articulation is
not preserved, and the output is a display skeleton, not a measurement.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float32, Int
from numpy import ndarray

KNN_NEIGHBORS: int = 8
"""Landmarks averaged per joint."""
SIGMA_NEIGHBORS: int = 4
"""Nearest neighbors whose mean distance sets each joint's gaussian sigma."""
DEFAULT_BODY_MODELS_DIR: Path = Path(
    os.environ.get("MAMMA_BODY_MODELS_DIR", str(Path(__file__).resolve().parents[3] / "data" / "body_models"))
).expanduser()
"""Package-relative body-model root populated by ``_download-mamma-body-models``."""


@dataclass(frozen=True, slots=True)
class SkeletonRegressor:
    """Fixed linear map from dense landmarks to 2D SMPL-X joints."""

    weights: Float32[ndarray, "j landmarks"]
    """Per-joint landmark weights, rows sum to 1."""
    parents: tuple[int, ...]
    """SMPL-X kinematic tree; ``parents[0] == -1``."""
    bones: Int[ndarray, "b 2"]
    """Child-parent joint indices for every non-root bone."""


def body_models_available(body_models_dir: Path | None = None) -> bool:
    """Return whether the downloaded SMPL-X model directory is present."""
    resolved_dir: Path = DEFAULT_BODY_MODELS_DIR if body_models_dir is None else body_models_dir
    return (resolved_dir / "smplx").exists()


@functools.lru_cache(maxsize=1)
def load_skeleton_regressor(body_models_dir: Path | None = None) -> SkeletonRegressor:
    """Load the display skeleton regressor from the package's body-model assets.

    Raises:
        RuntimeError: If the body-model download task has not populated the assets.
    """
    resolved_dir: Path = DEFAULT_BODY_MODELS_DIR if body_models_dir is None else body_models_dir
    if not body_models_available(resolved_dir):
        raise RuntimeError("SMPL-X body models not found. Run the pixi task _download-mamma-body-models in packages/mamma.")
    return build_skeleton_regressor(resolved_dir)


def skeleton_strips(
    joints_xy: Float32[ndarray, "j 2"],
    regressor: SkeletonRegressor,
) -> Float32[ndarray, "b 2 2"]:
    """Build child-parent line strips from one person's regressed joints.

    Args:
        joints_xy: Display skeleton joints in image pixels.
        regressor: Regressor carrying the SMPL-X kinematic tree.

    Returns:
        One two-point strip for every non-root joint.
    """
    return joints_xy[regressor.bones]


def build_skeleton_regressor(body_models_dir: Path) -> SkeletonRegressor:
    """Build the kNN joint regressor from the SMPL-X neutral model in rest pose.

    Args:
        body_models_dir: Root containing ``smplx/SMPLX_NEUTRAL.npz`` and
            ``downsampled_verts/verts_512.pkl`` (the ``_download-mamma-body-models``
            pixi task layout).

    Returns:
        The regressor over the joints exposed by the loaded SMPL-X model.
    """
    import pickle

    from mamma.fitting.smplx_wrapper import build_smplx_neutral

    model = build_smplx_neutral(body_models_dir, device="cpu")
    with open(body_models_dir / "downsampled_verts" / "verts_512.pkl", "rb") as f:
        sampling: Float32[torch.Tensor, "n v"] = pickle.load(f).float()

    v_rest: Float32[torch.Tensor, "v 3"] = model.v_template.float()
    landmarks_rest: Float32[torch.Tensor, "n 3"] = sampling @ v_rest
    joints_rest: Float32[torch.Tensor, "j 3"] = model.J_regressor.float() @ v_rest

    distances: Float32[torch.Tensor, "j n"] = torch.cdist(joints_rest, landmarks_rest)
    knn_dist, knn_idx = distances.topk(KNN_NEIGHBORS, largest=False)
    sigma: Float32[torch.Tensor, "j 1"] = knn_dist[:, :SIGMA_NEIGHBORS].mean(dim=1, keepdim=True).clamp_min(1e-4)
    gaussians: Float32[torch.Tensor, "j k"] = torch.exp(-0.5 * (knn_dist / sigma) ** 2)
    weights: Float32[torch.Tensor, "j n"] = torch.zeros(joints_rest.shape[0], landmarks_rest.shape[0])
    weights.scatter_(1, knn_idx, gaussians)
    weights = weights / weights.sum(dim=1, keepdim=True)

    parents: tuple[int, ...] = tuple(int(parent) for parent in model.parents.tolist())
    bones: Int[ndarray, "b 2"] = np.asarray(
        [(joint_idx, parent_idx) for joint_idx, parent_idx in enumerate(parents) if parent_idx >= 0], dtype=np.int64
    )
    return SkeletonRegressor(weights=weights.numpy().astype(np.float32), parents=parents, bones=bones)


def joints_from_landmarks(
    landmarks_xy: Float32[ndarray, "n landmarks 2"],
    regressor: SkeletonRegressor,
) -> Float32[ndarray, "n j 2"]:
    """Estimate 2D joints per person as fixed weighted landmark averages.

    All 512 landmarks participate regardless of visibility: MammaNet predicts
    xy for occluded landmarks too (the fitting pipeline consumes all of them),
    and in 2D projection a back-surface landmark lands at nearly the same pixel
    as a front one — visibility reweighting would only bias joints toward the
    visible side (SMPL-X spine joints sit near the back surface and would
    otherwise lose all their support).

    Args:
        landmarks_xy: Predicted landmark positions in image pixels.
        regressor: Fixed landmark->joint weights.

    Returns:
        2D joints per person.
    """
    joints: Float32[ndarray, "n j 2"] = np.einsum("jl,nlc->njc", regressor.weights, landmarks_xy)
    return joints.astype(np.float32)
