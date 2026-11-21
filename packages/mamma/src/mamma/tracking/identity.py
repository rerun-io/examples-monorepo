"""Forward-only person identity: CLIP feature banks + epipolar cross-camera scoring.

Ported from the original ``segmentation/core/pipeline.py`` with the anti-causal
parts removed: the bank is append-only with FIFO eviction (no whole-sequence
gallery passes), and cross-camera identity transfer uses single-frame epipolar
geometry exactly as the original bootstrap did.

Original defaults preserved: ViT-B-32 @ laion2b_s34b_b79k, bank size 64,
near-duplicate skip at cosine > 0.998, combined score = 0.35*CLIP + 0.65*epipolar,
sigma_px = max(24, 0.055*diag), max_dist_px = max(2*sigma, 0.12*diag).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float32, Float64, UInt8
from numpy import ndarray

CLIP_WEIGHT: float = 0.35
"""Weight of the CLIP similarity term in the combined cross-camera score."""
EPIPOLAR_WEIGHT: float = 0.65
"""Weight of the epipolar term in the combined cross-camera score."""
BANK_MAX_SIZE: int = 64
"""Maximum CLIP features kept per person (FIFO eviction)."""
DUPLICATE_COSINE: float = 0.998
"""Features more similar than this to an existing bank entry are skipped."""


class ClipEncoder:
    """open-clip ViT-B-32 (laion2b_s34b_b79k) image encoder for person crops."""

    def __init__(self, device: str = "cuda") -> None:
        from typing import Any

        import open_clip
        from PIL import Image

        self._image_cls = Image
        # open_clip's factory has loose stubs; keep the handles dynamically typed.
        created: Any = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        self.model: Any = created[0].to(device).eval()
        self.preprocess: Any = created[2]
        self.device: str = device

    def encode(self, crops: list[UInt8[ndarray, "ch cw 3"]]) -> Float32[torch.Tensor, "n 512"]:
        """Encode RGB crops to CLIP features (CPU float32, unnormalized)."""
        if not crops:
            return torch.zeros((0, 512), dtype=torch.float32)
        batch: Float32[torch.Tensor, "n 3 224 224"] = torch.stack(
            [self.preprocess(self._image_cls.fromarray(crop)) for crop in crops], dim=0
        ).to(self.device)
        with torch.inference_mode():
            feats: torch.Tensor = self.model.encode_image(batch)
        return feats.detach().cpu().float()


class FeatureBank:
    """Per-person rolling CLIP galleries with duplicate suppression."""

    def __init__(self, max_size: int = BANK_MAX_SIZE) -> None:
        self.max_size: int = max_size
        self._bank: dict[int, list[Float32[torch.Tensor, "512"]]] = {}

    @property
    def obj_ids(self) -> list[int]:
        """Known person ids."""
        return sorted(self._bank.keys())

    def append(self, obj_id: int, feature: Float32[torch.Tensor, "512"]) -> bool:
        """Add a feature unless it nearly duplicates an existing bank entry."""
        feature = feature.detach().float().reshape(-1)
        gallery: list[Float32[torch.Tensor, "512"]] = self._bank.setdefault(obj_id, [])
        if gallery:
            stacked: Float32[torch.Tensor, "g 512"] = torch.stack(gallery, dim=0)
            sims: Float32[torch.Tensor, "g"] = F.cosine_similarity(stacked, feature.unsqueeze(0), dim=1)
            if float(sims.max()) > DUPLICATE_COSINE:
                return False
        gallery.append(feature)
        if len(gallery) > self.max_size:
            gallery.pop(0)
        return True

    def similarity(self, det_feats: Float32[torch.Tensor, "n 512"]) -> Float32[ndarray, "k n"]:
        """1-NN cosine similarity of each detection against each person's gallery.

        Rows follow :attr:`obj_ids` order.
        """
        obj_ids: list[int] = self.obj_ids
        scores: Float32[ndarray, "k n"] = np.zeros((len(obj_ids), det_feats.shape[0]), dtype=np.float32)
        if det_feats.shape[0] == 0:
            return scores
        det_norm: Float32[torch.Tensor, "n 512"] = F.normalize(det_feats, dim=1)
        for row, obj_id in enumerate(obj_ids):
            gallery: Float32[torch.Tensor, "g 512"] = torch.stack(self._bank[obj_id], dim=0)
            gallery_norm: Float32[torch.Tensor, "g 512"] = F.normalize(gallery, dim=1)
            sim: Float32[torch.Tensor, "n g"] = det_norm @ gallery_norm.T
            scores[row] = sim.max(dim=1).values.numpy()
        return scores


def fundamental_matrix(
    k_source: Float64[ndarray, "3 3"],
    world_to_source: Float64[ndarray, "4 4"],
    k_target: Float64[ndarray, "3 3"],
    world_to_target: Float64[ndarray, "4 4"],
) -> Float64[ndarray, "3 3"] | None:
    """F mapping source-image points to epipolar lines in the target image."""
    r1: Float64[ndarray, "3 3"] = world_to_source[:3, :3]
    t1: Float64[ndarray, "3"] = world_to_source[:3, 3]
    r2: Float64[ndarray, "3 3"] = world_to_target[:3, :3]
    t2: Float64[ndarray, "3"] = world_to_target[:3, 3]
    r_rel: Float64[ndarray, "3 3"] = r2 @ r1.T
    t_rel: Float64[ndarray, "3"] = t2 - r_rel @ t1
    t_skew: Float64[ndarray, "3 3"] = np.array(
        [
            [0.0, -t_rel[2], t_rel[1]],
            [t_rel[2], 0.0, -t_rel[0]],
            [-t_rel[1], t_rel[0], 0.0],
        ]
    )
    essential: Float64[ndarray, "3 3"] = t_skew @ r_rel
    try:
        f_matrix: Float64[ndarray, "3 3"] = np.linalg.inv(k_target).T @ essential @ np.linalg.inv(k_source)
    except np.linalg.LinAlgError:
        return None
    norm: float = float(np.abs(f_matrix).max())
    if not np.isfinite(norm) or norm <= 0.0:
        return None
    return f_matrix / norm


def epipolar_score(
    f_matrix: Float64[ndarray, "3 3"],
    x_source: Float32[ndarray, "3"] | Float64[ndarray, "3"],
    x_target: Float32[ndarray, "3"] | Float64[ndarray, "3"],
    sigma_px: float,
    max_dist_px: float,
) -> float:
    """Gaussian score of the target point's distance to the source epipolar line."""
    line: Float64[ndarray, "3"] = f_matrix @ np.asarray(x_source, dtype=np.float64)
    denom: float = float(np.hypot(line[0], line[1]))
    if denom <= 1e-12:
        return 0.0
    dist: float = abs(float(line @ np.asarray(x_target, dtype=np.float64))) / denom
    if dist > max_dist_px:
        return 0.0
    sigma: float = max(sigma_px, 1e-3)
    return float(np.exp(-0.5 * (dist / sigma) ** 2))


def resolve_epipolar_px(width: int, height: int) -> tuple[float, float]:
    """``(sigma_px, max_dist_px)`` from the image diagonal (original 'auto' mode)."""
    diag: float = float(np.hypot(width, height))
    sigma_px: float = max(24.0, 0.055 * diag)
    max_dist_px: float = max(2.0 * sigma_px, 0.12 * diag)
    return sigma_px, max_dist_px


def assign_hungarian(score: Float32[ndarray, "k n"], min_score: float) -> dict[int, int]:
    """Optimal row->col assignment maximizing score; drops pairs below ``min_score``.

    Returns a map of row index -> column index.
    """
    from scipy.optimize import linear_sum_assignment

    if score.size == 0:
        return {}
    rows: ndarray
    cols: ndarray
    rows, cols = linear_sum_assignment(1.0 - score)
    return {int(r): int(c) for r, c in zip(rows, cols, strict=True) if score[r, c] >= min_score}


SUPPORT_ELIGIBLE_FRAC: float = 0.9
"""A bootstrap candidate stays eligible for the rig-centre ranking only when its
cross-camera epipolar support is at least this fraction of the best candidate's."""


def order_bootstrap_candidates(
    ref_idx: int,
    ref_centers: Float32[ndarray, "n 3"] | Float64[ndarray, "n 3"],
    ref_scores: Float32[ndarray, "n"],
    centers_per_cam: list[Float32[ndarray, "m 3"]],
    k_matrices: list[Float64[ndarray, "3 3"]],
    world_to_cams: list[Float64[ndarray, "4 4"]],
    sigma_px: float,
    max_dist_px: float,
    min_score: float,
) -> list[int]:
    """Reference-camera detections ordered for bootstrap: best-supported first, then nearest the rig centre.

    Support is the sum, over the other cameras, of the candidate's best epipolar score there: the capture
    subject stands inside the ring and every camera sees it, so its support is near the camera count. A
    person seen well by the reference camera alone (an operator behind a tripod) borrows the subject's
    detection in every other camera as its "best" match; those scores pass ``min_score`` but stay low,
    and the rays they triangulate meet near the subject, so the rig distance alone ranks such a person
    first. Candidates within ``SUPPORT_ELIGIBLE_FRAC`` of the best support are ranked by the XY distance
    of their multi-view position (two-view DLT per supporting camera, median) to the rig centre, as
    before; the rest follow by support. Bystanders that every camera sees well (MammaEval's ball
    thrower) stay eligible and the rig distance separates them.
    """
    p_ref: Float64[ndarray, "3 4"] = k_matrices[ref_idx] @ world_to_cams[ref_idx][:3, :]
    cam_centers: Float64[ndarray, "c 3"] = np.stack([-w2c[:3, :3].T @ w2c[:3, 3] for w2c in world_to_cams])
    rig_xy: Float64[ndarray, "2"] = cam_centers[:, :2].mean(axis=0)
    supports: list[float] = []
    distances: list[float] = []
    for x_ref in np.asarray(ref_centers, dtype=np.float64):
        support: float = 0.0
        points: list[Float64[ndarray, "3"]] = []
        for cam_idx, centers in enumerate(centers_per_cam):
            if cam_idx == ref_idx or len(centers) == 0:
                continue
            f_matrix: Float64[ndarray, "3 3"] | None = fundamental_matrix(
                k_matrices[ref_idx], world_to_cams[ref_idx], k_matrices[cam_idx], world_to_cams[cam_idx]
            )
            if f_matrix is None:
                continue
            scores: list[float] = [epipolar_score(f_matrix, x_ref, x_t, sigma_px, max_dist_px) for x_t in centers]
            best: int = int(np.argmax(scores))
            if scores[best] < min_score:
                continue
            support += scores[best]
            p_tgt: Float64[ndarray, "3 4"] = k_matrices[cam_idx] @ world_to_cams[cam_idx][:3, :]
            x_t: Float64[ndarray, "3"] = np.asarray(centers[best], dtype=np.float64)
            a: Float64[ndarray, "4 4"] = np.stack(
                [
                    x_ref[0] * p_ref[2] - p_ref[0],
                    x_ref[1] * p_ref[2] - p_ref[1],
                    x_t[0] * p_tgt[2] - p_tgt[0],
                    x_t[1] * p_tgt[2] - p_tgt[1],
                ]
            )
            homog: Float64[ndarray, "4"] = np.linalg.svd(a)[2][-1]
            if abs(homog[3]) > 1e-9:
                points.append(homog[:3] / homog[3])
        supports.append(support)
        distances.append(float(np.linalg.norm(np.median(np.stack(points), axis=0)[:2] - rig_xy)) if points else float("inf"))
    support_arr: Float64[ndarray, "n"] = np.asarray(supports)
    eligible: ndarray = support_arr >= SUPPORT_ELIGIBLE_FRAC * support_arr.max() if support_arr.max() > 0 else np.ones(len(supports), bool)
    # lexsort: last key is primary -> ineligible after eligible; then distance (eligible) / support; then score.
    secondary: Float64[ndarray, "n"] = np.where(eligible, np.asarray(distances), -support_arr)
    return [int(i) for i in np.lexsort((-np.asarray(ref_scores, dtype=np.float64), secondary, ~eligible))]
