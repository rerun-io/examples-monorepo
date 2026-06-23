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
