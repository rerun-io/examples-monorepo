"""MammaEval accuracy metrics in the MAMMA paper's protocol (arXiv 2506.13040, Table 4).

- MPJPE: mean Euclidean distance over the 55 SMPL-X kinematic joints, millimetres, world frame, NO alignment.
- PVE: mean Euclidean distance over all 10475 SMPL-X vertices, millimetres, world frame, NO alignment.
- PA-MPJPE (context only, not in the paper's table): MPJPE after a per-frame similarity Procrustes alignment
  (scale, rotation, translation) of the prediction onto the GT joints.

Per-frame arrays are kept so the caller can pick frames; :func:`aggregate` reports both the pooled mean over all
frames of all sequences and the mean of per-sequence means (they differ when sequences have different lengths).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float32, Float64, Int64
from numpy import ndarray


def per_frame_error_mm(pred: Float32[ndarray, "f n 3"], gt: Float32[ndarray, "f n 3"]) -> Float64[ndarray, "f"]:
    """Mean point-to-point distance per frame, millimetres (inputs in metres, no alignment).

    Args:
        pred: Predicted points (joints or vertices), world metres.
        gt: Ground-truth points in the same order, world metres.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"pred {pred.shape} and gt {gt.shape} differ")
    dist_m: Float64[ndarray, "f n"] = np.linalg.norm(pred.astype(np.float64) - gt.astype(np.float64), axis=-1)
    return dist_m.mean(axis=-1) * 1000.0


def procrustes_align(pred: Float32[ndarray, "f n 3"], gt: Float32[ndarray, "f n 3"]) -> Float64[ndarray, "f n 3"]:
    """Per-frame similarity alignment (Umeyama) of ``pred`` onto ``gt``.

    Args:
        pred: Points to align, metres.
        gt: Target points, metres.

    Returns:
        ``s * R @ pred + t`` minimizing the squared distance to ``gt`` per frame.
    """
    x: Float64[ndarray, "f n 3"] = pred.astype(np.float64)
    y: Float64[ndarray, "f n 3"] = gt.astype(np.float64)
    mu_x: Float64[ndarray, "f 1 3"] = x.mean(axis=1, keepdims=True)
    mu_y: Float64[ndarray, "f 1 3"] = y.mean(axis=1, keepdims=True)
    xc: Float64[ndarray, "f n 3"] = x - mu_x
    yc: Float64[ndarray, "f n 3"] = y - mu_y
    cov: Float64[ndarray, "f 3 3"] = np.einsum("fni,fnj->fij", yc, xc)
    u: Float64[ndarray, "f 3 3"]
    sigma: Float64[ndarray, "f 3"]
    vt: Float64[ndarray, "f 3 3"]
    u, sigma, vt = np.linalg.svd(cov)
    # Reflection guard: flip the last axis where det(U V^T) < 0.
    d: Float64[ndarray, "f"] = np.sign(np.linalg.det(u @ vt))
    d[d == 0] = 1.0
    correction: Float64[ndarray, "f 3"] = np.stack([np.ones_like(d), np.ones_like(d), d], axis=-1)
    rot: Float64[ndarray, "f 3 3"] = (u * correction[:, None, :]) @ vt
    var_x: Float64[ndarray, "f"] = (xc**2).sum(axis=(1, 2))
    scale: Float64[ndarray, "f"] = (sigma * correction).sum(axis=-1) / np.maximum(var_x, 1e-12)
    aligned: Float64[ndarray, "f n 3"] = scale[:, None, None] * np.einsum("fij,fnj->fni", rot, xc) + mu_y
    return aligned


@dataclass(frozen=True, slots=True)
class SequenceMetrics:
    """Per-frame paper metrics of one person in one sequence."""

    name: str
    """Sequence (and person) label."""
    frames: Int64[ndarray, "f"]
    """GT rows scored."""
    mpjpe_mm: Float64[ndarray, "f"]
    """Per-frame MPJPE (55 joints, no alignment)."""
    pve_mm: Float64[ndarray, "f"]
    """Per-frame PVE (10475 vertices, no alignment)."""
    pa_mpjpe_mm: Float64[ndarray, "f"]
    """Per-frame PA-MPJPE (similarity-aligned joints)."""

    @property
    def mean_mpjpe_mm(self) -> float:
        """Sequence-mean MPJPE."""
        return float(self.mpjpe_mm.mean())

    @property
    def mean_pve_mm(self) -> float:
        """Sequence-mean PVE."""
        return float(self.pve_mm.mean())

    @property
    def mean_pa_mpjpe_mm(self) -> float:
        """Sequence-mean PA-MPJPE."""
        return float(self.pa_mpjpe_mm.mean())


def sequence_metrics(
    name: str,
    frames: Int64[ndarray, "f"],
    pred_joints: Float32[ndarray, "f 55 3"],
    pred_vertices: Float32[ndarray, "f 10475 3"],
    gt_joints: Float32[ndarray, "f 55 3"],
    gt_vertices: Float32[ndarray, "f 10475 3"],
) -> SequenceMetrics:
    """Score one person's predicted SMPL-X against GT at the same rows (metres in, mm out).

    Args:
        name: Label for reports.
        frames: GT rows of the inputs.
        pred_joints: Predicted 55 kinematic joints, world metres.
        pred_vertices: Predicted vertices, world metres.
        gt_joints: GT 55 kinematic joints, world metres.
        gt_vertices: GT vertices, world metres.
    """
    if len(frames) == 0:
        raise ValueError(f"{name}: no frames to score")
    return SequenceMetrics(
        name=name,
        frames=frames.astype(np.int64),
        mpjpe_mm=per_frame_error_mm(pred_joints, gt_joints),
        pve_mm=per_frame_error_mm(pred_vertices, gt_vertices),
        pa_mpjpe_mm=per_frame_error_mm(procrustes_align(pred_joints, gt_joints).astype(np.float32), gt_joints),
    )


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    """Metrics over several sequences, both aggregation conventions."""

    num_sequences: int
    """Sequences (person-sequences) aggregated."""
    num_frames: int
    """Frames pooled."""
    pooled_mpjpe_mm: float
    """Mean over all frames of all sequences."""
    pooled_pve_mm: float
    """Mean over all frames of all sequences."""
    pooled_pa_mpjpe_mm: float
    """Mean over all frames of all sequences."""
    seqmean_mpjpe_mm: float
    """Mean of per-sequence means."""
    seqmean_pve_mm: float
    """Mean of per-sequence means."""
    seqmean_pa_mpjpe_mm: float
    """Mean of per-sequence means."""


def aggregate(sequences: list[SequenceMetrics]) -> AggregateMetrics:
    """Pool per-frame metrics and average sequence means.

    Args:
        sequences: Per-sequence metrics (at least one).
    """
    if not sequences:
        raise ValueError("nothing to aggregate")
    mpjpe: Float64[ndarray, "n"] = np.concatenate([s.mpjpe_mm for s in sequences])
    pve: Float64[ndarray, "n"] = np.concatenate([s.pve_mm for s in sequences])
    pa: Float64[ndarray, "n"] = np.concatenate([s.pa_mpjpe_mm for s in sequences])
    return AggregateMetrics(
        num_sequences=len(sequences),
        num_frames=int(mpjpe.size),
        pooled_mpjpe_mm=float(mpjpe.mean()),
        pooled_pve_mm=float(pve.mean()),
        pooled_pa_mpjpe_mm=float(pa.mean()),
        seqmean_mpjpe_mm=float(np.mean([s.mean_mpjpe_mm for s in sequences])),
        seqmean_pve_mm=float(np.mean([s.mean_pve_mm for s in sequences])),
        seqmean_pa_mpjpe_mm=float(np.mean([s.mean_pa_mpjpe_mm for s in sequences])),
    )
