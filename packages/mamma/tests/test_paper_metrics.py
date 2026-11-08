"""Paper-protocol metrics: zero on identity, exact on a known offset, PA invariant to similarity transforms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from jaxtyping import Float32, Float64
from numpy import ndarray
from scipy.spatial.transform import Rotation

from mamma.eval.mosh_gt import VARIANT_FILES, MoshSequence, PosedBody, load_mosh_gt, pose_mosh_person
from mamma.eval.paper_metrics import SequenceMetrics, aggregate, per_frame_error_mm, procrustes_align, sequence_metrics


def _random_body(rng: np.random.Generator, frames: int) -> tuple[Float32[ndarray, "f 55 3"], Float32[ndarray, "f 10475 3"]]:
    joints: Float32[ndarray, "f 55 3"] = rng.normal(scale=0.4, size=(frames, 55, 3)).astype(np.float32)
    vertices: Float32[ndarray, "f 10475 3"] = rng.normal(scale=0.4, size=(frames, 10475, 3)).astype(np.float32)
    return joints, vertices


def test_gt_vs_gt_is_zero() -> None:
    rng: np.random.Generator = np.random.default_rng(0)
    body: tuple[Float32[ndarray, "f 55 3"], Float32[ndarray, "f 10475 3"]] = _random_body(rng, 7)
    frames = np.arange(7, dtype=np.int64)
    metrics: SequenceMetrics = sequence_metrics("gt", frames, body[0], body[1], body[0], body[1])
    assert np.all(metrics.mpjpe_mm == 0.0)
    assert np.all(metrics.pve_mm == 0.0)
    assert metrics.mean_pa_mpjpe_mm < 1e-6


def test_translation_offset_gives_exact_mpjpe_and_pve() -> None:
    rng: np.random.Generator = np.random.default_rng(1)
    body: tuple[Float32[ndarray, "f 55 3"], Float32[ndarray, "f 10475 3"]] = _random_body(rng, 5)
    offset_m: Float32[ndarray, "3"] = np.array([0.003, -0.004, 0.012], dtype=np.float32)  # |offset| = 13 mm
    frames = np.arange(5, dtype=np.int64)
    metrics: SequenceMetrics = sequence_metrics("shifted", frames, body[0] + offset_m, body[1] + offset_m, body[0], body[1])
    np.testing.assert_allclose(metrics.mpjpe_mm, 13.0, atol=1e-3)
    np.testing.assert_allclose(metrics.pve_mm, 13.0, atol=1e-3)
    # No alignment in MPJPE/PVE, but PA removes a pure translation.
    assert metrics.mean_pa_mpjpe_mm < 1e-3


def test_pa_removes_similarity_transform_but_mpjpe_does_not() -> None:
    rng: np.random.Generator = np.random.default_rng(2)
    joints: Float32[ndarray, "f 55 3"] = _random_body(rng, 4)[0]
    rot: Float64[ndarray, "4 3 3"] = Rotation.random(4, random_state=3).as_matrix()
    moved: Float32[ndarray, "f 55 3"] = (1.1 * np.einsum("fij,fnj->fni", rot, joints) + np.array([0.5, -1.0, 0.2])).astype(np.float32)
    aligned: Float64[ndarray, "f 55 3"] = procrustes_align(moved, joints)
    np.testing.assert_allclose(aligned, joints, atol=1e-5)
    assert per_frame_error_mm(aligned.astype(np.float32), joints).max() < 1e-2
    assert per_frame_error_mm(moved, joints).min() > 100.0


def test_pa_does_not_reflect() -> None:
    rng: np.random.Generator = np.random.default_rng(4)
    joints: Float32[ndarray, "f 55 3"] = _random_body(rng, 3)[0]
    mirrored: Float32[ndarray, "f 55 3"] = joints * np.array([-1.0, 1.0, 1.0], dtype=np.float32)
    aligned: Float64[ndarray, "f 55 3"] = procrustes_align(mirrored, joints)
    # A proper rotation cannot undo a mirror, so the residual stays well above zero.
    assert per_frame_error_mm(aligned.astype(np.float32), joints).min() > 10.0


def test_aggregate_pooled_vs_sequence_mean() -> None:
    short = SequenceMetrics("a", np.arange(1, dtype=np.int64), np.array([10.0]), np.array([20.0]), np.array([1.0]))
    long = SequenceMetrics("b", np.arange(3, dtype=np.int64), np.full(3, 30.0), np.full(3, 40.0), np.full(3, 3.0))
    agg = aggregate([short, long])
    assert agg.num_frames == 4 and agg.num_sequences == 2
    assert agg.pooled_mpjpe_mm == pytest.approx((10.0 + 90.0) / 4)
    assert agg.seqmean_mpjpe_mm == pytest.approx(20.0)
    assert agg.pooled_pve_mm == pytest.approx((20.0 + 120.0) / 4)
    assert agg.seqmean_pve_mm == pytest.approx(30.0)
    assert agg.seqmean_pa_mpjpe_mm == pytest.approx(2.0)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        per_frame_error_mm(np.zeros((2, 55, 3), np.float32), np.zeros((2, 54, 3), np.float32))


_SINGLES_GT: Path = Path("/mnt/nas/datasets/mamma/mamma_eval_singles/230929_WhiteRabbit_CatchBall_50048_1/gt/global.npz")
_BODY_MODELS: Path = Path(__file__).parent.parent / "data" / "body_models"


@pytest.mark.golden
@pytest.mark.skipif(
    not (_SINGLES_GT.is_file() and (_BODY_MODELS / VARIANT_FILES["v1_1_male"]).is_file()), reason="MammaEval GT or v1.1 models absent"
)
def test_mosh_gt_posed_vs_itself_is_zero() -> None:
    gt: MoshSequence = load_mosh_gt(_SINGLES_GT)
    assert gt.method == "mosh_v_template" and gt.people[0].variant == "v1_1_male"
    frames = np.arange(0, 12, dtype=np.int64)
    body: PosedBody = pose_mosh_person(gt.people[0], frames, _BODY_MODELS)
    assert body.joints.shape == (12, 55, 3) and body.vertices.shape == (12, 10475, 3)
    metrics: SequenceMetrics = sequence_metrics("gt", frames, body.joints, body.vertices, body.joints, body.vertices)
    assert metrics.mean_mpjpe_mm == 0.0 and metrics.mean_pve_mm == 0.0
    # z-up world, metres: the pelvis of a standing adult sits roughly 0.8-1.2 m above the floor.
    assert 0.6 < float(body.joints[:, 0, 2].mean()) < 1.3
