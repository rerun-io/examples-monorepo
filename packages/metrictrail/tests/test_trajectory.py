"""Unit tests for the pure trajectory math: association rules and aligned ATE on synthetic trajectories."""

import numpy as np
import pytest
from jaxtyping import Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation

from metrictrail.eval.trajectory import AteResult, MatchedIndices, aligned_ate, evo_matching_indices, nearest_within


def random_path(rng: np.random.Generator, count: int) -> Float64[ndarray, "n 3"]:
    """A smooth-ish random walk of camera centres, metres."""
    steps: Float64[ndarray, "n 3"] = rng.normal(scale=0.05, size=(count, 3))
    return np.cumsum(steps, axis=0)


def similarity(rng: np.random.Generator, scale: float) -> Float64[ndarray, "4 4"]:
    """A random similarity transform with the given scale."""
    transform: Float64[ndarray, "4 4"] = np.eye(4)
    transform[:3, :3] = scale * Rotation.random(random_state=int(rng.integers(1 << 31))).as_matrix()
    transform[:3, 3] = rng.normal(scale=3.0, size=3)
    return transform


def apply(transform: Float64[ndarray, "4 4"], points: Float64[ndarray, "n 3"]) -> Float64[ndarray, "n 3"]:
    """Apply a homogeneous transform to points."""
    return points @ transform[:3, :3].T + transform[:3, 3]


def zero_moment_noise(rng: np.random.Generator, gt: Float64[ndarray, "n 3"], rms_m: float) -> Float64[ndarray, "n 3"]:
    """Noise with zero mean and zero cross-moment with the GT positions, scaled to an exact RMS norm.

    Every noise column is orthogonal to ``[1, gx, gy, gz]``, so the optimal rotation stays the identity and the
    expected ATE has a closed form (see the tests that use it).
    """
    basis: Float64[ndarray, "n 4"] = np.column_stack([np.ones(gt.shape[0]), gt])
    noise: Float64[ndarray, "n 3"] = rng.normal(size=gt.shape)
    projection: Float64[ndarray, "n 3"] = basis @ np.linalg.lstsq(basis, noise, rcond=None)[0]
    noise = noise - projection
    return noise * (rms_m / np.sqrt(np.mean(np.sum(noise**2, axis=1))))


@pytest.mark.parametrize("scale", [0.37, 1.0, 4.2])
def test_known_sim3_gives_zero_ate_and_recovers_the_scale(scale: float) -> None:
    rng: np.random.Generator = np.random.default_rng(0)
    gt: Float64[ndarray, "n 3"] = random_path(rng, 200)
    est: Float64[ndarray, "n 3"] = apply(similarity(rng, scale), gt)
    result: AteResult = aligned_ate(est, gt, "sim3")
    assert result.rmse_m == pytest.approx(0.0, abs=1e-9)
    assert result.scale == pytest.approx(1.0 / scale, rel=1e-9)
    assert result.num_pairs == 200


def test_known_se3_gives_zero_ate_without_scale() -> None:
    rng: np.random.Generator = np.random.default_rng(1)
    gt: Float64[ndarray, "n 3"] = random_path(rng, 150)
    est: Float64[ndarray, "n 3"] = apply(similarity(rng, 1.0), gt)
    result: AteResult = aligned_ate(est, gt, "se3")
    assert result.rmse_m == pytest.approx(0.0, abs=1e-9)
    assert result.scale == 1.0


def test_se3_mode_does_not_hide_a_scale_error() -> None:
    rng: np.random.Generator = np.random.default_rng(2)
    gt: Float64[ndarray, "n 3"] = random_path(rng, 150)
    est: Float64[ndarray, "n 3"] = apply(similarity(rng, 1.5), gt)
    assert aligned_ate(est, gt, "se3").rmse_m > 0.01
    assert aligned_ate(est, gt, "sim3").rmse_m == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("rms_m", [0.01, 0.05, 0.3])
def test_known_noise_gives_the_closed_form_ate(rms_m: float) -> None:
    """With zero-moment noise e on GT g (G = sum |g_c|^2, E = sum |e|^2, n points):
    SE(3) ATE = sqrt(E / n); Sim(3) ATE = sqrt(G E / (n (G + E))) (the scale shrinks to G / (G + E)).
    A similarity applied to the estimate first must not change either number's alignment-invariant part."""
    rng: np.random.Generator = np.random.default_rng(3)
    gt: Float64[ndarray, "n 3"] = random_path(rng, 300)
    noise: Float64[ndarray, "n 3"] = zero_moment_noise(rng, gt, rms_m)
    count: int = gt.shape[0]
    g: float = float(np.sum((gt - gt.mean(axis=0)) ** 2))
    e: float = float(np.sum(noise**2))
    rigid: Float64[ndarray, "4 4"] = similarity(rng, 1.0)
    se3: AteResult = aligned_ate(apply(rigid, gt + noise), gt, "se3")
    assert se3.rmse_m == pytest.approx(np.sqrt(e / count), rel=1e-9)
    scaled: Float64[ndarray, "4 4"] = similarity(rng, 2.5)
    sim3: AteResult = aligned_ate(apply(scaled, gt + noise), gt, "sim3")
    assert sim3.rmse_m == pytest.approx(np.sqrt(g * e / (count * (g + e))), rel=1e-9)
    assert sim3.scale == pytest.approx(g / (g + e) / 2.5, rel=1e-9)


def test_aligned_ate_refuses_non_finite_positions() -> None:
    gt: Float64[ndarray, "n 3"] = random_path(np.random.default_rng(4), 10)
    est: Float64[ndarray, "n 3"] = gt.copy()
    est[3, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        aligned_ate(est, gt, "sim3")


def test_aligned_ate_matches_evo_ape() -> None:
    """Same pairs, same metric as evo (the backend of VSLAM-LAB and of the paper's evaluator)."""
    evo_sync = pytest.importorskip("evo.core.trajectory", reason="evo cross-check needs evo")
    from evo.core import metrics  # pyrefly: ignore[missing-import]  # evo is not in the metrictrail env; importorskip above

    rng: np.random.Generator = np.random.default_rng(5)
    gt: Float64[ndarray, "n 3"] = random_path(rng, 250)
    est: Float64[ndarray, "n 3"] = apply(similarity(rng, 0.8), gt + rng.normal(scale=0.03, size=gt.shape))
    stamps: Float64[ndarray, "n"] = np.arange(gt.shape[0], dtype=np.float64)
    quats: Float64[ndarray, "n 4"] = np.tile([1.0, 0.0, 0.0, 0.0], (gt.shape[0], 1))
    ref = evo_sync.PoseTrajectory3D(positions_xyz=gt, orientations_quat_wxyz=quats, timestamps=stamps)
    for correct_scale, mode in ((True, "sim3"), (False, "se3")):
        traj = evo_sync.PoseTrajectory3D(positions_xyz=est.copy(), orientations_quat_wxyz=quats, timestamps=stamps)
        traj.align(ref, correct_scale=correct_scale)
        ape = metrics.APE(metrics.PoseRelation.translation_part)
        ape.process_data((ref, traj))
        assert aligned_ate(est, gt, mode).rmse_m == pytest.approx(ape.get_statistic(metrics.StatisticsType.rmse), rel=1e-9)


def test_nearest_within_tolerance_ties_and_edges() -> None:
    reference: Int64[ndarray, "r"] = np.asarray([100, 200, 300, 500], dtype=np.int64)
    query: Int64[ndarray, "q"] = np.asarray([0, 95, 150, 250, 260, 400, 520, 600], dtype=np.int64)
    got: Int64[ndarray, "q"] = nearest_within(query, reference, 20)
    # 150 and 250 are ties (50 either side) but beyond 20; 400 is 100 from both neighbours.
    assert got.tolist() == [-1, 0, -1, -1, -1, -1, 3, -1]
    wide: Int64[ndarray, "q"] = nearest_within(query, reference, 50)
    # Ties go to the earlier stamp (np.argmin), the boundary is inclusive, several queries may share a reference.
    assert wide.tolist() == [-1, 0, 0, 1, 2, -1, 3, -1]


def test_nearest_within_refuses_unsorted_reference() -> None:
    with pytest.raises(ValueError, match="increasing"):
        nearest_within(np.asarray([1], dtype=np.int64), np.asarray([3, 2], dtype=np.int64), 5)


def test_nearest_within_matches_brute_force_argmin() -> None:
    rng: np.random.Generator = np.random.default_rng(6)
    reference: Int64[ndarray, "r"] = np.unique(rng.integers(0, 10_000, size=300)).astype(np.int64)
    query: Int64[ndarray, "q"] = rng.integers(-500, 10_500, size=500).astype(np.int64)
    got: Int64[ndarray, "q"] = nearest_within(query, reference, 25)
    for value, index in zip(query.tolist(), got.tolist(), strict=True):
        diffs: Int64[ndarray, "r"] = np.abs(reference - value)
        best: int = int(np.argmin(diffs))
        assert index == (best if diffs[best] <= 25 else -1)


@pytest.mark.parametrize("est_count", [40, 120, 300])
def test_evo_matching_indices_equals_evo_associate(est_count: int) -> None:
    """Shorter list walked, many-to-one allowed; equal to evo's own associate_trajectories in every length order."""
    pytest.importorskip("evo.core.trajectory", reason="evo cross-check needs evo")
    from evo.core import sync  # pyrefly: ignore[missing-import]
    from evo.core.trajectory import PoseTrajectory3D  # pyrefly: ignore[missing-import]

    rng: np.random.Generator = np.random.default_rng(est_count)
    gt_stamps: Float64[ndarray, "g"] = np.sort(rng.uniform(0.0, 10.0, size=120))
    est_stamps: Float64[ndarray, "e"] = np.sort(rng.uniform(-0.5, 10.5, size=est_count))
    matched: MatchedIndices = evo_matching_indices(gt_stamps, est_stamps, 0.05)

    def trajectory(stamps: Float64[ndarray, "k"]) -> PoseTrajectory3D:
        xyz: Float64[ndarray, "k 3"] = np.column_stack([stamps, stamps * 2.0, stamps * 3.0])
        return PoseTrajectory3D(positions_xyz=xyz, orientations_quat_wxyz=np.tile([1.0, 0, 0, 0], (stamps.shape[0], 1)), timestamps=stamps)

    ref_sync, est_sync = sync.associate_trajectories(trajectory(gt_stamps), trajectory(est_stamps), max_diff=0.05)
    np.testing.assert_array_equal(gt_stamps[matched.gt], ref_sync.timestamps)
    np.testing.assert_array_equal(est_stamps[matched.est], est_sync.timestamps)
