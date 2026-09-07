"""Trajectory IO and the rigid-aligned ATE, driven by hypothesis where the property is exact."""

from pathlib import Path

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from jaxtyping import Float64, Int64
from numpy import ndarray

from slam_rs.reference import ReferenceManifest, load_manifest
from slam_rs.trajectory import (
    ASSOCIATION_TOLERANCE_NS,
    AteResult,
    Trajectory,
    associate,
    ate,
    coverage,
    passes_gate,
    read_trajectory,
    write_trajectory,
)

POSITIONS = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=6, max_value=40).map(lambda n: (n, 3)),
    elements=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False, width=64),
)
"""Position clouds big enough that Umeyama is determined and small enough to stay fast."""


def _quaternion_from(values: Float64[ndarray, " 4"]) -> Float64[ndarray, " 4"]:
    """Normalise four numbers into a unit quaternion, defaulting to the identity when degenerate."""
    norm: float = float(np.linalg.norm(values))
    return np.array([1.0, 0.0, 0.0, 0.0]) if norm < 1e-6 else values / norm


def _rotation_from(quaternion_wxyz: Float64[ndarray, " 4"]) -> Float64[ndarray, "3 3"]:
    """Rotation matrix of a w-first unit quaternion."""
    w, x, y, z = quaternion_wxyz.tolist()
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _trajectory(t_ns: Int64[ndarray, " n"], position_m: Float64[ndarray, "n 3"]) -> Trajectory:
    """A trajectory with identity rotations, for tests that only care about position."""
    quaternion_wxyz: Float64[ndarray, "n 4"] = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (position_m.shape[0], 1))
    return Trajectory(t_ns=t_ns, position_m=position_m, quaternion_wxyz=quaternion_wxyz)


@settings(deadline=None, max_examples=50)
@given(
    positions=POSITIONS,
    quaternion=arrays(dtype=np.float64, shape=(4,), elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, width=64)),
    translation=arrays(dtype=np.float64, shape=(3,), elements=st.floats(min_value=-20.0, max_value=20.0, allow_nan=False, width=64)),
)
def test_a_rigidly_transformed_trajectory_has_zero_ate(
    positions: Float64[ndarray, "n 3"], quaternion: Float64[ndarray, " 4"], translation: Float64[ndarray, " 3"]
) -> None:
    """Umeyama undoes any rigid transform exactly, so the aligned error is numerical noise."""
    # A cloud with no spread leaves the alignment undetermined, which the helper
    # rejects by design rather than returning an arbitrary rotation.
    assume(float(np.var(positions, axis=0).sum()) > 1e-3)
    rotation: Float64[ndarray, "3 3"] = _rotation_from(_quaternion_from(quaternion))
    t_ns: Int64[ndarray, " n"] = np.arange(positions.shape[0], dtype=np.int64) * 20_000_000
    moved: Float64[ndarray, "n 3"] = np.einsum("ij,nj->ni", rotation, positions) + translation
    result: AteResult = ate(_trajectory(t_ns, positions), _trajectory(t_ns, moved))
    assert result.n_associated == positions.shape[0]
    assert result.rmse_m < 1e-6
    assert result.max_m < 1e-6
    assert result.count_delta == 0.0


@settings(deadline=None, max_examples=50)
@given(positions=POSITIONS, offset_ns=st.integers(min_value=-10_000_000, max_value=10_000_000))
def test_association_matches_exactly_up_to_the_tolerance(positions: Float64[ndarray, "n 3"], offset_ns: int) -> None:
    """Every pose associates when the shift is within the tolerance, and none when it is beyond."""
    t_ns: Int64[ndarray, " n"] = np.arange(positions.shape[0], dtype=np.int64) * 1_000_000_000
    reference: Trajectory = _trajectory(t_ns, positions)
    candidate: Trajectory = _trajectory(t_ns + offset_ns, positions)
    expected: int = positions.shape[0] if abs(offset_ns) <= ASSOCIATION_TOLERANCE_NS else 0
    assert associate(reference, candidate).count == expected


def test_the_association_tolerance_is_inclusive_on_both_edges() -> None:
    positions: Float64[ndarray, "3 3"] = np.zeros((3, 3), dtype=np.float64)
    t_ns: Int64[ndarray, " 3"] = np.array([0, 1_000_000_000, 2_000_000_000], dtype=np.int64)
    reference: Trajectory = _trajectory(t_ns, positions)
    for offset_ns, expected in ((ASSOCIATION_TOLERANCE_NS, 3), (ASSOCIATION_TOLERANCE_NS + 1, 0), (-ASSOCIATION_TOLERANCE_NS, 3)):
        assert associate(reference, _trajectory(t_ns + offset_ns, positions)).count == expected


@settings(deadline=None, max_examples=50)
@given(
    positions=POSITIONS,
    start_ns=st.integers(min_value=0, max_value=2**52),
    quaternion=arrays(dtype=np.float64, shape=(4,), elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, width=64)),
)
def test_the_csv_round_trip_preserves_nanoseconds_and_values(
    positions: Float64[ndarray, "n 3"], start_ns: int, quaternion: Float64[ndarray, " 4"], tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Timestamps survive as int64 and the floats survive bit for bit."""
    count: int = positions.shape[0]
    t_ns: Int64[ndarray, " n"] = start_ns + np.arange(count, dtype=np.int64) * 18_518_519
    quaternion_wxyz: Float64[ndarray, "n 4"] = np.tile(_quaternion_from(quaternion), (count, 1))
    written: Trajectory = Trajectory(t_ns=t_ns, position_m=positions, quaternion_wxyz=quaternion_wxyz)
    path: Path = tmp_path_factory.mktemp("trajectory") / "run.csv"
    write_trajectory(path, written)
    read_back: Trajectory = read_trajectory(path)
    assert read_back.t_ns.dtype == np.int64
    np.testing.assert_array_equal(read_back.t_ns, t_ns)
    np.testing.assert_array_equal(read_back.position_m, positions)
    np.testing.assert_array_equal(read_back.quaternion_wxyz, quaternion_wxyz)


def test_a_row_with_the_wrong_field_count_is_rejected(tmp_path: Path) -> None:
    path: Path = tmp_path / "short.csv"
    path.write_text("#timestamp [ns], p_x, p_y, p_z, q_w, q_x, q_y, q_z\n1,0.0,0.0,0.0\n")
    with pytest.raises(ValueError, match="has 4 fields"):
        read_trajectory(path)


def test_coverage_is_the_overlapping_fraction_of_the_reference_span() -> None:
    positions: Float64[ndarray, "11 3"] = np.zeros((11, 3), dtype=np.float64)
    t_ns: Int64[ndarray, " 11"] = np.arange(11, dtype=np.int64) * 1_000_000_000
    reference: Trajectory = _trajectory(t_ns, positions)
    assert coverage(reference, reference) == pytest.approx(1.0)
    assert coverage(reference, _trajectory(t_ns[:6], positions[:6])) == pytest.approx(0.5)
    assert coverage(reference, _trajectory(t_ns[5:], positions[5:])) == pytest.approx(0.5)


def test_the_gate_needs_ten_associations_and_a_two_percent_count_match() -> None:
    angle: Float64[ndarray, " 40"] = np.linspace(0.0, 4.0, 40)
    positions: Float64[ndarray, "40 3"] = np.column_stack([np.cos(angle), np.sin(angle), angle * 0.3])
    t_ns: Int64[ndarray, " 40"] = np.arange(40, dtype=np.int64) * 20_000_000
    reference: Trajectory = _trajectory(t_ns, positions)
    assert passes_gate(ate(reference, reference), tolerance_m=0.02)
    # Enough associations and no count delta, but a scale error the rigid alignment cannot absorb.
    assert not passes_gate(ate(reference, _trajectory(t_ns, positions * 1.05)), tolerance_m=0.02)
    # A tight fit over 12 poses, rejected on the count delta alone.
    assert not passes_gate(ate(reference, _trajectory(t_ns[:12], positions[:12])), tolerance_m=0.02)
    # Fewer than ten associations is not a comparison at all.
    assert not passes_gate(ate(reference, _trajectory(t_ns[:9], positions[:9])), tolerance_m=0.02)


def test_it_reproduces_the_forks_robocap_gate_numbers() -> None:
    """The checked-in basalt outputs must still give 0.13 cm over 1,588 associated poses."""
    manifest: ReferenceManifest = load_manifest()
    golden: Trajectory = read_trajectory(manifest.package_root / manifest.robocap.fixtures.golden)
    candidate: Trajectory = read_trajectory(manifest.package_root / manifest.robocap.fixtures.candidate)
    result: AteResult = ate(golden, candidate)
    assert result.n_associated == manifest.robocap.fixtures.expected_associated
    assert result.n_reference == manifest.robocap.basalt_num_poses
    assert result.n_candidate == manifest.robocap.basalt_num_poses
    assert result.rmse_m * 100 == pytest.approx(manifest.robocap.fixtures.expected_ate_rmse_cm, abs=0.005)
    assert result.count_delta == 0.0
    assert passes_gate(result, tolerance_m=0.05)
    assert coverage(golden, candidate) == pytest.approx(1.0)
