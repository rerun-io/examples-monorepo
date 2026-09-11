"""Trajectory IO and the rigid-aligned ATE, driven by hypothesis where the property is exact."""

from pathlib import Path

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from jaxtyping import Float64, Int64
from numpy import ndarray
from simplecv.ops.umeyama import SimilarityTransform, umeyama_alignment

from slam_rs.trajectory import (
    ASSOCIATION_TOLERANCE_NS,
    AteResult,
    Trajectory,
    associate,
    ate,
    coverage,
    empty_trajectory,
    read_trajectory,
    rigid_alignment,
    shift_clock,
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
    """Umeyama undoes any rigid transform exactly, so the aligned error is numerical noise.

    No assumption filters the input: a stationary cloud is a legitimate reference
    run and must align to zero, not raise.
    """
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


def test_a_stationary_trajectory_aligns_to_zero_rather_than_raising() -> None:
    """golden_compare has no variance floor, so a rig that never moved is a pass, not an error."""
    positions: Float64[ndarray, "10 3"] = np.tile(np.array([1.5, -2.0, 0.25]), (10, 1))
    t_ns: Int64[ndarray, " 10"] = np.arange(10, dtype=np.int64) * 20_000_000
    stationary: Trajectory = _trajectory(t_ns, positions)
    result: AteResult = ate(stationary, stationary)
    assert result.n_associated == 10
    assert result.rmse_m == pytest.approx(0.0, abs=1e-12)
    # The shared helper refuses this input; that difference is the reason
    # `rigid_alignment` exists rather than calling it.
    with pytest.raises(ValueError, match="variance too small"):
        umeyama_alignment(positions, positions, allow_scaling=False)


def test_a_micrometre_span_aligns_to_zero_rather_than_raising() -> None:
    """A trajectory spanning 1e-5 m has a variance under the shared helper's 1e-9 floor."""
    span: Float64[ndarray, " 12"] = np.linspace(0.0, 1e-5, 12)
    positions: Float64[ndarray, "12 3"] = np.column_stack([span, span * 0.5, -span])
    t_ns: Int64[ndarray, " 12"] = np.arange(12, dtype=np.int64) * 20_000_000
    tiny: Trajectory = _trajectory(t_ns, positions)
    result: AteResult = ate(tiny, tiny)
    assert result.n_associated == 12
    assert result.rmse_m == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="variance too small"):
        umeyama_alignment(positions, positions, allow_scaling=False)


@settings(deadline=None, max_examples=50)
@given(
    positions=POSITIONS,
    quaternion=arrays(dtype=np.float64, shape=(4,), elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, width=64)),
    translation=arrays(dtype=np.float64, shape=(3,), elements=st.floats(min_value=-20.0, max_value=20.0, allow_nan=False, width=64)),
)
def test_the_ported_alignment_agrees_with_simplecvs_on_well_conditioned_input(
    positions: Float64[ndarray, "n 3"], quaternion: Float64[ndarray, " 4"], translation: Float64[ndarray, " 3"]
) -> None:
    """Where both are defined, the ported arithmetic and the shared helper are the same transform."""
    # The rotation is unique only when the cloud spans all three axes. A collinear
    # or planar set admits a family of equally good alignments, and the two
    # implementations may legitimately return different members of it, so the
    # smallest singular value of the centred cloud gates this comparison.
    centered: Float64[ndarray, "n 3"] = positions - positions.mean(axis=0)
    assume(float(np.linalg.svd(centered, compute_uv=False)[-1]) > 1e-2)
    rotation: Float64[ndarray, "3 3"] = _rotation_from(_quaternion_from(quaternion))
    moved: Float64[ndarray, "n 3"] = np.einsum("ij,nj->ni", rotation, positions) + translation
    ported: SimilarityTransform = rigid_alignment(moved, positions)
    shared: SimilarityTransform = umeyama_alignment(moved, positions, allow_scaling=False)
    np.testing.assert_allclose(ported.dst_R_src, shared.dst_R_src, atol=1e-8)
    np.testing.assert_allclose(ported.dst_t_src, shared.dst_t_src, atol=1e-8)
    assert ported.scale == 1.0
    # Whatever the alignment, both map the source onto the target identically.
    np.testing.assert_allclose(ported.apply(moved), shared.apply(moved), atol=1e-8)


def test_shift_clock_moves_a_trajectory_onto_the_absolute_device_clock() -> None:
    """The one conversion between video_time and the clock every basalt CSV uses."""
    offset_ns: int = 10_433_867_587_166
    positions: Float64[ndarray, "5 3"] = np.arange(15, dtype=np.float64).reshape(5, 3)
    relative: Trajectory = _trajectory(np.array([0, 17_504_134, 100, 2**53 + 1, 2**54 + 3], dtype=np.int64), positions)
    absolute: Trajectory = shift_clock(relative, offset_ns)
    assert absolute.t_ns.dtype == np.int64
    np.testing.assert_array_equal(absolute.t_ns, relative.t_ns + offset_ns)
    # The dossier's worked example: the sidecar's first row on the smoke segment.
    assert int(absolute.t_ns[1]) == 10_433_885_091_300
    # Positions and rotations are untouched, and the shift is exactly invertible.
    np.testing.assert_array_equal(absolute.position_m, relative.position_m)
    np.testing.assert_array_equal(shift_clock(absolute, -offset_ns).t_ns, relative.t_ns)
    # Nothing associates across the offset, which is the bug this prevents.
    assert associate(relative, absolute).count == 0


def test_the_association_tolerance_is_inclusive_on_both_edges() -> None:
    positions: Float64[ndarray, "3 3"] = np.zeros((3, 3), dtype=np.float64)
    t_ns: Int64[ndarray, " 3"] = np.array([0, 1_000_000_000, 2_000_000_000], dtype=np.int64)
    reference: Trajectory = _trajectory(t_ns, positions)
    for offset_ns, expected in ((ASSOCIATION_TOLERANCE_NS, 3), (ASSOCIATION_TOLERANCE_NS + 1, 0), (-ASSOCIATION_TOLERANCE_NS, 3)):
        assert associate(reference, _trajectory(t_ns + offset_ns, positions)).count == expected


@settings(deadline=None, max_examples=50)
@given(
    positions=POSITIONS,
    # Reach well past 2**53, where float64 stops representing every integer, and
    # keep the low bits odd: routing these through a float would round them and
    # the round-trip assertion below would fail.
    start_ns=st.integers(min_value=2**53 + 1, max_value=2**62).map(lambda value: value | 1),
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


def test_coverage_is_zero_when_either_trajectory_has_no_pose() -> None:
    """The documented ``0.0`` and not an ``IndexError``: a run that produced nothing covers nothing.

    Both indexes — ``reference.t_ns[-1]`` for the span and ``candidate.t_ns[-1]``
    for the overlap — were read before anything was checked, so an empty
    reference (a segment with no ``gt`` layer) and an empty candidate (a machine
    that tracked nothing) both raised ``IndexError: index -1 is out of bounds``
    where :func:`~slam_rs.trajectory.associate` and
    :func:`~slam_rs.trajectory.ate` refuse their empty cases in one sentence
    (S25 review).
    """
    t_ns: Int64[ndarray, " 3"] = np.arange(3, dtype=np.int64) * 1_000_000_000
    populated: Trajectory = _trajectory(t_ns, np.zeros((3, 3), dtype=np.float64))
    assert coverage(empty_trajectory(), populated) == 0.0
    assert coverage(populated, empty_trajectory()) == 0.0
    assert coverage(empty_trajectory(), empty_trajectory()) == 0.0
    # The populated pair still measures what it measured.
    assert coverage(populated, populated) == pytest.approx(1.0)


def test_shift_clock_refuses_an_offset_that_would_leave_the_int64_clock() -> None:
    """A wrapped timestamp is not a clock conversion, and int64 addition wraps in silence.

    ``t_ns`` is int64 because a nanosecond clock past 2**53 is not exact in a
    float, and the same choice makes the last representable timestamp one
    addition away from ``-2**63``: the pose would move backwards across the whole
    clock, associate with nothing and be read as a machine that tracked the
    wrong window. Shipped manifest clocks are nowhere near the boundary, which is
    exactly why the failure would arrive unannounced (S25 review).
    """
    positions: Float64[ndarray, "2 3"] = np.zeros((2, 3), dtype=np.float64)
    latest: Trajectory = _trajectory(np.array([0, 2**63 - 1], dtype=np.int64), positions)
    with pytest.raises(ValueError, match="9223372036854775807 ns cannot be shifted by 1 ns"):
        shift_clock(latest, 1)
    earliest: Trajectory = _trajectory(np.array([-(2**63), 0], dtype=np.int64), positions)
    with pytest.raises(ValueError, match="-9223372036854775808 ns cannot be shifted by -1 ns"):
        shift_clock(earliest, -1)
    # The boundary itself is a shift, not a margin: the two offsets that land
    # exactly on the ends are still made.
    assert int(shift_clock(_trajectory(np.array([2**63 - 2], dtype=np.int64), positions[:1]), 1).t_ns[0]) == 2**63 - 1
    assert int(shift_clock(_trajectory(np.array([-(2**63) + 1], dtype=np.int64), positions[:1]), -1).t_ns[0]) == -(2**63)
    # An empty trajectory has no timestamp to shift and none to refuse.
    assert len(shift_clock(empty_trajectory(), 2**62)) == 0


