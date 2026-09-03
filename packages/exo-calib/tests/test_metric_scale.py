import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from jaxtyping import Float64, Int64
from numpy import ndarray

from exo_calib.metric_scale import grouped_medians


def _reference(group: Int64[ndarray, " n"], values: Float64[ndarray, " n"], min_count: int) -> tuple[Float64[ndarray, " g"], int]:
    """The mask-and-median loop the vectorized helper replaced."""
    medians: list[float] = []
    n_samples: int = 0
    for group_id in np.unique(group):
        members: Float64[ndarray, " m"] = values[group == group_id]
        if members.size >= min_count:
            medians.append(float(np.median(members)))
            n_samples += members.size
    return np.asarray(medians, dtype=np.float64), n_samples


@settings(max_examples=300, deadline=None)
@given(
    seed=st.integers(0, 2**31 - 1),
    n_samples=st.integers(0, 400),
    n_groups=st.integers(1, 40),
    min_count=st.integers(1, 6),
)
def test_grouped_medians_matches_per_group_numpy_median(seed: int, n_samples: int, n_groups: int, min_count: int) -> None:
    rng = np.random.default_rng(seed)
    group: Int64[ndarray, " n"] = rng.integers(0, n_groups, n_samples).astype(np.int64)
    # Repeated values exercise the even-count midpoint and ties in the sort.
    values: Float64[ndarray, " n"] = rng.choice(rng.uniform(0.5, 1.5, 12), n_samples)

    medians, pooled = grouped_medians(group, values, min_count)
    expected_medians, expected_pooled = _reference(group, values, min_count)

    np.testing.assert_array_equal(medians, expected_medians)
    assert pooled == expected_pooled


def test_grouped_medians_drops_small_groups_and_reports_pooled_count() -> None:
    group: Int64[ndarray, " n"] = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    values: Float64[ndarray, " n"] = np.array([3.0, 1.0, 2.0, 9.0, 9.0, 4.0, 1.0, 3.0, 2.0])

    medians, pooled = grouped_medians(group, values, min_count=3)

    np.testing.assert_array_equal(medians, np.array([2.0, 2.5]))
    assert pooled == 7


def test_grouped_medians_on_no_samples_is_empty() -> None:
    medians, pooled = grouped_medians(np.zeros(0, dtype=np.int64), np.zeros(0), min_count=3)

    assert medians.shape == (0,)
    assert pooled == 0
