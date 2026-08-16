"""Behavior checks for chosen-frame MoGe normal inference."""

import numpy as np
import pytest
import torch
from arkitscenes_download.ingest.paths import FRAME_SELECTION_WIDE
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn
from jaxtyping import Float32, Int64, UInt8
from numpy import ndarray
from torch import Tensor

from gauss_surf.catalog import match_exact_timestamps
from gauss_surf.contracts import WIDE_CHOSEN_SHARPNESS_COLUMN
from gauss_surf.normals_encoding import decode_normals_png, encode_normals_png

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@st.composite
def exact_timestamp_cases(
    draw: DrawFn,
) -> tuple[ndarray, ndarray, ndarray]:
    """Generate a sorted packet timeline and a permuted exact target subset."""
    packet_values: list[int] = sorted(
        draw(
            st.lists(
                st.integers(-1_000_000_000, 1_000_000_000),
                min_size=1,
                max_size=20,
                unique=True,
            )
        )
    )
    target_indices: list[int] = draw(
        st.lists(
            st.integers(0, len(packet_values) - 1),
            min_size=1,
            max_size=len(packet_values),
            unique=True,
        )
    )
    permuted_indices: list[int] = list(draw(st.permutations(target_indices)))
    packet_times_n: ndarray = np.asarray(packet_values, dtype=np.int64).astype(
        "timedelta64[ns]"
    )
    target_times_n: ndarray = packet_times_n[permuted_indices]
    expected_indices_n: ndarray = np.asarray(permuted_indices, dtype=np.int64)
    return packet_times_n, target_times_n, expected_indices_n


@st.composite
def drift_timestamp_cases(draw: DrawFn) -> tuple[ndarray, ndarray, ndarray]:
    """Generate one exact timestamp and an absent one-nanosecond neighbor."""
    start: int = draw(st.integers(-1_000_000_000, 1_000_000_000))
    gaps: list[int] = draw(
        st.lists(st.integers(3, 1_000_000), min_size=1, max_size=16)
    )
    packet_values: list[int] = [start]
    for gap in gaps:
        packet_values.append(packet_values[-1] + gap)
    chosen_index: int = draw(st.integers(0, len(packet_values) - 1))
    drift_ns: int = draw(st.sampled_from((-1, 1)))
    packet_times_n: ndarray = np.asarray(packet_values, dtype=np.int64).astype(
        "timedelta64[ns]"
    )
    target_times_n: ndarray = packet_times_n[[chosen_index]]
    drifted_times_n: ndarray = (
        target_times_n.astype(np.int64) + drift_ns
    ).astype("timedelta64[ns]")
    return packet_times_n, target_times_n, drifted_times_n


@settings(max_examples=64, deadline=None)
@given(case=exact_timestamp_cases())
def test_exact_timestamp_matching_is_invariant_to_target_permutation(
    case: tuple[ndarray, ndarray, ndarray],
) -> None:
    """Target order changes output order only; packet identities remain exact."""
    packet_times_n: ndarray = case[0]
    target_times_n: ndarray = case[1]
    expected_indices_n: ndarray = case[2]

    matched_indices_n: ndarray = match_exact_timestamps(
        packet_times_n, target_times_n
    )

    np.testing.assert_array_equal(matched_indices_n, expected_indices_n)


@settings(max_examples=32, deadline=None)
@given(
    packet_values=st.lists(
        st.integers(-1_000_000_000, 1_000_000_000),
        min_size=2,
        max_size=20,
        unique=True,
    )
)
def test_exact_timestamp_matching_rejects_an_unsorted_packet_timeline(
    packet_values: list[int],
) -> None:
    """Binary-search matching refuses packet timelines that are not sorted."""
    descending_values: list[int] = sorted(packet_values, reverse=True)
    packet_times_n: ndarray = np.asarray(
        descending_values, dtype=np.int64
    ).astype("timedelta64[ns]")
    target_times_n: ndarray = packet_times_n[:1]

    with pytest.raises(ValueError, match="Packet timestamps must be sorted"):
        match_exact_timestamps(packet_times_n, target_times_n)


@settings(max_examples=48, deadline=None)
@given(case=drift_timestamp_cases())
def test_exact_timestamp_matching_refuses_plus_or_minus_one_nanosecond_drift(
    case: tuple[ndarray, ndarray, ndarray],
) -> None:
    """An exact hit succeeds, but either one-nanosecond neighbor is absent."""
    packet_times_n: ndarray = case[0]
    target_times_n: ndarray = case[1]
    drifted_times_n: ndarray = case[2]
    chosen_index: int = int(np.searchsorted(packet_times_n, target_times_n[0]))

    matched_indices_n: ndarray = match_exact_timestamps(
        packet_times_n, target_times_n
    )
    np.testing.assert_array_equal(
        matched_indices_n, np.asarray([chosen_index], dtype=np.int64)
    )
    with pytest.raises(ValueError, match="no exact video-packet match"):
        match_exact_timestamps(packet_times_n, drifted_times_n)


def test_moge_inference_is_anchored_to_chosen_wide_rows() -> None:
    """The provenance stage reads frame-selection rows, not the dense PromptDA grid."""
    assert f"/{FRAME_SELECTION_WIDE}:sharpness" == WIDE_CHOSEN_SHARPNESS_COLUMN


def test_chosen_timestamps_match_exact_packet_times_despite_jitter() -> None:
    """Exact packet identities survive irregular inter-frame timing."""
    packet_times_n: ndarray = np.array([0, 16_666_667, 33_333_333, 50_000_001, 66_666_669], dtype="timedelta64[ns]")
    chosen_times_n: ndarray = np.array([0, 33_333_333, 66_666_669], dtype="timedelta64[ns]")

    matched_indices_n: Int64[ndarray, "n_chosen=3"] = match_exact_timestamps(packet_times_n, chosen_times_n)

    np.testing.assert_array_equal(matched_indices_n, np.array([0, 2, 4], dtype=np.int64))


def test_chosen_timestamp_matching_refuses_one_nanosecond_drift() -> None:
    """A nearby packet never substitutes for a missing chosen timestamp."""
    packet_times_n: ndarray = np.array([0, 16_666_667, 33_333_333], dtype="timedelta64[ns]")
    drifted_chosen_n: ndarray = np.array([33_333_334], dtype="timedelta64[ns]")

    with pytest.raises(ValueError, match="no exact video-packet match"):
        match_exact_timestamps(packet_times_n, drifted_chosen_n)


@requires_cuda
def test_trt_front_facing_normals_keep_unit_length_and_positive_z_after_png() -> None:
    """The catalog representation stores away-from-camera RDF normals (the gaussurf training convention)."""
    from monopriors.models.surface_normal.moge_v2_trt import DEFAULT_IMAGE_HW, MoGeV2NormalOutput, MoGeV2TrtNormalPredictor

    height: int = DEFAULT_IMAGE_HW[0]
    width: int = DEFAULT_IMAGE_HW[1]
    x_w: Float32[Tensor, "w"] = torch.linspace(0.0, 255.0, width, dtype=torch.float32, device="cuda")
    y_h: Float32[Tensor, "h"] = torch.linspace(0.0, 255.0, height, dtype=torch.float32, device="cuda")
    red_hw: Float32[Tensor, "h w"] = x_w[None, :].expand(height, width)
    green_hw: Float32[Tensor, "h w"] = y_h[:, None].expand(height, width)
    blue_hw: Float32[Tensor, "h w"] = (red_hw + green_hw) / 2.0
    rgb_bhw3: UInt8[Tensor, "b=1 h=756 w=1008 3"] = torch.stack((red_hw, green_hw, blue_hw), dim=-1).to(torch.uint8)[None]
    predictor: MoGeV2TrtNormalPredictor = MoGeV2TrtNormalPredictor(batch_size=8)

    prediction: MoGeV2NormalOutput = predictor(rgb_bhw3)
    # Mirror the layer path: MoGe's toward-camera output is negated before encoding.
    normals_hw3: Float32[ndarray, "h=756 w=1008 3"] = (-prediction.normals_bhw3[0]).detach().cpu().numpy()
    decoded_hw3: Float32[ndarray, "h=756 w=1008 3"] = decode_normals_png(encode_normals_png(normals_hw3))
    lengths_hw: Float32[ndarray, "h=756 w=1008"] = np.linalg.norm(decoded_hw3, axis=-1)

    np.testing.assert_allclose(lengths_hw, 1.0, atol=0.007, rtol=0.0)
    assert float(np.mean(decoded_hw3[..., 2] > 0.0)) > 0.9
