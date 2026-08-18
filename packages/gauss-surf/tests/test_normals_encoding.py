"""Behavior checks for the lossless uint8 normal-map representation."""

import numpy as np
from jaxtyping import Float32
from numpy import ndarray

from gauss_surf.normals_encoding import decode_normals_png, encode_normals_png, to_away_from_camera


def test_random_unit_normals_survive_png_with_sub_half_degree_error() -> None:
    """Lossless PNG storage adds only the expected uint8 angular quantization."""
    rng: np.random.Generator = np.random.default_rng(20260815)
    raw_hw3: Float32[ndarray, "h=48 w=64 3"] = rng.normal(size=(48, 64, 3)).astype(np.float32)
    normals_hw3: Float32[ndarray, "h=48 w=64 3"] = raw_hw3 / np.linalg.norm(raw_hw3, axis=-1, keepdims=True)

    png_bytes: bytes = encode_normals_png(normals_hw3)
    decoded_hw3: Float32[ndarray, "h=48 w=64 3"] = decode_normals_png(png_bytes)

    cosine_hw: Float32[ndarray, "h=48 w=64"] = np.sum(normals_hw3 * decoded_hw3, axis=-1) / np.linalg.norm(decoded_hw3, axis=-1)
    angular_error_hw: Float32[ndarray, "h=48 w=64"] = np.rad2deg(np.arccos(np.clip(cosine_hw, -1.0, 1.0)))
    assert float(np.max(angular_error_hw)) < 0.5
    assert float(np.mean(angular_error_hw)) < 0.2


def test_signed_mapping_inverts_endpoints_and_zero_exactly() -> None:
    """The three signed anchor values retain exact meanings after quantization."""
    anchors_hw3: Float32[ndarray, "h=1 w=1 3"] = np.array([[[-1.0, 0.0, 1.0]]], dtype=np.float32)

    decoded_hw3: Float32[ndarray, "h=1 w=1 3"] = decode_normals_png(encode_normals_png(anchors_hw3))

    np.testing.assert_array_equal(decoded_hw3, anchors_hw3)


def test_moge_toward_camera_normals_are_negated_for_gaussurf_training() -> None:
    """Both normal layers store the away-from-camera RDF convention."""
    toward_hw3: Float32[ndarray, "h=1 w=2 3"] = np.array(
        [[[0.0, 0.0, -1.0], [0.6, 0.0, -0.8]]],
        dtype=np.float32,
    )

    away_hw3: Float32[ndarray, "h=1 w=2 3"] = to_away_from_camera(toward_hw3)

    np.testing.assert_array_equal(away_hw3, -toward_hw3)
    assert float(np.mean(away_hw3[..., 2] > 0.0)) == 1.0
