import numpy as np
import pytest
from jaxtyping import Float32, Float64
from numpy import ndarray

from simplecv.ops.umeyama import SimilarityTransform, umeyama_alignment


def _rotation_about_z(angle_rad: float) -> Float64[ndarray, "3 3"]:
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def test_umeyama_alignment_recovers_known_similarity() -> None:
    rng = np.random.default_rng(0)
    src: Float64[ndarray, "n 3"] = rng.normal(size=(12, 3))
    rotation: Float64[ndarray, "3 3"] = _rotation_about_z(0.7)
    translation: Float64[ndarray, "3"] = np.array([0.3, -1.2, 2.5])
    dst: Float64[ndarray, "n 3"] = 1.8 * (rotation @ src.T).T + translation

    similarity: SimilarityTransform = umeyama_alignment(src, dst, allow_scaling=True)

    assert np.allclose(similarity.dst_R_src, rotation, atol=1e-9)
    assert np.allclose(similarity.dst_t_src, translation, atol=1e-9)
    assert similarity.scale == pytest.approx(1.8, abs=1e-9)
    assert np.allclose(similarity.scale * (similarity.dst_R_src @ src.T).T + similarity.dst_t_src, dst, atol=1e-9)


def test_umeyama_alignment_without_scaling_keeps_unit_scale() -> None:
    src: Float64[ndarray, "n 3"] = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    dst: Float64[ndarray, "n 3"] = 2.0 * src + 1.0

    similarity: SimilarityTransform = umeyama_alignment(src, dst, allow_scaling=False)

    assert similarity.scale == 1.0


def test_umeyama_alignment_estimates_in_float64_from_float32_inputs() -> None:
    src: Float32[ndarray, "n 3"] = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    similarity: SimilarityTransform = umeyama_alignment(src, src + np.float32(0.5))

    assert similarity.dst_R_src.dtype == np.float64 and similarity.dst_t_src.dtype == np.float64
    assert np.allclose(similarity.dst_t_src, np.full(3, 0.5), atol=1e-12)


def test_umeyama_alignment_rejects_degenerate_source() -> None:
    src: Float64[ndarray, "n 3"] = np.zeros((4, 3))
    with pytest.raises(ValueError, match="variance"):
        umeyama_alignment(src, src + 1.0)


def test_umeyama_alignment_rejects_fewer_than_three_correspondences() -> None:
    src: Float64[ndarray, "n 3"] = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # two distinct points pass the variance check
    with pytest.raises(ValueError, match="at least 3 correspondences"):
        umeyama_alignment(src, src + 1.0)


def test_similarity_transform_apply_and_matrix_agree_with_the_formula() -> None:
    rng = np.random.default_rng(1)
    src: Float64[ndarray, "n 3"] = rng.normal(size=(6, 3))
    dst: Float64[ndarray, "n 3"] = 0.5 * (_rotation_about_z(-0.4) @ src.T).T + np.array([1.0, 2.0, 3.0])

    similarity: SimilarityTransform = umeyama_alignment(src, dst, allow_scaling=True)
    src_h: Float64[ndarray, "n 4"] = np.column_stack((src, np.ones(6)))

    assert np.allclose(similarity.apply(src), dst, atol=1e-9)
    assert np.allclose((similarity.as_matrix() @ src_h.T).T[:, :3], dst, atol=1e-9)
    assert similarity.as_matrix().dtype == np.float64
