"""Unit tests for ExoOnlyCalibService and Umeyama transform."""

from __future__ import annotations

import numpy as np
import pytest
from jaxtyping import Float32
from numpy import ndarray

from mv_api.api.exo_only_calibration import (
    WB_UPPER_BODY_IDS,
    _umeyama_transform,
)


class TestUmeyamaTransform:
    """Tests for the _umeyama_transform function."""

    def test_identity_when_points_match(self) -> None:
        """When source and destination are identical, result should be identity."""
        pts: Float32[ndarray, "5 3"] = np.random.randn(5, 3).astype(np.float32)
        transform: Float32[ndarray, "4 4"] = _umeyama_transform(
            src_points=pts, dst_points=pts, allow_scaling=False
        )
        # Should be close to identity
        expected: Float32[ndarray, "4 4"] = np.eye(4, dtype=np.float32)
        np.testing.assert_allclose(transform, expected, atol=1e-5)

    def test_translation_only(self) -> None:
        """Test that pure translation is recovered."""
        src: Float32[ndarray, "4 3"] = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
        )
        offset: Float32[ndarray, "3"] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        dst: Float32[ndarray, "4 3"] = src + offset

        transform: Float32[ndarray, "4 4"] = _umeyama_transform(
            src_points=src, dst_points=dst, allow_scaling=False
        )

        # Translation should be recovered
        np.testing.assert_allclose(transform[:3, 3], offset, atol=1e-5)
        # Rotation should be identity
        np.testing.assert_allclose(transform[:3, :3], np.eye(3), atol=1e-5)

    def test_rotation_is_recovered(self) -> None:
        """Test that 90-degree rotation around Z is recovered."""
        src: Float32[ndarray, "4 3"] = np.array(
            [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=np.float32
        )
        # Rotate 90 degrees about Z: (x, y, z) -> (-y, x, z)
        dst: Float32[ndarray, "4 3"] = np.array(
            [[0, 1, 0], [-1, 0, 0], [0, -1, 0], [1, 0, 0]], dtype=np.float32
        )

        transform: Float32[ndarray, "4 4"] = _umeyama_transform(
            src_points=src, dst_points=dst, allow_scaling=False
        )

        # Apply transform and check alignment
        src_h: Float32[ndarray, "4 4"] = np.hstack([src, np.ones((4, 1), dtype=np.float32)])
        aligned: Float32[ndarray, "4 3"] = (transform @ src_h.T).T[:, :3]
        np.testing.assert_allclose(aligned, dst, atol=1e-5)

    def test_scaling_is_recovered(self) -> None:
        """Test that isotropic scaling is recovered when allow_scaling=True."""
        src: Float32[ndarray, "4 3"] = np.array(
            [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=np.float32
        )
        scale: float = 2.0
        dst: Float32[ndarray, "4 3"] = src * scale

        transform: Float32[ndarray, "4 4"] = _umeyama_transform(
            src_points=src, dst_points=dst, allow_scaling=True
        )

        # Apply transform and check alignment
        src_h: Float32[ndarray, "4 4"] = np.hstack([src, np.ones((4, 1), dtype=np.float32)])
        aligned: Float32[ndarray, "4 3"] = (transform @ src_h.T).T[:, :3]
        np.testing.assert_allclose(aligned, dst, atol=1e-5)

    def test_insufficient_points_raises(self) -> None:
        """Umeyama requires at least 3 points."""
        src: Float32[ndarray, "2 3"] = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        dst: Float32[ndarray, "2 3"] = src.copy()

        with pytest.raises(ValueError, match="at least 3 correspondences"):
            _umeyama_transform(src_points=src, dst_points=dst)


class TestWBUpperBodyIDs:
    """Verify the precomputed filter indices constant."""

    def test_contains_expected_indices(self) -> None:
        """WB_UPPER_BODY_IDS should contain upper body, face, and hand keypoints."""
        # Upper body: 5, 6, 7, 8, 9, 10 (shoulders, elbows, wrists)
        for idx in [5, 6, 7, 8, 9, 10]:
            assert idx in WB_UPPER_BODY_IDS

    def test_no_lower_body(self) -> None:
        """WB_UPPER_BODY_IDS should not contain lower body keypoints (11-16)."""
        for idx in [11, 12, 13, 14, 15, 16]:
            assert idx not in WB_UPPER_BODY_IDS
