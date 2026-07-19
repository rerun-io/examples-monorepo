"""Tests for the PSNR and SSIM metrics implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from jaxtyping import Float32, UInt8
from PIL import Image

from gsplat_rust_renderer.metrics import load_image_rgb, psnr, ssim


def test_load_image_rgb_matches_nerfbaselines_uint8_alpha_composite(tmp_path: Path) -> None:
    """RGBA ground truth is composited over white and truncated to 8-bit."""
    rgba_hwc: UInt8[np.ndarray, "h w 4"] = np.array([[[64, 32, 16, 128]]], dtype=np.uint8)
    image_path: Path = tmp_path / "rgba.png"
    Image.fromarray(rgba_hwc, mode="RGBA").save(image_path)

    rgb_hwc: Float32[np.ndarray, "h w 3"] = load_image_rgb(image_path)

    np.testing.assert_array_equal(rgb_hwc, np.array([[[159, 143, 135]]], dtype=np.float32) / 255.0)


def test_psnr_identical_images() -> None:
    """PSNR of identical images should be very high (effectively infinite)."""
    img: Float32[np.ndarray, "h w 3"] = np.random.rand(64, 64, 3).astype(np.float32)
    result: float = psnr(img, img)
    assert result >= 99.0


def test_psnr_known_value() -> None:
    """PSNR with known MSE should match the analytical formula."""
    a: Float32[np.ndarray, "h w 3"] = np.zeros((64, 64, 3), dtype=np.float32)
    b: Float32[np.ndarray, "h w 3"] = np.ones((64, 64, 3), dtype=np.float32) * 0.1
    # MSE = 0.01, PSNR = 10 * log10(1/0.01) = 20 dB
    result: float = psnr(a, b)
    np.testing.assert_allclose(result, 20.0, atol=0.01)


def test_psnr_symmetric() -> None:
    """PSNR should be symmetric: psnr(a, b) == psnr(b, a)."""
    rng: np.random.Generator = np.random.default_rng(42)
    a: Float32[np.ndarray, "h w 3"] = rng.random((32, 32, 3)).astype(np.float32)
    b: Float32[np.ndarray, "h w 3"] = rng.random((32, 32, 3)).astype(np.float32)
    np.testing.assert_allclose(psnr(a, b), psnr(b, a), atol=1e-6)


def test_ssim_identical_images() -> None:
    """SSIM of identical images should be 1.0."""
    img: Float32[np.ndarray, "h w 3"] = np.random.rand(64, 64, 3).astype(np.float32)
    result: float = ssim(img, img)
    np.testing.assert_allclose(result, 1.0, atol=1e-4)


def test_ssim_different_images() -> None:
    """SSIM of very different images should be low."""
    a: Float32[np.ndarray, "h w 3"] = np.zeros((64, 64, 3), dtype=np.float32)
    b: Float32[np.ndarray, "h w 3"] = np.ones((64, 64, 3), dtype=np.float32)
    result: float = ssim(a, b)
    assert result < 0.1


def test_ssim_range() -> None:
    """SSIM should always be in [-1, 1] range."""
    rng: np.random.Generator = np.random.default_rng(42)
    a: Float32[np.ndarray, "h w 3"] = rng.random((64, 64, 3)).astype(np.float32)
    b: Float32[np.ndarray, "h w 3"] = rng.random((64, 64, 3)).astype(np.float32)
    result: float = ssim(a, b)
    assert -1.0 <= result <= 1.0


def test_ssim_matches_nerfbaselines_valid_window_reference() -> None:
    """SSIM uses dm-pix's valid window and stabilized covariance convention."""
    rng: np.random.Generator = np.random.default_rng(7)
    a: Float32[np.ndarray, "h w 3"] = rng.random((16, 16, 3), dtype=np.float32)
    b: Float32[np.ndarray, "h w 3"] = np.clip(a * 0.8 + 0.1, 0.0, 1.0)

    result: float = ssim(a, b)

    np.testing.assert_allclose(result, 0.9755787587223764, atol=1e-7)
