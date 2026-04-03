"""Image quality metrics for evaluating Gaussian splat rendering.

Provides PSNR, SSIM, and LPIPS implementations matching the standard
definitions used by 3DGS, NeRF, and Brush for benchmarking novel view
synthesis.

LPIPS uses the standard VGG-based ``lpips`` package (requires PyTorch).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from jaxtyping import Float32


def load_image_rgb(path: Path) -> Float32[np.ndarray, "h w 3"]:
    """Load an image as float32 RGB in [0, 1].

    Handles RGBA PNGs by compositing alpha over a white background
    (matching the NeRF Synthetic convention).

    Args:
        path: Path to the image file (PNG, JPG, etc.).

    Returns:
        Float32 RGB array with shape ``(H, W, 3)`` in ``[0, 1]``.
    """
    import cv2

    raw: np.ndarray = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    if raw.ndim == 2:
        # Grayscale → RGB
        raw = np.stack([raw, raw, raw], axis=-1)

    if raw.shape[2] == 4:
        # RGBA → composite over white background (NeRF Synthetic convention)
        alpha: Float32[np.ndarray, "h w 1"] = raw[:, :, 3:4].astype(np.float32) / 255.0
        rgb: Float32[np.ndarray, "h w 3"] = raw[:, :, :3].astype(np.float32) / 255.0
        # OpenCV loads BGR, convert to RGB
        rgb = rgb[:, :, ::-1].copy()
        img: Float32[np.ndarray, "h w 3"] = rgb * alpha + (1.0 - alpha)
    else:
        # BGR → RGB
        img = raw[:, :, ::-1].astype(np.float32) / 255.0

    return np.ascontiguousarray(img)


def psnr(
    rendered: Float32[np.ndarray, "h w 3"],
    ground_truth: Float32[np.ndarray, "h w 3"],
) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images.

    Both images must be float32 in [0, 1] with the same shape.
    Higher is better.  Typical values for good 3DGS reconstructions:
    25-35 dB.

    Args:
        rendered: The rendered image.
        ground_truth: The reference ground truth image.

    Returns:
        PSNR in decibels.
    """
    mse: float = float(np.mean((rendered - ground_truth) ** 2))
    if mse < 1e-10:
        return 100.0  # Perfect match
    return float(10.0 * np.log10(1.0 / mse))


def _gaussian_kernel_1d(size: int, sigma: float) -> Float32[np.ndarray, "k"]:
    """Create a 1D Gaussian kernel.

    Args:
        size: Kernel size (should be odd).
        sigma: Standard deviation.

    Returns:
        Normalized 1D Gaussian kernel.
    """
    coords: Float32[np.ndarray, "k"] = np.arange(size, dtype=np.float32) - size // 2
    kernel: Float32[np.ndarray, "k"] = np.exp(-0.5 * (coords / sigma) ** 2)
    return kernel / kernel.sum()


def _gaussian_blur(
    img: Float32[np.ndarray, "h w c"],
    window_size: int = 11,
    sigma: float = 1.5,
) -> Float32[np.ndarray, "h w c"]:
    """Apply separable Gaussian blur to an image.

    Uses 1D convolutions for efficiency (matching Brush's SSIM implementation).

    Args:
        img: Input image with shape ``(H, W, C)``.
        window_size: Size of the Gaussian window.
        sigma: Standard deviation of the Gaussian.

    Returns:
        Blurred image with same shape.
    """
    kernel: Float32[np.ndarray, "k"] = _gaussian_kernel_1d(window_size, sigma)
    pad: int = window_size // 2

    # Pad with reflect to avoid border artifacts
    padded: Float32[np.ndarray, "h2 w2 c"] = np.pad(
        img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect"
    )

    # Separable convolution: horizontal then vertical
    h, w, c = padded.shape
    # Horizontal pass
    horiz: Float32[np.ndarray, "h w2 c"] = np.zeros((h, w - 2 * pad, c), dtype=np.float32)
    for i in range(window_size):
        horiz += padded[:, i : i + horiz.shape[1], :] * kernel[i]

    # Vertical pass
    result: Float32[np.ndarray, "h w c"] = np.zeros((h - 2 * pad, horiz.shape[1], c), dtype=np.float32)
    for i in range(window_size):
        result += horiz[i : i + result.shape[0], :, :] * kernel[i]

    return result


def ssim(
    rendered: Float32[np.ndarray, "h w 3"],
    ground_truth: Float32[np.ndarray, "h w 3"],
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    data_range: float = 1.0,
) -> float:
    """Compute mean Structural Similarity Index between two images.

    Uses the standard SSIM formulation with a Gaussian window, computed
    per channel and averaged.  Matches the implementation used by Brush
    and standard 3DGS evaluation.

    Args:
        rendered: The rendered image, float32 in [0, 1].
        ground_truth: The reference ground truth image, float32 in [0, 1].
        window_size: Size of the Gaussian window.
        sigma: Standard deviation of the Gaussian window.
        k1: Stability constant for luminance.
        k2: Stability constant for contrast.
        data_range: Dynamic range of the images (1.0 for [0, 1] images).

    Returns:
        Mean SSIM value in [0, 1].  Higher is better.
    """
    c1: float = (k1 * data_range) ** 2
    c2: float = (k2 * data_range) ** 2

    mu_x: Float32[np.ndarray, "h w 3"] = _gaussian_blur(rendered, window_size, sigma)
    mu_y: Float32[np.ndarray, "h w 3"] = _gaussian_blur(ground_truth, window_size, sigma)

    mu_x_sq: Float32[np.ndarray, "h w 3"] = mu_x ** 2
    mu_y_sq: Float32[np.ndarray, "h w 3"] = mu_y ** 2
    mu_xy: Float32[np.ndarray, "h w 3"] = mu_x * mu_y

    sigma_x_sq: Float32[np.ndarray, "h w 3"] = _gaussian_blur(rendered ** 2, window_size, sigma) - mu_x_sq
    sigma_y_sq: Float32[np.ndarray, "h w 3"] = _gaussian_blur(ground_truth ** 2, window_size, sigma) - mu_y_sq
    sigma_xy: Float32[np.ndarray, "h w 3"] = _gaussian_blur(rendered * ground_truth, window_size, sigma) - mu_xy

    numerator: Float32[np.ndarray, "h w 3"] = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator: Float32[np.ndarray, "h w 3"] = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)

    ssim_map: Float32[np.ndarray, "h w 3"] = numerator / denominator
    return float(np.mean(ssim_map))


_lpips_model: object | None = None


def _get_lpips_model() -> object:
    """Return a cached LPIPS VGG model instance.

    The model is loaded on first call and reused for subsequent calls.
    Weights are downloaded automatically by the ``lpips`` package on
    first use (~22 MB).
    """
    global _lpips_model
    if _lpips_model is None:
        import lpips as lpips_pkg

        _lpips_model = lpips_pkg.LPIPS(net="vgg", verbose=False)
    return _lpips_model


def lpips(
    rendered: Float32[np.ndarray, "h w 3"],
    ground_truth: Float32[np.ndarray, "h w 3"],
) -> float:
    """Compute Learned Perceptual Image Patch Similarity (LPIPS).

    Uses the standard VGG-based LPIPS model from the ``lpips`` package.
    Lower is better.  Typical values for good 3DGS reconstructions:
    0.05-0.20.

    Args:
        rendered: The rendered image, float32 in [0, 1].
        ground_truth: The reference ground truth image, float32 in [0, 1].

    Returns:
        LPIPS distance (lower = more perceptually similar).
    """
    import torch

    model: object = _get_lpips_model()

    # lpips expects NCHW tensors in [-1, 1].
    rendered_t: torch.Tensor = torch.from_numpy(rendered).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    gt_t: torch.Tensor = torch.from_numpy(ground_truth).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0

    with torch.no_grad():
        score: torch.Tensor = model(rendered_t, gt_t)

    return float(score.item())


def compute_metrics(
    rendered_path: Path,
    ground_truth_path: Path,
) -> dict[str, float]:
    """Compute PSNR, SSIM, and LPIPS between a rendered image and ground truth.

    Args:
        rendered_path: Path to the rendered image.
        ground_truth_path: Path to the ground truth image.

    Returns:
        Dictionary with ``"psnr"``, ``"ssim"``, and ``"lpips"`` keys.
    """
    rendered: Float32[np.ndarray, "h w 3"] = load_image_rgb(rendered_path)
    gt: Float32[np.ndarray, "h w 3"] = load_image_rgb(ground_truth_path)

    if rendered.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch: rendered {rendered.shape} vs ground_truth {gt.shape}"
        )

    # 8-bit roundtrip to match Brush's eval convention
    rendered = np.round(rendered * 255.0).astype(np.float32) / 255.0
    gt = np.round(gt * 255.0).astype(np.float32) / 255.0

    return {
        "psnr": psnr(rendered, gt),
        "ssim": ssim(rendered, gt),
        "lpips": lpips(rendered, gt),
    }
