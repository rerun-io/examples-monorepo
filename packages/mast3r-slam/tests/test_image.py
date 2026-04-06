"""Unit tests for mast3r_slam.image."""

import torch
from jaxtyping import Float
from torch import Tensor

from mast3r_slam.image import img_gradient


def test_img_gradient_shape_preserved() -> None:
    """Output gradients should have the same shape as input."""
    img: Float[Tensor, "1 3 16 16"] = torch.randn(1, 3, 16, 16)
    gx: Float[Tensor, "1 3 16 16"]
    gy: Float[Tensor, "1 3 16 16"]
    gx, gy = img_gradient(img)
    assert gx.shape == img.shape
    assert gy.shape == img.shape


def test_img_gradient_constant_image() -> None:
    """Gradient of a constant image should be zero everywhere."""
    img: Float[Tensor, "1 1 8 8"] = torch.ones(1, 1, 8, 8) * 42.0
    gx, gy = img_gradient(img)
    torch.testing.assert_close(gx, torch.zeros_like(gx), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(gy, torch.zeros_like(gy), atol=1e-6, rtol=0.0)


def test_img_gradient_horizontal_step() -> None:
    """Horizontal step edge should produce non-zero gx, near-zero gy in center."""
    img: Float[Tensor, "1 1 8 8"] = torch.zeros(1, 1, 8, 8)
    img[:, :, :, 4:] = 1.0  # step at column 4
    gx, gy = img_gradient(img)
    # Center of step should have positive gx
    assert gx[0, 0, 4, 4].item() > 0.01
