from __future__ import annotations

import cv2
import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from jaxtyping import Bool, Float, Int, UInt8
from numpy import ndarray
from torch import Tensor

from wilor_nano.torch_image_patch import generate_rgb_image_patches_torch
from wilor_nano.utils import utils

TORCH_UINT8: torch.dtype = torch.__dict__["uint8"]


@given(
    height=st.integers(min_value=24, max_value=96),
    width=st.integers(min_value=24, max_value=96),
    center_x=st.floats(min_value=4.0, max_value=91.0, allow_nan=False, allow_infinity=False),
    center_y=st.floats(min_value=4.0, max_value=91.0, allow_nan=False, allow_infinity=False),
    box_size=st.floats(min_value=8.0, max_value=80.0, allow_nan=False, allow_infinity=False),
    do_flip=st.booleans(),
)
@settings(max_examples=80, deadline=None)
def test_torch_rgb_patch_matches_opencv_affine(
    height: int,
    width: int,
    center_x: float,
    center_y: float,
    box_size: float,
    do_flip: bool,
) -> None:
    patch_size: int = 32
    rng: np.random.Generator = np.random.default_rng(height * 1000 + width)
    rgb_image: UInt8[ndarray, "h w 3"] = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    bounded_center_x: float = min(max(center_x, -float(width)), 2.0 * float(width))
    bounded_center_y: float = min(max(center_y, -float(height)), 2.0 * float(height))

    opencv_patch: UInt8[ndarray, "patch patch 3"]
    opencv_patch, _trans = utils.generate_image_patch_cv2(
        rgb_image,
        bounded_center_x,
        bounded_center_y,
        box_size,
        box_size,
        patch_size,
        patch_size,
        do_flip,
        1.0,
        0.0,
        border_mode=cv2.BORDER_CONSTANT,
    )

    frames_rgb: UInt8[Tensor, "batch=1 h w 3"] = torch.from_numpy(rgb_image[None])
    frame_indices: Int[Tensor, "num_patches=1"] = torch.tensor([0], dtype=torch.long)
    centers_xy: Float[Tensor, "num_patches=1 2"] = torch.tensor([[bounded_center_x, bounded_center_y]], dtype=torch.float32)
    box_sizes: Float[Tensor, "num_patches=1"] = torch.tensor([box_size], dtype=torch.float32)
    flips: Bool[Tensor, "num_patches=1"] = torch.tensor([do_flip], dtype=torch.bool)

    torch_patch: UInt8[Tensor, "num_patches=1 patch patch 3"] = (
        generate_rgb_image_patches_torch(
            frames_rgb,
            frame_indices=frame_indices,
            centers_xy=centers_xy,
            box_sizes=box_sizes,
            flips=flips,
            patch_size=patch_size,
        )
        .round()
        .clamp(0, 255)
        .to(TORCH_UINT8)
    )

    np.testing.assert_allclose(torch_patch[0].numpy(), opencv_patch, atol=8)


def test_torch_rgb_patch_runs_on_cuda_when_available() -> None:
    if not torch.cuda.is_available():
        return
    frames_rgb: UInt8[Tensor, "batch h w 3"] = torch.arange(2 * 32 * 32 * 3, device="cuda").to(TORCH_UINT8).reshape(2, 32, 32, 3)
    patches: Float[Tensor, "num_patches patch patch 3"] = generate_rgb_image_patches_torch(
        frames_rgb,
        frame_indices=torch.tensor([0, 1], dtype=torch.long, device="cuda"),
        centers_xy=torch.tensor([[16.0, 16.0], [12.0, 18.0]], dtype=torch.float32, device="cuda"),
        box_sizes=torch.tensor([24.0, 20.0], dtype=torch.float32, device="cuda"),
        flips=torch.tensor([False, True], dtype=torch.bool, device="cuda"),
        patch_size=16,
    )

    assert patches.device.type == "cuda"
    assert patches.shape == (2, 16, 16, 3)
