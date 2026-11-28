"""The publication boundary's uint8 frames equal the released host-side conversion (CPU tensor; the CUDA case is integration)."""

import numpy as np
import pytest
import torch

from fdanyone.model.vae import _rgb_frames


def _released(video: torch.Tensor) -> list[np.ndarray]:
    scaled = video.detach().float().add_(1.0).mul_(127.5).clamp_(0.0, 255.0).to(device="cpu")
    return [scaled[:, index].permute(1, 2, 0).numpy().astype(np.uint8) for index in range(scaled.shape[1])]


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=[pytest.mark.integration, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")])],
)
def test_rgb_frames_match_the_released_conversion(device: str) -> None:
    generator = torch.Generator().manual_seed(0)
    # [-1.2, 1.2] covers both clamps; exact .5 steps sit on the truncation boundaries.
    video = (torch.rand((3, 5, 16, 8), generator=generator) * 2.4 - 1.2).to(torch.bfloat16).to(device)
    video[0, 0, 0, :4] = torch.tensor([-1.0, 1.0, 1.0 / 127.5 - 1.0, 0.0])
    frames = _rgb_frames(video)
    expected = _released(video)
    assert len(frames) == 5
    assert all(a.shape == (16, 8, 3) and a.dtype == np.uint8 for a in frames)
    assert all(np.array_equal(a, b) for a, b in zip(frames, expected, strict=True))
