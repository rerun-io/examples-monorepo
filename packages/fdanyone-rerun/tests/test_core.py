"""The run core: view layouts and the observers attached to upstream (the STEP_HOOKS seam of recipes/4danyone-src)."""

import logging
from dataclasses import dataclass
from typing import cast

import pytest
import torch

from fdanyone_rerun.core import ViewLayout, observers


def test_layout_counts_views_across_rings() -> None:
    assert ViewLayout().num_views == 6
    assert ViewLayout(views_per_layer=16, layer_pitches=(-10, 15, 35)).num_views == 48


@pytest.mark.parametrize("views_per_layer", [1, 4, 16])
def test_layout_rejects_counts_upstream_cannot_group(views_per_layer: int) -> None:
    with pytest.raises(ValueError, match="multiple of 6"):
        ViewLayout(views_per_layer=views_per_layer)


class _Scheduler:
    def __init__(self) -> None:
        self.sigmas: torch.Tensor = torch.tensor([0.9, 0.5])
        self.timesteps: torch.Tensor = self.sigmas * 1000.0

    def step(self, prediction: torch.Tensor, _timestep: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        return latents - 0.4 * prediction


class _Model:
    def __call__(self, **inputs: torch.Tensor) -> torch.Tensor:
        return torch.full_like(inputs["x"], 2.0)


@dataclass
class _Denoiser:
    model: _Model
    scheduler: _Scheduler
    dtype: torch.dtype = torch.float32


def _step(view_indices: tuple[int, ...] | None) -> torch.Tensor:
    from fdanyone.model.denoise import denoise_group
    from fdanyone.model.loader import Denoiser  # noqa: F401 (named by the cast)

    latents: torch.Tensor = torch.ones(2, 3)
    empty: torch.Tensor = torch.zeros(1)
    # A duck-typed stand-in for upstream's Denoiser: model, scheduler, dtype.
    denoiser = cast("Denoiser", _Denoiser(_Model(), _Scheduler()))
    return denoise_group(denoiser, latents, empty, empty, empty, empty, 0, view_indices=view_indices)


def test_step_hook_sees_the_clean_estimate_and_views() -> None:
    seen: list[tuple[int, tuple[int, ...] | None, torch.Tensor]] = []
    with observers(None, lambda step, views, clean: seen.append((step, views, clean))):
        stepped: torch.Tensor = _step((3, 4))
    assert torch.allclose(stepped, torch.full((2, 3), 0.2))
    assert len(seen) == 1
    step, views, clean = seen[0]
    assert (step, views) == (0, (3, 4))
    # x_0 = x_t - sigma * prediction = 1 - 0.9 * 2
    assert torch.allclose(clean, torch.full((2, 3), -0.8))


def test_observers_detach_after_an_error() -> None:
    from fdanyone.model.denoise import STEP_HOOKS

    stages: list[tuple[str, float]] = []
    progress: logging.Logger = logging.getLogger("fdanyone.progress")
    with pytest.raises(RuntimeError), observers(lambda label, fraction: stages.append((label, fraction)), lambda *_: None):
        progress.info("Recovering human motion with GVHMR", extra={"fraction": 0.15})
        raise RuntimeError("stop")
    progress.info("after", extra={"fraction": 1.0})
    assert stages == [("Recovering human motion with GVHMR", 0.15)]
    assert STEP_HOOKS == []
