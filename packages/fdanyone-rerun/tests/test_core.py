"""The run core: view layouts and the observers attached to upstream (the STEP_HOOKS seam of fdanyone.model.denoise)."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
import torch

from fdanyone.errors import FourDAnyoneError
from fdanyone_rerun.core import RunConfig, ViewLayout, observers


@pytest.mark.parametrize(
    ("layout", "num_views"),
    [(ViewLayout(), 6), (ViewLayout(views_per_layer=24), 24), (ViewLayout(views_per_layer=16, layer_pitches=(-10, 15, 35)), 48)],
)
def test_layout_counts_views_across_rings(layout: ViewLayout, num_views: int) -> None:
    assert layout.num_views == num_views


@pytest.mark.parametrize(
    ("layout", "run", "message"),
    [
        ({"views_per_layer": 16}, {}, "divisible by 6"),  # one 16-view ring: upstream groups views by six
        ({"views_per_layer": 4}, {}, "divisible by 6"),
        ({"layer_pitches": (99,)}, {}, "between -15 and 45"),
        ({"layer_pitches": (15, 15)}, {}, "duplicates"),
        ({"yaw_span": 0}, {}, "yaw_span"),
        ({}, {"seed": -1}, "seed"),
        ({}, {"start_time": -1.0}, "start_time"),
    ],
)
def test_invalid_runs_are_rejected_before_overwrite_deletes_the_output(tmp_path: Path, layout: dict, run: dict, message: str) -> None:
    output_dir: Path = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "metadata.json").write_text("{}")
    with pytest.raises(FourDAnyoneError, match=message):
        RunConfig(output_dir=output_dir, layout=ViewLayout(**layout), overwrite=True, **run)
    assert (output_dir / "metadata.json").is_file()


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

    stages: list[tuple[str, float, Path | None]] = []
    progress: logging.Logger = logging.getLogger("fdanyone.progress")
    with pytest.raises(RuntimeError), observers(lambda label, fraction, clip: stages.append((label, fraction, clip)), lambda *_: None):
        progress.info("Recovering human motion with GVHMR", extra={"fraction": 0.15})
        raise RuntimeError("stop")
    progress.info("after", extra={"fraction": 1.0})
    assert stages == [("Recovering human motion with GVHMR", 0.15, None)]
    assert STEP_HOOKS == []


def test_overwrite_keeps_the_old_run_until_the_new_one_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fdanyone import device, pipeline
    from fdanyone.errors import VideoContractError
    from fdanyone_rerun import core

    class Published(Exception):
        """Raised by the patched metadata read: the run finished and published."""

    output_dir: Path = tmp_path / "runs" / "example"
    output_dir.mkdir(parents=True)
    (output_dir / "metadata.json").write_text("old")
    monkeypatch.setattr(core, "gvhmr_root", lambda models: tmp_path)
    monkeypatch.setattr(device, "has_low_memory_gpu", lambda _devices: False)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "")

    def fails(**_kwargs: object) -> dict:
        Path(str(_kwargs["output_dir"])).mkdir()  # whatever a failed run leaves behind goes too
        raise VideoContractError("the clip is shorter than 121 frames")

    monkeypatch.setattr(pipeline, "run_pipeline", fails)
    with pytest.raises(VideoContractError):
        core.run_views(RunConfig(output_dir=output_dir, overwrite=True))
    assert (output_dir / "metadata.json").read_text() == "old"
    assert [path.name for path in output_dir.parent.iterdir()] == ["example"]

    def publishes(**kwargs: object) -> dict:
        new: Path = Path(str(kwargs["output_dir"]))
        new.mkdir()
        (new / "metadata.json").write_text("new")
        return {}

    def read(_output_dir: Path) -> core.RunMetadata:
        raise Published

    monkeypatch.setattr(pipeline, "run_pipeline", publishes)
    monkeypatch.setattr(core, "read_run_metadata", read)
    with pytest.raises(Published):
        core.run_views(RunConfig(output_dir=output_dir, overwrite=True))
    assert (output_dir / "metadata.json").read_text() == "new"
    assert [path.name for path in output_dir.parent.iterdir()] == ["example"]
