"""CPU-only recipe and metric oracles for the direct-gsplat port."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest
import torch

from gauss_surf.train_gsplat.core import per_frame_average_psnr

PART6_REFINE_STEPS: tuple[int, ...] = (
    1_000,
    1_100,
    1_200,
    1_300,
    1_400,
    1_500,
    1_600,
    1_700,
    1_800,
    1_900,
    2_000,
    2_100,
    2_200,
    2_300,
    2_400,
    2_500,
    2_600,
    2_700,
    2_800,
    2_900,
    4_000,
    4_100,
    4_200,
    4_300,
    4_400,
    4_500,
    4_600,
    4_700,
    4_800,
    4_900,
    5_000,
    5_100,
    5_200,
    5_300,
    5_400,
)
"""All 35 actual split/duplicate steps printed by the Part 6 run."""

PART10_REFINE_STEPS: tuple[int, ...] = (
    1_400,
    1_500,
    1_600,
    1_700,
    1_800,
    1_900,
    2_000,
    2_100,
    2_200,
    2_300,
    2_400,
    2_500,
    2_600,
    2_700,
    2_800,
    2_900,
    4_400,
    4_500,
    4_600,
    4_700,
    4_800,
    4_900,
    5_000,
    5_100,
    5_200,
    5_300,
    5_400,
)
"""The 27 refinements printed by both real 1,221-image reference runs."""


def _require_local_artifact(path: Path) -> Path:
    """Skip golden-data checks when the untracked local run is unavailable."""
    if not path.is_file():
        pytest.skip(f"local Gauss Surf artifact is unavailable: {path}")
    return path


def _part10_should_refine(step: int, *, num_train_data: int) -> bool:
    """Return the integer-only Part 10 refinement predicate."""
    warmup_length: int = 500
    refine_every: int = 100
    reset_every: int = 3_000
    refine_stop: int = 5_500
    pause_refine_after_reset: int = num_train_data + refine_every
    return (
        step > warmup_length
        and step < refine_stop
        and step % refine_every == 0
        and step % reset_every >= pause_refine_after_reset
    )


def _downscale_factor(step: int) -> int:
    """Return the historical quarter-to-half-to-full resolution factor."""
    return 2 ** max(2 - step // 3_000, 0)


def _sh_degree(step: int) -> int:
    """Return the historical progressive spherical-harmonic degree."""
    return min(step // 1_000, 3)


def test_part6_training_log_contains_the_exact_35_refinement_steps() -> None:
    """The checked-in real run, not a reconstructed config, owns this golden."""
    training_log: Path = _require_local_artifact(
        Path(__file__).parents[1] / "data/splat_runs/47115416-gaussurf/gaussurf-arkit/part6/training.log"
    )
    refine_pattern: re.Pattern[str] = re.compile(
        r"^Step (\d+): \d+ GSs duplicated, \d+ GSs split\.", re.MULTILINE
    )

    parsed_steps: tuple[int, ...] = tuple(
        int(match) for match in refine_pattern.findall(training_log.read_text())
    )

    assert parsed_steps == PART6_REFINE_STEPS
    assert len(parsed_steps) == 35


def test_part10_and_part8_0b_logs_contain_the_same_27_refinement_steps() -> None:
    """The two real reference runs, not reconstructed config, own the golden."""
    run_root: Path = Path(__file__).parents[1] / "data/splat_runs/47115416-gaussurf/gaussurf-arkit"
    refine_pattern: re.Pattern[str] = re.compile(
        r"^Step (\d+): \d+ GSs duplicated, \d+ GSs split\.", re.MULTILINE
    )

    parsed_by_run: dict[str, tuple[int, ...]] = {
        run_name: tuple(
            int(match)
            for match in refine_pattern.findall(_require_local_artifact(run_root / run_name / "training.log").read_text())
        )
        for run_name in ("part10", "part8-0b")
    }

    assert parsed_by_run == {"part10": PART10_REFINE_STEPS, "part8-0b": PART10_REFINE_STEPS}


def test_part10_pause_after_reset_is_derived_from_train_frame_count() -> None:
    """Use all train images, correcting the old wide-only 570-image contract.

    Both `part10/training.log` and `part8-0b/training.log` print the same 27
    refinement events at 1400–2900 and 4400–5400. The prior 670-step pause
    predicted 41 events and therefore contradicted both empirical logs.
    """
    transforms_path: Path = _require_local_artifact(Path(__file__).parents[1] / "data/training_bundle_part10/47115416/transforms.json")
    transforms: dict[str, object] = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames: list[dict[str, object]] = transforms["frames"]  # type: ignore[assignment]
    chosen_frames: int = len(frames)
    holdout_frames: int = sum(bool(frame["holdout"]) for frame in frames)
    num_train_data: int = chosen_frames - holdout_frames
    refine_every: int = 100
    pause_refine_after_reset: int = num_train_data + refine_every

    assert chosen_frames == 1_302
    assert holdout_frames == 81
    assert num_train_data == 1_221
    assert pause_refine_after_reset == 1_321
    assert pause_refine_after_reset != 0
    assert tuple(step for step in range(7_000) if _part10_should_refine(step, num_train_data=num_train_data)) == PART10_REFINE_STEPS


def test_progressive_resolution_boundaries_are_exact() -> None:
    """Training is quarter scale before 3000, half before 6000, then full."""
    boundary_steps: tuple[int, ...] = (0, 2_999, 3_000, 5_999, 6_000, 6_999)

    factors: tuple[int, ...] = tuple(_downscale_factor(step) for step in boundary_steps)

    assert factors == (4, 4, 2, 2, 1, 1)


def test_spherical_harmonic_degree_progression_boundaries_are_exact() -> None:
    """SH degree rises at 1000, 2000, and 3000, then remains at degree three."""
    boundary_steps: tuple[int, ...] = (0, 999, 1_000, 1_999, 2_000, 2_999, 3_000, 6_999)

    degrees: tuple[int, ...] = tuple(_sh_degree(step) for step in boundary_steps)

    assert degrees == (0, 0, 1, 1, 2, 2, 3, 3)


def test_psnr_is_averaged_per_frame_not_computed_from_pooled_mse() -> None:
    """Pin the accepted metric definition on synthetic images.

    Full checkpoint parity against ``evaluation.json`` belongs to migration
    stage 1. This pre-port oracle pins only the formula, including a case where
    averaging frame PSNR and converting pooled MSE differ by much more than
    0.1 dB.
    """
    prediction_nchw: torch.Tensor = torch.zeros(
        (2, 1, 2, 2), dtype=torch.float64, device="cpu"
    )
    target_nchw: torch.Tensor = torch.stack(
        (
            torch.full((1, 2, 2), 0.1, dtype=torch.float64, device="cpu"),
            torch.full((1, 2, 2), 0.5, dtype=torch.float64, device="cpu"),
        )
    )

    per_frame_psnr: torch.Tensor = per_frame_average_psnr(
        prediction_nchw, target_nchw
    )
    pooled_mse: torch.Tensor = (prediction_nchw - target_nchw).square().mean()
    pooled_psnr: torch.Tensor = -10.0 * torch.log10(pooled_mse)

    assert math.isclose(per_frame_psnr.item(), 13.010299956639813, abs_tol=1e-12)
    assert math.isclose(pooled_psnr.item(), 8.860566476931632, abs_tol=1e-12)
    assert abs(per_frame_psnr.item() - pooled_psnr.item()) >= 0.1
