"""Named pipeline presets: ``quality`` and ``fast``.

Both run the same resident streaming process (RAM-resident, TRT MammaNet, no
disk writes); they differ only in the operating point that trades artifact
fidelity against wall time. The numbers here are grounded in the 2026-06-10
dig over running_jumping and the gate encodings Pablo confirmed:

  * ``quality`` — every logged artifact within the tight band of the original
    DAG (kpt2d p95 <= 2%, tri3d p95 <= 2%H, SMPL-X PVE/trans/betas-disp within
    2%H, masks mean IoU >= 0.95 / p5 >= 0.90 / min >= 0.80); wall <= 25% of
    realtime is the secondary target. Uses the finer SAM2.1 hiera-small decoder
    (the 128-cell EfficientTAM-ti grid floors mask mean IoU near 0.93) and
    4K-sampled landmark crops.
  * ``fast`` — PVE p95/max <= 30 mm, SMPL-X within 5%H, masks mean IoU >= 0.90
    / min >= 0.70, at >= 75% of realtime. Keeps EfficientTAM-ti, trades a
    couple of fit iterations and the mask cadence for speed.

Both presets enable the causal mask-quality fixes (transient-collapse hold +
routine anchor re-prompts) since those correct genuine tracker failures
regardless of operating point.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mamma.fitting.window_fitter import FitterConfig
from mamma.tracking.tracker import TrackerConfig

PresetName = Literal["quality", "fast"]


@dataclass(frozen=True, slots=True)
class PipelinePreset:
    """A complete operating point for :func:`build_streaming_pipeline`."""

    name: PresetName
    """Preset identifier (also selects the validate_artifacts gate set)."""
    tracker: TrackerConfig
    """Tracker configuration (mask model + cadence)."""
    fitter: FitterConfig
    """Sliding-window SMPL-X fitter configuration."""
    resize_hw: tuple[int, int]
    """Engine resolution for tracking/logging."""
    hires_crops: bool
    """Decode native resolution and sample MammaNet crops from it (vs decoding
    directly at ``resize_hw``)."""
    realtime_fraction: float
    """PASS bound for the benchmark: wall <= clip_seconds / realtime_fraction."""


def quality_preset(expected_subjects: int | None = 1) -> PipelinePreset:
    """Highest-fidelity operating point (every artifact within the tight band).

    SAM2.1 hiera-small at image_size 1024 gives a 256-cell decoder grid (half
    the boundary-cell size of EfficientTAM-ti's 128 grid) plus stronger
    semantics — the lever the dig identified for clearing mask mean IoU >= 0.95.
    A 45-tick anchor cadence + transient hold remove the drift/collapse tails.
    """
    return PipelinePreset(
        name="quality",
        tracker=TrackerConfig(
            sam2_config="configs/sam2.1/sam2.1_hiera_s.yaml",
            sam2_checkpoint=Path("data/weights/sam2/sam2.1_hiera_small.pt"),
            expected_subjects=expected_subjects,
            redetect_interval=45,
            reprompt_alive_on_routine=True,
            transient_hold=True,
            track_stride=1,
        ),
        fitter=FitterConfig(emit_stride=1),
        resize_hw=(720, 1280),
        hires_crops=True,
        realtime_fraction=0.25,
    )


def fast_preset(expected_subjects: int | None = 1) -> PipelinePreset:
    """Throughput operating point (>= 75% realtime) holding the looser band.

    Keeps EfficientTAM-ti and the 4K-sampled crops that bring PVE under 30 mm,
    but lightens the fitter (24 tick iterations) and the mask cadence. The
    free mask fixes are still on so the min-IoU gate holds.
    """
    return PipelinePreset(
        name="fast",
        tracker=TrackerConfig(
            expected_subjects=expected_subjects,
            redetect_interval=45,
            reprompt_alive_on_routine=True,
            transient_hold=True,
            track_stride=1,
        ),
        fitter=FitterConfig(emit_stride=1, tick_iters=24),
        resize_hw=(720, 1280),
        hires_crops=True,
        realtime_fraction=0.75,
    )


def get_preset(name: PresetName, expected_subjects: int | None = 1) -> PipelinePreset:
    """Build a preset by name."""
    if name == "quality":
        return quality_preset(expected_subjects)
    if name == "fast":
        return fast_preset(expected_subjects)
    raise ValueError(f"unknown preset {name!r}")
