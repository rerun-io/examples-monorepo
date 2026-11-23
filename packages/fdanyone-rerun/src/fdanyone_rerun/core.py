"""The one run core the CLI and the app share: upstream 4DAnyone's pipeline with our observers attached.

Upstream reports coarse progress on the ``fdanyone.progress`` logger and, with the STEP_HOOKS seam in src/fdanyone, every
denoising group step through ``fdanyone.model.denoise.STEP_HOOKS``. :func:`run_views` attaches a stage callback and a
step callback for one run and always detaches them.
"""

import gc
import logging
import shutil
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

import torch

from fdanyone.errors import ConfigurationError
from fdanyone.video import validate_clip_options
from fdanyone.views import resolve_view_plan
from fdanyone_rerun.assets import EXAMPLE_VIDEO, gvhmr_root, model_dir
from fdanyone_rerun.outputs import RunMetadata, read_run_metadata

AttentionChoice: TypeAlias = Literal["auto", "sageattention", "sdpa", "flash_attn_3"]
StageCallback: TypeAlias = Callable[[str, float, Path | None], None]
"""``on_stage(label, fraction, canonical_clip)``: an upstream stage starts; ``fraction`` is its 0-1 position in the run;
``canonical_clip`` is the run's 121-frame clip (an mp4, frame k at k / fps) on the stage that publishes it, else None;
it lives in upstream's scratch directory, so read it inside the callback (it is deleted when the run ends)."""
StepCallback: TypeAlias = Callable[[int, tuple[int, ...] | None, torch.Tensor], None]
"""``on_step(step_index, view_indices, clean_estimate)``: one group finished a denoising step (STEP_HOOKS)."""


def free_gpu() -> tuple[float, float]:
    """Collect garbage and return cached CUDA blocks; returns (allocated, reserved) GB after."""
    gc.collect()
    torch.cuda.empty_cache()
    return torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9

@dataclass(frozen=True, slots=True)
class ViewLayout:
    """Where the generated cameras sit around the person."""

    views_per_layer: int = 6
    """Evenly spaced views on each pitch ring; the total must be divisible by six (upstream's group size)."""
    layer_pitches: tuple[int, ...] = (15,)
    """Ring pitch per layer, degrees, -15..45; positive is above the subject."""
    start_yaw: int = 0
    """Yaw of the first view on every ring, degrees; 0 faces the person."""
    yaw_span: int = 360
    """Yaw range of each ring, degrees, 1..360 (the end is excluded)."""

    def __post_init__(self) -> None:
        # Upstream's checks, when the layout is made: before any output is deleted or work starts.
        resolve_view_plan(
            views_per_layer=self.views_per_layer,
            layer_pitches=self.layer_pitches,
            start_yaw=self.start_yaw,
            yaw_span=self.yaw_span,
            enable_rcp=True,
            enable_tcr=True,
        )

    @property
    def num_views(self) -> int:
        """Total generated views."""
        return self.views_per_layer * len(self.layer_pitches)


@dataclass(frozen=True, slots=True)
class RunConfig:
    """One 4DAnyone run (4DAnyone-Turbo: four denoising steps)."""

    video: Path = EXAMPLE_VIDEO
    """Input video: one person, full or upper body, mild camera motion, at least 121 frames."""
    output_dir: Path = Path("data/runs/example")
    """Run directory (upstream's layout: videos/, skeletons/, cameras.json, metadata.json, gvhmr/)."""
    layout: ViewLayout = field(default_factory=ViewLayout)
    """The generated camera rig."""
    seed: int = 42
    """Noise seed."""
    start_time: float = 0.0
    """Clip start on the input timeline, seconds."""
    attention: AttentionChoice = "auto"
    """Attention backend; auto prefers FlashAttention-3, then SageAttention, then SDPA (SDPA on GPUs <= 24 GiB)."""
    overwrite: bool = False
    """Delete an existing run directory first (upstream refuses to write into one)."""

    def __post_init__(self) -> None:
        # Upstream's input checks, here so the CLI and the app reject a bad run before overwrite deletes anything.
        if self.seed < 0:
            raise ConfigurationError(f"seed must be non-negative, got {self.seed}.")
        validate_clip_options(start_time=self.start_time, fps=None)


class _ProgressHandler(logging.Handler):
    """Forwards upstream's ``fdanyone.progress`` records to a stage callback."""

    def __init__(self, on_stage: StageCallback) -> None:
        super().__init__(level=logging.INFO)
        self.on_stage: StageCallback = on_stage

    def emit(self, record: logging.LogRecord) -> None:
        fraction: float = float(getattr(record, "fraction", 0.0))
        clip: str | None = getattr(record, "canonical_clip", None)
        self.on_stage(record.getMessage(), fraction, None if clip is None else Path(clip))


@contextmanager
def observers(on_stage: StageCallback | None, on_step: StepCallback | None) -> Iterator[None]:
    """Attach the callbacks to upstream for the duration of the block."""
    from fdanyone.model.denoise import STEP_HOOKS

    progress: logging.Logger = logging.getLogger("fdanyone.progress")
    level: int = progress.level
    handler: _ProgressHandler | None = _ProgressHandler(on_stage) if on_stage is not None else None
    if handler is not None:
        # Upstream's records are INFO; the logger would otherwise inherit the root's WARNING outside run_pipeline.
        progress.setLevel(logging.INFO)
        progress.addHandler(handler)
    if on_step is not None:
        STEP_HOOKS.append(on_step)
    try:
        yield
    finally:
        if handler is not None:
            progress.removeHandler(handler)
            progress.setLevel(level)
        if on_step is not None:
            STEP_HOOKS.remove(on_step)


def run_views(
    config: RunConfig, *, on_stage: StageCallback | None = None, on_step: StepCallback | None = None, cancel: threading.Event | None = None
) -> RunMetadata:
    """Run upstream 4DAnyone on ``config`` and return the run's metadata.

    The run writes upstream's output directory at ``config.output_dir``. GVHMR motion of an earlier run in the same
    directory is reused only when ``overwrite`` is off and upstream accepts the directory.
    """
    from fdanyone.device import configure_inference_cuda_allocator, has_low_memory_gpu

    # As upstream's inference.py: the allocator policy must be set before the first CUDA allocation.
    low_memory: bool = has_low_memory_gpu(None)
    configure_inference_cuda_allocator(use_expandable_segments=low_memory)
    attention: str = "sdpa" if config.attention == "auto" and low_memory else config.attention

    from fdanyone.pipeline import run_pipeline

    output_dir: Path = config.output_dir.expanduser().resolve()
    # Overwrite moves the old run aside and deletes it only once this run has succeeded; a failed or stopped run puts
    # it back (upstream publishes its output only on success, so nothing of the failed run is in the way).
    previous: Path | None = None
    if config.overwrite and output_dir.exists():
        previous = output_dir.with_name(f".{output_dir.name}.previous-{uuid.uuid4().hex[:8]}")
        output_dir.rename(previous)
    models: Path = model_dir()
    try:
        with observers(on_stage, on_step):
            run_pipeline(
                video_path=str(config.video.expanduser().resolve()),
                output_dir=str(output_dir),
                model_dir=str(models),
                checkpoint_path=None,
                mhr70_regressor_path=None,
                gvhmr_root=str(gvhmr_root(models)),
                gpu_ids=None,
                attention_backend=attention,
                start_time=config.start_time,
                target_fps="auto",
                seed=config.seed,
                views_per_layer=config.layout.views_per_layer,
                layer_pitches=list(config.layout.layer_pitches),
                start_yaw=config.layout.start_yaw,
                yaw_span=config.layout.yaw_span,
                enable_rcp=True,
                enable_tcr=True,
                enable_turbo=True,
                cancel=cancel,
            )
    except BaseException:
        if previous is not None:
            shutil.rmtree(output_dir, ignore_errors=True)
            previous.rename(output_dir)
        raise
    if previous is not None:
        shutil.rmtree(previous)
    return read_run_metadata(output_dir)
