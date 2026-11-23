"""One run streamed into a Rerun recording, phase by phase: the CLI and the Gradio app both run through this.

1. Motion: the input clip and the progress log (``motion_blueprint``).
2. Conditioning: once GVHMR's motion is saved, the body in 3D and over the source (``body_blueprint``).
3. Denoising: per-view previews on ``diffusion_step`` from upstream's step hook (``diffusion_blueprint``).
4. Result: the camera rig and the generated videos (``result_blueprint``).
5. With a ``PersonConfig``: the 4D person trains on the views (``training_blueprint``: preview, loss, progress) and
   plays in 3D as Animated SOG (``person_blueprint``; ``fdanyone_rerun.person4d``).
"""

import threading
from pathlib import Path

import rerun as rr
import torch
from taehv import TAEHV

from fdanyone.assets import TAEW2_2
from fdanyone.errors import RunCancelled
from fdanyone.model.vae import load_tiny_decoder
from fdanyone_rerun.assets import model_dir
from fdanyone_rerun.core import RunConfig, run_views
from fdanyone_rerun.outputs import RunMetadata, read_camera_rig
from fdanyone_rerun.person4d import PersonConfig, train_person
from fdanyone_rerun.previews import decode_previews
from fdanyone_rerun.rerun_log import (
    DIFFUSION_TIMELINE,
    FRAME_TIMELINE,
    PROGRESS,
    TIMELINE,
    TRAIN_TIMELINE,
    body_blueprint,
    diffusion_blueprint,
    load_body,
    log_body,
    log_cameras,
    log_canonical_clip,
    motion_blueprint,
    preview_entity,
    result_blueprint,
)

CONDITIONING_STAGE: str = "Building foreground masks and skeletons"
"""Upstream's progress label once GVHMR's motion is saved in the run directory."""
GENERATION_STAGE: str = "Generating target-view videos"
"""Upstream's progress label when denoising starts."""


class RunStreamer:
    """Logs one run into ``recording`` as it happens."""

    def __init__(self, config: RunConfig, recording: rr.RecordingStream, previews: bool, person: PersonConfig | None = None) -> None:
        self.config: RunConfig = config
        self.recording: rr.RecordingStream = recording
        self.previews: bool = previews
        """Decode each denoising step's x_0 estimate with TAEW2.2 and log it."""
        self.person: PersonConfig | None = person
        """Train the 4D person after the views (None: views only)."""
        self.stop: threading.Event = threading.Event()
        """Set from another thread to cancel: a worker subprocess is stopped at once, the rest at the next stage or step."""
        self.output_dir: Path = config.output_dir.expanduser().resolve()

    def message(self, text: str, step: int = 0) -> None:
        """One progress line, in the log view and on stdout.

        A log view lists only rows on its active timeline, so every line sits at the start of ``time``, ``frame`` and
        ``train_step`` and at its denoising step: it shows in every phase's layout.
        """
        print(f"[fdanyone-rerun] {text}")
        self.recording.set_time(TIMELINE, duration=0.0)
        self.recording.set_time(FRAME_TIMELINE, sequence=0)
        self.recording.set_time(DIFFUSION_TIMELINE, sequence=step)
        self.recording.set_time(TRAIN_TIMELINE, sequence=0)
        self.recording.log(PROGRESS, rr.TextLog(text, level=rr.TextLogLevel.INFO))
        self.recording.reset_time()

    def on_stage(self, label: str, fraction: float, canonical_clip: Path | None) -> None:
        """Upstream's stage callback: log the canonical clip once it exists, and switch the layout at phase boundaries."""
        if self.stop.is_set():
            raise RunCancelled("stopped")
        self.message(f"{fraction:4.0%} {label}")
        if canonical_clip is not None:
            log_canonical_clip(canonical_clip, self.recording)
        if label == CONDITIONING_STAGE:
            fps: float = self._motion_fps()
            log_body(load_body(self.output_dir, model_dir()), fps, self.recording)
            self.recording.send_blueprint(body_blueprint())
        elif label == GENERATION_STAGE:
            self.recording.send_blueprint(diffusion_blueprint(self.config.layout.num_views))

    def on_step(self, step_index: int, view_indices: tuple[int, ...] | None, clean_estimate: torch.Tensor) -> None:
        """Upstream's step hook: decode and log each view's preview (the RCP proposals are not shown)."""
        if self.stop.is_set():
            raise RunCancelled("stopped")
        if view_indices is None or not self.previews:
            return
        # Upstream's cached TAEW2.2, shared with its target decode; loaded here, after run_views set the allocator policy.
        decoder: TAEHV = load_tiny_decoder(model_dir() / TAEW2_2, str(clean_estimate.device))
        self.message(f"denoising step {step_index + 1}: views {', '.join(f'{view:02d}' for view in view_indices)}", step=step_index)
        self.recording.set_time(DIFFUSION_TIMELINE, sequence=step_index)
        for view, image in zip(view_indices, decode_previews(decoder, clean_estimate), strict=True):
            self.recording.log(preview_entity(view), rr.Image(image).compress(jpeg_quality=85))
        self.recording.reset_time()

    def _motion_fps(self) -> float:
        """The canonical clip rate, from GVHMR's saved motion."""
        from fdanyone.motion.result import MotionResult

        return float(MotionResult.load(self.output_dir / "gvhmr").fps)

    def run(self) -> RunMetadata:
        """Run the pipeline and stream every phase; returns upstream's run summary."""
        self.recording.send_blueprint(motion_blueprint())
        self.recording.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
        metadata: RunMetadata = run_views(self.config, on_stage=self.on_stage, on_step=self.on_step, cancel=self.stop)
        log_cameras(read_camera_rig(self.output_dir), self.output_dir, self.recording)
        self.recording.send_blueprint(result_blueprint(self.config.layout.num_views))
        generation = metadata.generation
        self.message(
            f"views done in {generation.total_elapsed_seconds:.1f} s ({generation.attention_backend}, "
            f"{generation.peak_vram_reserved_bytes / 1e9:.2f} GB reserved)"
        )
        if self.person is not None:
            train_person(self.output_dir, self.person, self.recording, report=self.message, cancel=self.stop)
        return metadata
