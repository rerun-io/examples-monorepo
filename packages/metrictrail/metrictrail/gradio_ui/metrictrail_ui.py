"""Gradio app: one video in, a MetricTrail run (``ours``, the ``video`` preset) streamed into an embedded Rerun viewer.

MetricTrail calibrates the camera itself (SharedK), so there is no intrinsics input. The run is
:func:`metrictrail.apis.run.run_and_log`, the code path of ``metrictrail-run``, with the default MoGe-2 provider
(TensorRT ``moge-2-vitl-normal``, torch FP16 when TensorRT does not import).

Streaming: ``@rr.thread_local_stream`` gives each call its own recording, ``rr.binary_stream()`` collects its bytes and
the generator yields ``(stream.read(), status, summary, run log)`` (:data:`RunYield`). The pipeline runs in a worker thread and logs into that recording
explicitly; the capture hooks (:mod:`metrictrail.capture`, ``live`` mode) log every tracked frame as DPVO tracks it, and
the generator yields what has accumulated twice a second. The whole input video is logged first, from its own samples
(``rr.VideoStream``). While SharedK calibrates, the viewer shows the calibration layout
(:func:`metrictrail.blueprint.calibration_blueprint`) and the generator moves its cursor along ``calibration_step`` to each
frame the sweep reads (``gradio_rerun.commands.set_time``); once DPVO tracks, it switches to the run layout and plays
the video at real time behind the tracked frames (:class:`Playback`), so DPVO's patches, depth, normals and DINOv3 PCA
play smoothly over it. The trajectory and the final mesh come
at the end, from the artifacts, and the clip plays on from where it is, in a loop.

The page has the EgoHand app's look (:mod:`metrictrail.gradio_ui.theme`): Input and Outputs tabs, Run and the status on
the left, the viewer filling the rest of the window.
"""

import threading
import time
from collections import deque
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypeAlias

import gradio as gr
import numpy as np
import rerun as rr
import torch
from beartype.roar import BeartypeException
from gradio_rerun import Rerun
from gradio_rerun.commands import set_time
from vipe.metrictrail.calibration.config import SharedKCalibrationConfig
from vipe.metrictrail.calibration.policy import PIPELINE_BUDGETS
from vipe.utils.model_cache import ModelCache

from metrictrail import entities
from metrictrail.apis.run import RunResult, RunSettings, run_and_log
from metrictrail.benchmarks import UPSTREAM_MAX_FRAMES
from metrictrail.blueprint import calibration_blueprint, run_blueprint
from metrictrail.calibration_live import RunPhase
from metrictrail.examples import DATA_DIR, example_videos
from metrictrail.gradio_ui.theme import APP_CSS, DARK_HEAD, RERUN_LOGOS
from metrictrail.outputs import RunMetrics

APPLICATION_ID: str = "metrictrail"
"""Application id of every streamed recording."""
RUN_LOG_LINES: int = 100
"""Lines the run log keeps (the textbox is sent whole when it changes)."""
READY: str = "Upload a clip, or pick an example, then press Run."
DESCRIPTION: str = (
    f'<h1 id="app-title"><a href="https://rerun.io" target="_blank" rel="noopener" title="rerun.io">'
    f'<img class="logo-light" src="{RERUN_LOGOS["light"]}" alt="Rerun"><img class="logo-dark" src="{RERUN_LOGOS["dark"]}" alt="Rerun">'
    f'</a> <span class="title-x">×</span> MetricTrail</h1>\n\n'
    "Metric camera trajectory and surface from one phone video: ViPE's `pipeline=metrictrail`, DPVO with MoGe-2 metric depth "
    "and DINOv3 features. **Pick an example or upload a clip, then press Run:** the camera's focal is calibrated from the video "
    "first (SharedK, 15-30 s), then the tracking, MoGe-2's depth and normals and a TSDF mesh stream in as the video plays."
)
"""Markdown above the app: Rerun's wordmark, then two short lines."""
POLL_S: float = 0.5
"""How often the generator yields the bytes logged so far and moves the cursor to the newest tracked frame."""

UPSTREAM_BUDGET: str = SharedKCalibrationConfig.model_fields["budget"].default
"""ViPE's in-the-wild default."""
APP_BUDGET: str = "fast64"
"""The app's default: on pool.mp4 it reads 413 frames for 64 keyframes, 15 s instead of 29 s for robust128's 840 / 128,
with a focal 0.5 % from robust128's (both converged). A clip too short for 128 keyframes gains little (IMG_0493: 16 s,
focal within 0.1 %)."""
BUDGET_LABELS: dict[str, str] = {
    "fast64": "fast64: 64 keyframes (about half the time)",
    "robust128": "robust128: 128 keyframes (ViPE's default)",
    "robust256": "robust256: up to 256 keyframes (stops at 128 if converged)",
}
"""What the dropdown shows for each budget."""
RUN_LOCK: threading.Lock = threading.Lock()
"""Held while a pipeline runs: the capture hooks patch ViPE's classes, so two runs must never overlap (a cancelled
run's worker can still be finishing when the next click arrives)."""
MODELS: ModelCache = ModelCache()
"""The app's networks (MoGe-2 engines, DROID, DINO-DPVO): the first run builds them, every later run reuses them."""


class RunCancelled(BaseException):
    """Raised from the run's hooks when the user stops the run. Not an ``Exception``: ViPE's calibration turns every
    ``Exception`` into its MoGe-2 focal fallback and goes on to track."""


@dataclass(slots=True)
class Playback:
    """The input video plays at real time while DPVO tracks ahead of it, instead of the cursor jumping to each newly
    tracked frame (tracking runs several times faster than real time on a desktop GPU, so the jumps flew by). It starts
    once :attr:`resume_s` of video is tracked, waits where it is when it comes within :attr:`lead_s` of the tracked
    frames (a slower GPU), and goes on once :attr:`resume_s` more is tracked. The viewer keeps its own clock: this only
    estimates where the playhead is, to know when to pause."""

    speed: float = 1.0
    """Playback speed (the viewer's, 1 = real time)."""
    lead_s: float = 1.0
    """Pause when the playhead comes this close to the newest tracked frame, seconds."""
    resume_s: float = 2.0
    """Play (again) once this much video past the playhead is tracked, seconds."""
    origin_ns: int = 0
    """``video_time`` where the viewer last started playing."""
    origin_wall: float | None = None
    """``time.perf_counter()`` when it did; None while paused (or not started)."""
    paused_ns: int = 0
    """Where it waits while paused."""

    def position_ns(self, now: float) -> int:
        """Where the viewer's playhead is now (estimated)."""
        if self.origin_wall is None:
            return self.paused_ns
        return self.origin_ns + int((now - self.origin_wall) * self.speed * 1e9)

    def update(self, frontier_ns: int, now: float) -> dict[str, object] | None:
        """The viewer command for a poll that found ``frontier_ns`` (the newest tracked frame's ``video_time``), or None."""
        if self.origin_wall is None:
            if frontier_ns - self.paused_ns < self.resume_s * 1e9:
                return None
            self.origin_ns, self.origin_wall = self.paused_ns, now
            return dict(set_time(entities.VIDEO_TIMELINE, duration=np.timedelta64(self.origin_ns, "ns"), play=True))
        position: int = self.position_ns(now)
        if position <= frontier_ns - self.lead_s * 1e9:
            return None
        self.paused_ns, self.origin_wall = min(position, frontier_ns), None
        return dict(set_time(entities.VIDEO_TIMELINE, duration=np.timedelta64(self.paused_ns, "ns"), play=False))

    def finish(self, now: float) -> dict[str, object]:
        """Every frame is in: play on from where the playhead is (the final layout would otherwise restart the clip)."""
        return dict(set_time(entities.VIDEO_TIMELINE, duration=np.timedelta64(self.position_ns(now), "ns"), play=True))


@dataclass(slots=True)
class RunStatus:
    """What the worker reports to the generator."""

    text: str = "starting"
    """Latest status line."""
    cancel: threading.Event = field(default_factory=threading.Event)
    """Set when the generator stops; the next stage or tracked frame raises :class:`RunCancelled`."""
    cursor: tuple[int, RunPhase] | None = None
    """Where the newest frame is and which stage moved the cursor there: the calibration's ``calibration_step``, or
    DPVO's ``video_time`` in ns; None before the first."""
    sent: tuple[int, RunPhase] | None = None
    """The calibration cursor the viewer was last moved to."""
    playback: Playback = field(default_factory=Playback)
    """How the video plays while DPVO tracks."""

    def update(self, text: str) -> None:
        """The ``on_stage`` callback of :func:`run_and_log`."""
        if self.cancel.is_set():
            raise RunCancelled("stopped from the app")
        self.text = text

    def at(self, value: int, phase: RunPhase) -> None:
        """The ``on_cursor`` callback of :func:`run_and_log`."""
        self.cursor = (value, phase)

    def cursor_command(self, now: float) -> dict[str, object] | None:
        """The viewer update for this poll, or None: during calibration the cursor moves (paused) to the newest frame
        the sweep read; while DPVO tracks, :attr:`playback` plays the video behind the tracked frames."""
        if self.cursor is None:
            return None
        value, phase = self.cursor
        if phase == "tracking":
            return self.playback.update(value, now)
        if self.cursor == self.sent:
            return None
        self.sent = self.cursor
        return dict(set_time(entities.CALIBRATION_TIMELINE, sequence=value, play=False))


def app_overrides(tracking_stride: int, max_frames: int, budget: str) -> list[str]:
    """Hydra overrides of the three knobs; none at upstream's values (stride 1, :data:`UPSTREAM_MAX_FRAMES`,
    :data:`UPSTREAM_BUDGET`).

    Raises:
        ValueError: For a stride below 1, fewer than 2 frames, or a budget a MetricTrail pipeline cannot select.
    """
    if tracking_stride < 1 or max_frames < 2:
        raise ValueError(f"tracking stride must be >= 1 and max frames >= 2, got {tracking_stride} and {max_frames}")
    if budget not in PIPELINE_BUDGETS:
        raise ValueError(f"calibration budget must be one of {PIPELINE_BUDGETS}, got {budget!r}")
    overrides: list[str] = []
    if tracking_stride != 1:
        overrides.append(f"pipeline.init.tracking_stride={tracking_stride}")
    if max_frames != UPSTREAM_MAX_FRAMES:
        overrides.append(f"pipeline.init.input_raster.max_frames={max_frames}")
    if budget != UPSTREAM_BUDGET:
        overrides.append(f"pipeline.init.calibration.budget={budget}")
    return overrides


def summary(metrics: RunMetrics, wall_s: float) -> str:
    """The final status line."""
    mesh: str = f", TSDF mesh of {metrics.mesh_triangles:,} triangles" if metrics.mesh_triangles is not None else ""
    return f"done in {wall_s:.0f} s: {metrics.num_tracked} frames tracked ({metrics.num_finite} finite poses){mesh}"


def summary_table(metrics: RunMetrics, wall_s: float, budget: str) -> str:
    """The Outputs tab's table of a finished run."""
    rows: list[tuple[str, str]] = [
        ("Frames tracked", f"{metrics.num_tracked:,} ({metrics.num_finite:,} finite poses)"),
        ("TSDF mesh", f"{metrics.mesh_triangles:,} triangles" if metrics.mesh_triangles is not None else "-"),
        ("Elapsed", f"{wall_s:.1f} s"),
        ("Calibration budget", f"`{budget}`"),
    ]
    return "| Run | |\n|---|---|\n" + "".join(f"| {name} | {value} |\n" for name, value in rows)


@dataclass(slots=True)
class RunLog:
    """The Outputs tab's run log: one line per stage (a status's text before its first colon), with the seconds since
    Run, newest first. A stage's later statuses (its frame counts) show in the status box only."""

    started: float
    """``time.perf_counter()`` at Run."""
    lines: deque[str] = field(default_factory=lambda: deque(maxlen=RUN_LOG_LINES))
    """The recent lines, newest first."""
    stage: str | None = None
    """The stage of the newest line."""

    def note(self, text: str, now: float) -> str | dict[str, object]:
        """The textbox value after ``text`` (a new line), or ``gr.skip()`` when ``text`` is of the newest line's stage."""
        stage: str = text.split(":", 1)[0]
        if stage == self.stage:
            return gr.skip()
        self.stage = stage
        self.lines.appendleft(f"{now - self.started:6.1f} s  {text}")
        return "\n".join(self.lines)


Update: TypeAlias = str | dict[str, object]
"""A Markdown or Textbox value, or ``gr.skip()`` (no change)."""
RunYield: TypeAlias = tuple[bytes | dict[str, object], Update, Update, Update]
"""One step of a run: Rerun bytes (or a viewer command), the status box, the Outputs summary and the run log."""


@rr.thread_local_stream(APPLICATION_ID)
def metrictrail_streaming_fn(
    video_path: str | None, tracking_stride: int | float, max_frames: int | float, budget: str = APP_BUDGET
) -> Generator[RunYield, None, None]:
    """Run MetricTrail on ``video_path`` and stream the recording to the viewer.

    Args:
        video_path: The uploaded video (a local path); None before an upload.
        tracking_stride: DPVO tracks every k-th prepared frame (``pipeline.init.tracking_stride``).
        max_frames: Longer videos are strided down to at most this many frames (``pipeline.init.input_raster.max_frames``).
        budget: SharedK calibration's keyframe budget (``pipeline.init.calibration.budget``, :data:`PIPELINE_BUDGETS`).

    Yields:
        The Rerun bytes logged since the last yield (or a cursor command for the viewer), the status line, the Outputs
        tab's summary and its run log (:data:`RunYield`).
    """
    skip: dict[str, object] = gr.skip()
    if video_path is None:
        yield b"", "Choose a video first.", skip, skip
        return
    overrides: list[str] = app_overrides(int(tracking_stride), int(max_frames), budget)
    status: RunStatus = RunStatus()
    started: float = time.perf_counter()
    log: RunLog = RunLog(started)
    # Before the lock: a stop at this yield closes the generator here, and nothing is held yet.
    yield b"", "Starting…", "Running…", log.note("starting", started)
    if not RUN_LOCK.acquire(blocking=False):
        yield b"", "The previous run is still finishing; try again in a moment.", skip, skip
        return
    # The generator owns the lock from here on: the pool's exit waits for the worker (a stop sets the cancel event
    # first), so the lock is released only when nothing runs, whatever fails or stops in between.
    try:
        stream: rr.BinaryStream = rr.binary_stream()
        rec: rr.RecordingStream | None = rr.get_thread_local_data_recording()
        if rec is None:
            raise RuntimeError("rr.thread_local_stream set no recording")
        rec.send_blueprint(calibration_blueprint(), make_active=True)
        tracking_layout: bool = False
        with TemporaryDirectory(prefix="metrictrail-app-") as run_dir, ThreadPoolExecutor(max_workers=1, thread_name_prefix="metrictrail") as pool:
            settings: RunSettings = RunSettings(input=Path(video_path), output=Path(run_dir), benchmark="video", extra_overrides=overrides)
            future: Future[RunResult] = pool.submit(
                run_and_log, settings, rec, live=True, on_stage=status.update, on_cursor=status.at, models=MODELS
            )
            try:
                while True:
                    try:
                        result: RunResult = future.result(timeout=POLL_S)
                        break
                    except FutureTimeoutError:
                        now: float = time.perf_counter()
                        line: str = f"{status.text} ({now - started:.0f} s)"
                        yield stream.read() or b"", line, skip, log.note(status.text, now)
                        if status.cursor is not None and status.cursor[1] == "tracking" and not tracking_layout:
                            rec.send_blueprint(run_blueprint("ours", has_capture=True, has_gt=False, has_video=True, has_calibration=True, tracking=True), make_active=True)
                            tracking_layout = True
                            # The layout (its time panel waits paused) must reach the viewer, and be ingested, before any play
                            # command: a play applied first is paused again by the layout. Commands wait for the next poll.
                            yield stream.read() or b"", skip, skip, skip
                            continue
                        command: dict[str, object] | None = status.cursor_command(time.perf_counter())
                        if command is not None:
                            yield command, skip, skip, skip
            except BeartypeException:
                raise
            except RunCancelled:
                yield stream.read() or b"", "Stopped.", "Stopped.", log.note("stopped", time.perf_counter())
                return
            except Exception as error:
                raise gr.Error(f"MetricTrail failed: {error}") from error
            finally:
                # Stop button or closed tab: the worker stops at its next stage or tracked frame; the pool waits for it.
                status.cancel.set()
    finally:
        RUN_LOCK.release()
    # The run's tensors are gone (the models stay in MODELS); give the allocator's cached blocks back to the GPU, which
    # the app shares with other jobs between clicks.
    torch.cuda.empty_cache()
    rec.send_blueprint(
        run_blueprint(
            "ours",
            has_capture=result.has_capture,
            has_gt=result.has_gt,
            has_video=result.has_video,
            has_calibration=result.has_calibration,
            world_eye=result.logged.world_eye,
            metric_eye=result.logged.metric_eye,
        ),
        make_active=True,
    )
    wall_s: float = time.perf_counter() - started
    done: str = summary(result.logged.metrics, wall_s)
    yield stream.read() or b"", done, summary_table(result.logged.metrics, wall_s, budget), log.note(done, time.perf_counter())
    yield status.playback.finish(time.perf_counter()), skip, skip, skip


RunFn = Callable[[str | None, int | float, int | float, str], Iterator[RunYield]]
"""``(video_path, tracking_stride, max_frames, budget) -> RunYield`` per yield: :func:`metrictrail_streaming_fn` or a
host's wrapper of it (a ZeroGPU Space runs it in a ``@spaces.GPU`` worker)."""


def build_app(
    data_dir: Path = DATA_DIR, *, run: RunFn = metrictrail_streaming_fn, examples: list[Path] | None = None, max_frames: int = UPSTREAM_MAX_FRAMES,
    example_url: str | None = None,
) -> gr.Blocks:
    """Input and Outputs tabs, Run, Stop and the status on the left; the streaming Rerun viewer on the right (the EgoHand
    app's layout). Serve it with :func:`launch` (its look comes with the launch).

    Args:
        data_dir: Where :func:`example_videos` finds the example clips (when ``examples`` is None).
        run: What the Run button calls; a host wraps :func:`metrictrail_streaming_fn` (a GPU reservation) and patches nothing.
        examples: The example clips; None: :func:`example_videos` of ``data_dir``.
        max_frames: Default of the max-frames field (a host caps the work a click reserves).
        example_url: The example's page on rerun.io (full walkthrough), linked at the header's right; None links nothing.
    """
    clips: list[Path] = examples if examples is not None else example_videos(data_dir)
    # fill_width: Gradio caps the page at 1,536 px otherwise, and the viewer would not take the window's width. fill_height
    # with the main row's scale=1: the row takes the height the header leaves (theme.APP_CSS), so the page fits the window.
    with gr.Blocks(title="Rerun × MetricTrail", fill_width=True, fill_height=True) as demo:
        gr.Markdown(DESCRIPTION, elem_id="app-description")
        if example_url is not None:
            gr.HTML(f'<a href="{example_url}" target="_blank" rel="noopener">Full example and walkthrough on rerun.io →</a>', elem_id="example-link")
        with gr.Row(elem_id="main-row", equal_height=False, scale=1):
            with gr.Column(scale=1, min_width=360, elem_id="left-column"):
                with gr.Tabs(selected="input") as tabs:
                    with gr.Tab("Input", id="input"):
                        # format=None: Gradio hands the upload over as it is (a .MOV stays a .MOV; prepare_input links it).
                        video_input: gr.Video = gr.Video(
                            value=str(clips[0]) if clips else None, label="Source clip", sources=["upload"], format=None, elem_id="source-video"
                        )
                        with gr.Accordion("Advanced settings", open=False):
                            stride_slider: gr.Slider = gr.Slider(
                                label="Tracking stride (DPVO tracks every k-th frame)", minimum=1, maximum=5, step=1, value=1
                            )
                            max_frames_number: gr.Number = gr.Number(
                                label="Max frames (a longer video is strided down to this many)", value=max_frames, precision=0, minimum=2
                            )
                            budget_dropdown: gr.Dropdown = gr.Dropdown(
                                label="Calibration budget (keyframes SharedK keeps before it fixes the focal)",
                                choices=[(BUDGET_LABELS[budget], budget) for budget in PIPELINE_BUDGETS],
                                value=APP_BUDGET,
                            )
                        gr.Examples(
                            examples=[[str(path)] for path in clips], inputs=[video_input], examples_per_page=4, cache_examples=False, elem_id="examples"
                        )
                    with gr.Tab("Outputs", id="outputs"):
                        run_summary: gr.Markdown = gr.Markdown("No run yet.", elem_id="run-summary")
                        run_log: gr.Textbox = gr.Textbox(
                            label="Run log (newest first)", lines=8, max_lines=8, autoscroll=False, interactive=False, elem_id="run-log"
                        )
                with gr.Row(elem_id="run-row"):
                    run_button: gr.Button = gr.Button("Run", variant="primary")
                    stop_button: gr.Button = gr.Button("Stop", variant="stop")
                status_text: gr.Markdown = gr.Markdown(READY, elem_id="run-status")
            with gr.Column(scale=3):
                viewer: Rerun = Rerun(
                    streaming=True, panel_states={"time": "collapsed", "blueprint": "hidden", "selection": "hidden"},
                    height="100%",
                    elem_id="viewer",
                )
        run_event = run_button.click(lambda: gr.Tabs(selected="outputs"), outputs=[tabs]).then(
            fn=run, inputs=[video_input, stride_slider, max_frames_number, budget_dropdown], outputs=[viewer, status_text, run_summary, run_log]
        )
        stop_button.click(lambda: "Stopped.", outputs=[status_text], cancels=[run_event], queue=False)
    return demo


def launch(demo: gr.Blocks, *, server_name: str | None = None, allowed_paths: list[str] | None = None, max_queue: int | None = 2) -> None:
    """Serve ``demo`` with the app's look (forced dark, like the EgoHand app); one run at a time, as the pipeline holds the
    GPU. ``server_name`` (None: ``GRADIO_SERVER_NAME``, else 127.0.0.1) and ``GRADIO_SERVER_PORT`` pick where; a host whose
    examples live outside the working directory names them in ``allowed_paths``. ``max_queue`` clicks may wait (None: any
    number, the public ZeroGPU Space, where each click reserves its own GPU)."""
    demo.queue(max_size=max_queue, default_concurrency_limit=1)
    demo.launch(server_name=server_name, allowed_paths=allowed_paths, ssr_mode=False, css=APP_CSS, head=DARK_HEAD, show_error=True)
