"""Gradio app: one video in, a MetricTrail run (``ours``, the ``video`` preset) streamed into an embedded Rerun viewer.

MetricTrail calibrates the camera itself (SharedK), so there is no intrinsics input. The run is
:func:`metrictrail.apis.run.run_and_log`, the code path of ``metrictrail-run``, with the default MoGe-2 provider
(TensorRT ``moge-2-vitl-normal``, torch FP16 when TensorRT does not import).

Streaming: ``@rr.thread_local_stream`` gives each call its own recording, ``rr.binary_stream()`` collects its bytes and
the generator yields ``(stream.read(), status)``. The pipeline runs in a worker thread and logs into that recording
explicitly; the capture hooks (:mod:`metrictrail.capture`, ``live`` mode) log every tracked frame as DPVO tracks it, and
the generator yields what has accumulated twice a second. The whole input video is logged first, from its own samples
(``rr.VideoStream``); the viewer waits paused on it during calibration (which has no hook; the status line says what
runs), then the generator moves the viewer's cursor to each newly tracked frame (``gradio_rerun.commands.set_time``), so
DPVO's patches, depth, normals and DINOv3 PCA show as MetricTrail computes them. The trajectory and the final mesh come
at the end, from the artifacts, and the clip then plays from the start in a loop.
"""

import threading
import time
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import gradio as gr
import numpy as np
import rerun as rr
from beartype.roar import BeartypeException
from gradio_rerun import Rerun
from gradio_rerun.commands import set_time

from metrictrail import entities
from metrictrail.apis.run import RunResult, RunSettings, run_and_log
from metrictrail.benchmarks import UPSTREAM_MAX_FRAMES
from metrictrail.blueprint import app_blueprint
from metrictrail.outputs import RunMetrics

APPLICATION_ID: str = "metrictrail"
"""Application id of every streamed recording."""
DATA_DIR: Path = Path(__file__).resolve().parents[2] / "data"
"""``packages/metrictrail/data``: the example clips (``metrictrail-app`` downloads them)."""
VIDEO_TYPES: list[str] = [".mp4", ".mov", ".MOV", ".webm"]
"""Upload types the file picker accepts."""
POLL_S: float = 0.5
"""How often the generator yields the bytes logged so far and moves the cursor to the newest tracked frame."""

RUN_LOCK: threading.Lock = threading.Lock()
"""Held while a pipeline runs: the capture hooks patch ViPE's classes, so two runs must never overlap (a cancelled
run's worker can still be finishing when the next click arrives)."""


class RunCancelled(RuntimeError):
    """Raised from the capture hook when the user stops the run."""


@dataclass(slots=True)
class RunStatus:
    """What the worker reports to the generator."""

    text: str = "starting"
    """Latest status line."""
    cancel: threading.Event = field(default_factory=threading.Event)
    """Set when the generator stops; the next stage or tracked frame raises :class:`RunCancelled`."""
    cursor_ns: int | None = None
    """``video_time`` of the newest tracked frame, ns; None before tracking starts."""
    sent_cursor_ns: int | None = None
    """The cursor the viewer was last moved to."""

    def update(self, text: str) -> None:
        """The ``on_stage`` callback of :func:`run_and_log`."""
        if self.cancel.is_set():
            raise RunCancelled("stopped from the app")
        self.text = text

    def tracked_at(self, video_time_ns: int) -> None:
        """The ``on_cursor`` callback of :func:`run_and_log`."""
        self.cursor_ns = video_time_ns

    def cursor_command(self) -> dict[str, object] | None:
        """A viewer update that moves the cursor to the newest tracked frame (paused), or None when it has not moved."""
        if self.cursor_ns is None or self.cursor_ns == self.sent_cursor_ns:
            return None
        self.sent_cursor_ns = self.cursor_ns
        return dict(set_time(entities.VIDEO_TIMELINE, duration=np.timedelta64(self.cursor_ns, "ns"), play=False))


def example_videos(data_dir: Path) -> list[Path]:
    """The example clips under ``data_dir``: the iPhone ones (``iphone/``), then dpvo's ``movies/``."""
    return sorted((data_dir / "iphone").glob("*.mp4")) + sorted((data_dir / "movies").glob("*.mp4"))


def app_overrides(tracking_stride: int, max_frames: int) -> list[str]:
    """Hydra overrides of the two knobs; none at upstream's values (stride 1, :data:`UPSTREAM_MAX_FRAMES`)."""
    if tracking_stride < 1 or max_frames < 2:
        raise ValueError(f"tracking stride must be >= 1 and max frames >= 2, got {tracking_stride} and {max_frames}")
    overrides: list[str] = []
    if tracking_stride != 1:
        overrides.append(f"pipeline.init.tracking_stride={tracking_stride}")
    if max_frames != UPSTREAM_MAX_FRAMES:
        overrides.append(f"pipeline.init.input_raster.max_frames={max_frames}")
    return overrides


def summary(metrics: RunMetrics, wall_s: float) -> str:
    """The final status line."""
    mesh: str = f", TSDF mesh of {metrics.mesh_triangles:,} triangles" if metrics.mesh_triangles is not None else ""
    cloud: str = f", {metrics.cloud_points:,} cloud points" if metrics.cloud_points is not None else ""
    return f"done in {wall_s:.0f} s: {metrics.num_tracked} frames tracked ({metrics.num_finite} finite poses){mesh}{cloud}"


def run_locked(settings: RunSettings, rec: rr.RecordingStream, status: RunStatus) -> RunResult:
    """:func:`run_and_log` in live mode; releases :data:`RUN_LOCK` (taken by the caller) when it returns."""
    try:
        return run_and_log(settings, rec, live=True, on_stage=status.update, on_cursor=status.tracked_at)
    finally:
        RUN_LOCK.release()


@rr.thread_local_stream(APPLICATION_ID)
def metrictrail_streaming_fn(
    video_path: str | None, tracking_stride: int | float, max_frames: int | float
) -> Generator[tuple[bytes | dict[str, object], str], None, None]:
    """Run MetricTrail on ``video_path`` and stream the recording to the viewer.

    Args:
        video_path: The uploaded video (a local path); None before an upload.
        tracking_stride: DPVO tracks every k-th prepared frame (``pipeline.init.tracking_stride``).
        max_frames: Longer videos are strided down to at most this many frames (``pipeline.init.input_raster.max_frames``).

    Yields:
        The Rerun bytes logged since the last yield (or a cursor command for the viewer), and a status line.
    """
    if video_path is None:
        yield b"", "Upload a video or pick an example first."
        return
    overrides: list[str] = app_overrides(int(tracking_stride), int(max_frames))
    if not RUN_LOCK.acquire(blocking=False):
        yield b"", "The previous run is still finishing; try again in a moment."
        return
    stream: rr.BinaryStream = rr.binary_stream()
    rec: rr.RecordingStream | None = rr.get_thread_local_data_recording()
    if rec is None:
        RUN_LOCK.release()
        raise RuntimeError("rr.thread_local_stream set no recording")
    rec.send_blueprint(app_blueprint(None), make_active=True)
    status: RunStatus = RunStatus()
    started: float = time.perf_counter()
    with TemporaryDirectory(prefix="metrictrail-app-") as run_dir, ThreadPoolExecutor(max_workers=1, thread_name_prefix="metrictrail") as pool:
        settings: RunSettings = RunSettings(input=Path(video_path), output=Path(run_dir), benchmark="video", extra_overrides=overrides)
        try:
            future: Future[RunResult] = pool.submit(run_locked, settings, rec, status)
        except BaseException:
            RUN_LOCK.release()
            raise
        try:
            while True:
                try:
                    result: RunResult = future.result(timeout=POLL_S)
                    break
                except FutureTimeoutError:
                    line: str = f"{status.text} ({time.perf_counter() - started:.0f} s)"
                    yield stream.read() or b"", line
                    command: dict[str, object] | None = status.cursor_command()
                    if command is not None:
                        yield command, line
        except BeartypeException:
            raise
        except RunCancelled:
            yield stream.read() or b"", "Stopped."
            return
        except Exception as error:
            raise gr.Error(f"MetricTrail failed: {error}") from error
        finally:
            # Stop button or closed tab: the worker stops at its next stage or tracked frame; the pool waits for it.
            status.cancel.set()
    rec.send_blueprint(app_blueprint(result.logged.world_eye), make_active=True)
    done: str = summary(result.logged.metrics, time.perf_counter() - started)
    yield stream.read() or b"", done
    yield dict(set_time(entities.VIDEO_TIMELINE, duration=np.timedelta64(0, "ns"), play=True)), done


RunFn = Callable[[str | None, int | float, int | float], Iterator[tuple[bytes | dict[str, object], str]]]
"""``(video_path, tracking_stride, max_frames) -> (Rerun bytes, status)`` per yield: :func:`metrictrail_streaming_fn`
or a host's wrapper of it (a ZeroGPU Space runs it in a ``@spaces.GPU`` worker)."""


def build_app(
    data_dir: Path = DATA_DIR, *, run: RunFn = metrictrail_streaming_fn, examples: list[Path] | None = None, max_frames: int = UPSTREAM_MAX_FRAMES
) -> gr.Blocks:
    """The MetricTrail app: video upload, knobs, examples and status on the left; the streaming viewer on the right.

    Args:
        data_dir: Where :func:`example_videos` finds the example clips (when ``examples`` is None).
        run: What the Run button calls; a host wraps :func:`metrictrail_streaming_fn` (a GPU reservation) and patches nothing.
        examples: The example clips; None: :func:`example_videos` of ``data_dir``.
        max_frames: Default of the max-frames field (a host caps the work a click reserves).
    """
    clips: list[Path] = examples if examples is not None else example_videos(data_dir)
    viewer: Rerun = Rerun(
        streaming=True,
        panel_states={"time": "collapsed", "blueprint": "collapsed", "selection": "collapsed"},
        height=800,
    )
    with gr.Blocks(title="MetricTrail") as demo:
        gr.Markdown(
            "# MetricTrail\nMonocular metric SLAM (ViPE `pipeline=metrictrail`): DPVO with MoGe-2 metric depth and DINOv3 "
            "features. Your video plays at once; the camera is calibrated from it (about 40 s), then tracking streams in over "
            "the video, with a TSDF mesh growing from each keyframe; the trajectory and the final mesh arrive at the end."
        )
        with gr.Row():
            with gr.Column(scale=1):
                video_input: gr.File = gr.File(file_count="single", file_types=VIDEO_TYPES, label="Input video")
                with gr.Row():
                    run_button: gr.Button = gr.Button("Run MetricTrail", variant="primary")
                    stop_button: gr.Button = gr.Button("Stop")
                status_text: gr.Textbox = gr.Textbox(label="Status", value="Upload a video to begin.", interactive=False)
                with gr.Accordion("Settings", open=False):
                    stride_slider: gr.Slider = gr.Slider(
                        label="Tracking stride (DPVO tracks every k-th frame)", minimum=1, maximum=5, step=1, value=1
                    )
                    max_frames_number: gr.Number = gr.Number(
                        label="Max frames (a longer video is strided down to this many)", value=max_frames, precision=0, minimum=2
                    )
                gr.Examples(examples=[[str(path)] for path in clips], inputs=[video_input], cache_examples=False)
            with gr.Column(scale=5):
                viewer.render()
        run_event = run_button.click(
            fn=run, inputs=[video_input, stride_slider, max_frames_number], outputs=[viewer, status_text]
        )
        stop_button.click(fn=None, inputs=None, outputs=None, cancels=[run_event])
    return demo
