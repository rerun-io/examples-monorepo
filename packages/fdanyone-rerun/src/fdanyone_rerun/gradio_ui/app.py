"""The 4DAnyone app: pick a clip, press Run, watch the phases stream into the embedded Rerun viewer.

The run is :class:`fdanyone_rerun.stream.RunStreamer`, the same core as ``fdanyone-run --train``; this module only adds
the widgets, a worker thread and the byte pump into the viewer. The widgets start at a short training run with no MoGe-2
prior (:data:`APP_STEPS`, :data:`APP_PRIOR`); everything else is the CLI's default.

The page has our apps' shared look and layout (:mod:`simplecv.app_theme`): Input and Outputs tabs, Run, Stop and the status
on the left, the viewer filling the rest of the window. When the browser has no WebGPU, a notice above the viewer says so
(the gsplat renderer needs it).
"""

import dataclasses
import fcntl
import json
import os
import threading
import time
import traceback
import uuid
from collections import deque
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import gradio as gr
import rerun as rr
from gradio_rerun import Rerun
from simplecv.app_theme import APP_CSS, DARK_HEAD, EXAMPLES_PER_PAGE, app_title, run_table

from fdanyone.errors import RunCancelled
from fdanyone_rerun.assets import EXAMPLE_VIDEO, PACKAGE_ROOT
from fdanyone_rerun.core import RunConfig, ViewLayout, free_gpu
from fdanyone_rerun.person4d import PersonConfig, PersonSummary
from fdanyone_rerun.stream import RunStreamer

APPLICATION_ID: str = "fdanyone-rerun"
RUNS_DIR: Path = PACKAGE_ROOT / "data" / "app-runs"
"""One run directory per click (upstream's outputs), and beside it ``<run>.rrd``: the recording the viewer showed,
4D person (Animated SOG) included."""
PUMP_SECONDS: float = 0.5
"""How often new recording bytes go to the viewer."""
RUN_LOG_LINES: int = 100
"""Lines the run log keeps."""
APP_STEPS: int = 5_000
"""The steps setting's start value: a short training run (the CLI's default is :class:`PersonConfig`'s)."""
APP_PRIOR: bool = False
"""The prior setting's start value: off, so a click runs no extra estimator model (MoGe-2)."""
PRIOR_INFO: str = (
    "MoGe-2 depth and normals on every view shape the person's surface: on the 18-view example the held-out normal error "
    "falls from 30° to 16° at the same PSNR. Costs about 70 s more for 6 views on an RTX 5090 and up to 11 GB of GPU memory."
)
"""The prior setting's hint: what it adds and what it costs."""
VIEW_CHOICES: tuple[int, ...] = (6, 12, 18, 24)
"""Views on the ring (multiples of six, upstream's group size)."""
READY: str = "Upload a clip of one person (at least 121 frames), or pick an example, then press Run."
DESCRIPTION: str = (
    app_title("4DAnyone") + "\n\n"
    "One casual video of a person becomes synchronized videos from new cameras around them "
    "([4DAnyone](https://github.com/ant-research/4DAnyone)-Turbo), then a 4D Gaussian model of the person trains on those views "
    "([Open4DGS](https://github.com/ASU-NEM-Labs/open4dgs-trigs) TRiGS) and plays in 3D. **Pick an example or upload a clip, then "
    "press Run:** motion recovery, the views as they denoise, the camera rig, then the person training (about 6 min on an RTX 5090)."
)
"""Markdown above the app: Rerun's wordmark, then two short lines."""
NO_WEBGPU: str = (
    "<b>This browser has no WebGPU</b>, so the viewer cannot run the gsplat renderer: the 4D person is drawn with "
    "Rerun's basic splats and looks blurry. On Linux Chrome: enable <code>chrome://flags/#enable-vulkan</code> and "
    "relaunch. Or download the run's recording and open it in the native gsplat viewer "
    "(<code>packages/gsplat-rust-renderer/target/release/gsplat-rust-renderer &lt;run&gt;.rrd</code>)."
)
"""Shown above the viewer when the page finds no WebGPU adapter (the gsplat renderer needs WebGPU's compute limits)."""
WEBGPU_CHECK_JS: str = (
    "async () => { const adapter = navigator.gpu ? await navigator.gpu.requestAdapter().catch(() => null) : null;"
    " if (adapter) return; const block = document.getElementById('webgpu-notice-block');"
    " const notice = document.getElementById('webgpu-notice'); if (!block || !notice) return;"
    f" notice.innerHTML = {json.dumps(NO_WEBGPU)}; block.style.display = 'block'; }}"
)
"""Runs once when the page loads: fills ``#webgpu-notice`` and shows its block when ``navigator.gpu`` gives no adapter."""
NOTICE_CSS: str = """
#webgpu-notice-block { display: none; }
#webgpu-notice { padding: 0.6rem 0.8rem; margin-bottom: 0.4rem; border: 1px solid #b45309; border-radius: var(--radius-lg);
    background: #2a1a05; color: #fde68a; font-size: 0.9rem; }
"""
"""The WebGPU notice, on top of :data:`simplecv.app_theme.APP_CSS`; its block takes no room until the check shows it."""

Update: TypeAlias = str | dict[str, object]
"""A Markdown or Textbox value, or ``gr.skip()`` (no change)."""
RunYield: TypeAlias = tuple[bytes | None, Update, Update, Update]
"""One step of a run: Rerun bytes, the status box, the Outputs summary and the run log."""


@dataclass(frozen=True, slots=True)
class Config:
    """Serve the 4DAnyone app."""

    server_name: str = "127.0.0.1"
    """Interface on which Gradio listens."""
    server_port: int = 7873
    """Gradio HTTP port."""
    gpu_lock: Path | None = None
    """On a shared GPU host: a lock file held exclusively (flock) around each run, never while the app idles. Wrapping
    the whole app in ``flock`` would hold the GPU for the app's lifetime and block every other job."""


class _RunSlot:
    """One run on the GPU at a time. The slot is taken before a run's worker starts and released by that worker only
    when its run has returned: the worker subprocess reaped, the prefetch closed and the observers detached. A Stop
    ends the request at once, so the slot, not Gradio's queue, keeps the next run out until then."""

    def __init__(self) -> None:
        self.lock: threading.Lock = threading.Lock()
        self.worker: threading.Thread | None = None
        """The worker of the latest run; Stop reports "Stopping…" while it is alive."""
        self.gpu_lock: Path | None = None
        """``Config.gpu_lock``."""
        self.recording: Path | None = None
        """The latest finished run's saved recording (offered for download)."""
        self.stop: threading.Event | None = None
        """The latest run's stop event. Stop sets it itself: cancelling the Gradio event does not close the run's
        generator, so the generator's ``finally`` cannot be relied on to stop the worker."""


SLOT: _RunSlot = _RunSlot()


def example_videos() -> list[Path]:
    """The example clips ``fdanyone-download`` fetches, the README's first."""
    return sorted(EXAMPLE_VIDEO.parent.glob("*.mp4"), key=lambda path: path != EXAMPLE_VIDEO)


def summary_table(views: int, wall_s: float, person: PersonSummary | None, saved: Path) -> str:
    """The Outputs tab's table of a finished run."""
    rows: list[tuple[str, str]] = [("Views", f"{views}"), ("Elapsed", f"{wall_s:.0f} s")]
    if person is not None:
        rows += [
            ("4D person", f"{person.rows:,} Gaussians, {person.steps:,} steps at {person.it_per_s:.0f} it/s"),
            ("Animated SOG", f"{person.asog_bytes / 1e6:.1f} MB in the recording"),
        ]
    rows.append(("Recording", f"`{saved.name}`"))
    return run_table(rows)


def _release(gpu_lock: int | None) -> None:
    """Free the GPU flock (when held) and the run slot."""
    if gpu_lock is not None:
        os.close(gpu_lock)
    SLOT.lock.release()


def run_app(
    video: str | None, views_per_layer: int | float, seed: int | float, train: bool, steps: int | float, prior: bool
) -> Generator[RunYield, None, None]:
    """Run 4DAnyone on ``video`` (then train its 4D person) in a worker thread and pump the recording into the viewer
    until it ends; the status line shows the newest progress line, the run log all of them (newest first)."""
    skip: dict[str, object] = gr.skip()
    if video is None:
        yield None, "Choose a video first.", skip, skip
        return
    run_id: str = uuid.uuid4().hex[:12]
    config: RunConfig = RunConfig(
        video=Path(video), output_dir=RUNS_DIR / run_id, layout=ViewLayout(views_per_layer=int(views_per_layer)), seed=int(seed)
    )
    person: PersonConfig | None = dataclasses.replace(PersonConfig().with_steps(int(steps)), prior=prior) if train else None
    while not SLOT.lock.acquire(timeout=PUMP_SECONDS):
        yield b"", "Waiting for the previous run to stop…", skip, skip
    # The slot is ours: whatever fails before the worker owns it releases it (and the GPU flock) here.
    gpu_lock: int | None = None
    try:
        if SLOT.gpu_lock is not None:
            gpu_lock = os.open(SLOT.gpu_lock, os.O_RDWR | os.O_CREAT, 0o664)
            while True:
                try:
                    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    yield b"", "Waiting for the GPU (another job holds it)…", skip, skip
                    time.sleep(PUMP_SECONDS)
        SLOT.recording = None
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        saved: Path = RUNS_DIR / f"{run_id}.rrd"
        recording: rr.RecordingStream = rr.RecordingStream(APPLICATION_ID, recording_id=run_id)
        # Two sinks: the bytes the viewer gets, and the SDK's own file (appending stream reads would not be one valid .rrd).
        stream: rr.BinaryStream = recording.binary_stream()
        recording.set_sinks(rr.FileSink(str(saved)), stream)
        streamer: RunStreamer = RunStreamer(config, recording, previews=True, person=person)
    except BaseException:
        _release(gpu_lock)
        raise
    errors: list[BaseException] = []

    def work() -> None:
        try:
            streamer.run()
        except RunCancelled:
            pass  # a Stop; keeping it would keep its frames, the trainer's GPU memory among them, alive
        except BaseException as error:  # re-raised by the generator below, beartype violations included
            traceback.clear_frames(error.__traceback__)  # the trace stays; the frames' tensors go
            errors.append(error)
        finally:
            recording.disconnect()  # closes the saved .rrd, stopped runs included; the stream keeps its unread bytes
            free_gpu()  # a stopped or failed run leaves no cached GPU memory to the next job
            _release(gpu_lock)

    worker: threading.Thread = threading.Thread(target=work, name=f"fdanyone-{run_id}", daemon=True)
    started: float = time.perf_counter()
    try:
        worker.start()
    except BaseException:
        recording.disconnect()
        _release(gpu_lock)
        raise
    SLOT.worker = worker
    SLOT.stop = streamer.stop
    logged: int = 0
    lines_shown: deque[str] = deque(maxlen=RUN_LOG_LINES)

    def log_text() -> Update:
        """The run log textbox after this poll: new progress lines stamped with the seconds since Run, newest first."""
        nonlocal logged
        new: list[str] = streamer.messages[logged:]
        if not new:
            return skip
        logged += len(new)
        stamp: float = time.perf_counter() - started
        lines_shown.extendleft(f"{stamp:6.1f} s  {line}" for line in new)
        return "\n".join(lines_shown)

    try:
        while worker.is_alive():
            worker.join(PUMP_SECONDS)
            latest: str = streamer.messages[-1] if streamer.messages else "starting"
            yield stream.read() or b"", f"{latest} ({time.perf_counter() - started:.0f} s)", skip, log_text()
        if errors:
            raise errors[0]
        wall_s: float = time.perf_counter() - started
        if streamer.stop.is_set():
            yield stream.read() or b"", f"Stopped after {wall_s:.0f} s", "Stopped.", log_text()
            return
        SLOT.recording = saved
        person_line: str = " and the 4D person" if person is not None else ""
        yield (
            stream.read() or b"",
            f"Done in {wall_s:.0f} s: {config.layout.num_views} views{person_line}.",
            summary_table(config.layout.num_views, wall_s, streamer.person_summary, saved),
            log_text(),
        )
    finally:
        # A closed page ends this generator (a Stop sets the stop event in stop_status): the worker stops its subprocess
        # at once, the rest at the next stage, denoising step or training step, and frees the slot when it has.
        streamer.stop.set()


def latest_recording() -> gr.File:
    """The download box: shown with the latest finished run's recording, hidden until there is one."""
    path: Path | None = SLOT.recording if SLOT.recording is not None and SLOT.recording.is_file() else None
    return gr.File(value=str(path) if path is not None else None, visible=path is not None)


def stop_status() -> Iterator[str]:
    """Stop the latest run (its worker stops the subprocess at once, the rest at the next stage, denoising step or
    training step); the status line says "Stopping…" until the worker has exited."""
    if SLOT.stop is not None:
        SLOT.stop.set()
    worker: threading.Thread | None = SLOT.worker
    while worker is not None and worker.is_alive():
        yield "Stopping…"
        worker.join(PUMP_SECONDS)
    yield "Stopped."


def build(examples: list[Path] | None = None) -> gr.Blocks:
    """The shared app layout: Input and Outputs tabs, Run, Stop and the status on the left; the WebGPU notice and the
    streaming Rerun viewer on the right. Serve it with :func:`launch` (its look comes with the launch).

    Args:
        examples: The example clips; None: :func:`example_videos`.
    """
    clips: list[Path] = examples if examples is not None else example_videos()
    # fill_width: Gradio caps the page at 1,536 px otherwise; fill_height with the main row's scale=1: the row takes the
    # height the header leaves (simplecv.app_theme), so the page fits the window.
    with gr.Blocks(title="Rerun × 4DAnyone", fill_width=True, fill_height=True) as demo:
        gr.Markdown(DESCRIPTION, elem_id="app-description")
        with gr.Row(elem_id="main-row", equal_height=False, scale=1):
            with gr.Column(scale=1, min_width=360, elem_id="left-column"):
                with gr.Tabs(selected="input") as tabs:
                    with gr.Tab("Input", id="input"):
                        # format=None: Gradio hands the upload over as it is.
                        video: gr.Video = gr.Video(
                            value=str(clips[0]) if clips else None, label="Source clip", sources=["upload"], format=None, elem_id="source-video"
                        )
                        with gr.Accordion("Advanced settings", open=False):
                            views: gr.Dropdown = gr.Dropdown(
                                label="Views (one ring, 15° above the person)", choices=list(VIEW_CHOICES), value=ViewLayout().views_per_layer
                            )
                            seed: gr.Number = gr.Number(label="Seed", value=RunConfig().seed, precision=0)
                            train: gr.Checkbox = gr.Checkbox(label="Train the 4D person after the views", value=True)
                            steps: gr.Number = gr.Number(
                                label=f"Training steps ({APP_STEPS:,} take about 90 s on an RTX 5090)", value=APP_STEPS, precision=0, minimum=100
                            )
                            prior: gr.Checkbox = gr.Checkbox(label="Depth + normal prior (MoGe-2)", value=APP_PRIOR, info=PRIOR_INFO)
                        gr.Examples(
                            examples=[[str(path)] for path in clips], inputs=[video], examples_per_page=EXAMPLES_PER_PAGE, cache_examples=False, elem_id="examples"
                        )
                    with gr.Tab("Outputs", id="outputs"):
                        run_summary: gr.Markdown = gr.Markdown("No run yet.", elem_id="run-summary")
                        recording: gr.File = gr.File(
                            label="Recording (.rrd, 4D person included)", interactive=False, visible=False, elem_id="recording"
                        )
                        run_log: gr.Textbox = gr.Textbox(
                            label="Run log (newest first)", lines=8, max_lines=8, autoscroll=False, interactive=False, elem_id="run-log"
                        )
                with gr.Row(elem_id="run-row"):
                    run_button: gr.Button = gr.Button("Run", variant="primary")
                    stop_button: gr.Button = gr.Button("Stop", variant="stop")
                status: gr.Markdown = gr.Markdown(READY, elem_id="run-status")
            with gr.Column(scale=3):
                gr.HTML('<div id="webgpu-notice"></div>', padding=False, elem_id="webgpu-notice-block")
                viewer: Rerun = Rerun(
                    streaming=True, panel_states={"time": "collapsed", "blueprint": "hidden", "selection": "hidden"}, height="100%", elem_id="viewer"
                )
        run_event = run_button.click(lambda: gr.Tabs(selected="outputs"), outputs=[tabs]).then(
            run_app, inputs=[video, views, seed, train, steps, prior], outputs=[viewer, status, run_summary, run_log]
        )
        run_event.success(latest_recording, outputs=[recording])
        stop_button.click(stop_status, outputs=[status], cancels=[run_event])
    return demo


def launch(demo: gr.Blocks, config: Config) -> None:
    """Serve ``demo`` with the shared look (forced dark), one run at a time (the GPU holds one pipeline)."""
    SLOT.gpu_lock = config.gpu_lock
    demo.queue(max_size=1, default_concurrency_limit=1)
    demo.launch(
        server_name=config.server_name,
        server_port=config.server_port,
        ssr_mode=False,
        allowed_paths=[str(EXAMPLE_VIDEO.parent), str(RUNS_DIR)],
        css=APP_CSS + NOTICE_CSS,
        head=DARK_HEAD,
        js=WEBGPU_CHECK_JS,
        show_error=True,
    )
