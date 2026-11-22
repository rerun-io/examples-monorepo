"""The 4DAnyone app: pick a clip, press Run, watch the phases stream into the embedded Rerun viewer.

The run is :class:`fdanyone_rerun.stream.RunStreamer`, the same core and defaults as ``fdanyone-run --train``; this
module only adds the widgets, a worker thread, the byte pump into the viewer and the app's training preset
(:data:`APP_PERSON`: a short run and no MoGe-2 prior).
"""

import dataclasses
import threading
import time
import uuid
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import rerun as rr
from gradio_rerun import Rerun

from fdanyone.errors import RunCancelled
from fdanyone_rerun.assets import EXAMPLE_VIDEO, PACKAGE_ROOT
from fdanyone_rerun.core import RunConfig, ViewLayout
from fdanyone_rerun.person4d import PersonConfig
from fdanyone_rerun.stream import RunStreamer

APPLICATION_ID: str = "fdanyone-rerun"
RUNS_DIR: Path = PACKAGE_ROOT / "data" / "app-runs"
"""One run directory per click."""
PUMP_SECONDS: float = 0.5
"""How often new recording bytes go to the viewer."""
APP_PERSON: PersonConfig = PersonConfig(train=dataclasses.replace(PersonConfig().train, steps=5_000), prior=False)
"""The app's preset: 5,000 steps (about 2 min on the 5090) and no MoGe-2 prior (the demo runs no extra estimator model)."""
READY: str = "Pick the example or upload a clip of one person (at least 121 frames), then press Run."
DESCRIPTION: str = (
    "# 4DAnyone × Rerun\n"
    "One casual video of a person becomes synchronized videos from new cameras around them "
    "([4DAnyone](https://github.com/ant-research/4DAnyone), 4DAnyone-Turbo, four denoising steps). The viewer follows the run: "
    "motion recovery, the recovered body, a preview of every view at each denoising step, then the camera rig and the views. "
    "Then a 4D Gaussian model of the person trains on the views ([Open4DGS](https://github.com/ASU-NEM-Labs/open4dgs-trigs) TRiGS) "
    "and plays in 3D as Animated SOG, inside the recording."
)
# The page fits the window (posekit #594): Blocks(fill_height) + Row(scale=1) give the main row the height the header leaves,
# the viewer takes its column's height, and the footer stays hidden (Gradio adds its height to the size it posts to an iframe).
# Below 900 px (iPad portrait, ~744 px) the two columns stack: the controls shrink to their content above the viewer, which
# takes the rest of the window. Windows under 800 px tall get a smaller clip preview. Below 600 px (a phone) the page may scroll, and the viewer keeps 70% of the window height.
# Touch targets are at least 44 px.
APP_CSS: str = """
footer { display: none !important; }
#main-row { min-height: 30rem; }
#viewer-column { display: flex; flex-direction: column; min-height: 0; }
#viewer { flex: 1 1 0; min-height: 0; }
#left-column { height: 100%; max-height: 100%; overflow-y: auto; flex-wrap: nowrap; }
#left-column button { min-height: 44px; min-width: 44px; }
#left-column .gr-accordion { overflow-x: hidden !important; }
#left-column .label-wrap { min-width: 0; box-sizing: border-box; }
@media (max-width: 900px) {
    #main-row { flex-direction: column; flex-wrap: nowrap; }
    #left-column { height: auto; max-height: none; flex: 0 0 auto !important; overflow: visible; }
    #source-video, #source-video video { max-height: 180px; }
    #viewer-column { flex: 1 1 0; min-height: 0; }
}
@media (max-height: 800px) {
    #source-video, #source-video video { max-height: 200px; }
}
@media (max-width: 600px) {
    #viewer-column, #viewer { min-height: 70vh; }
}
"""


@dataclass(frozen=True, slots=True)
class Config:
    """Serve the 4DAnyone app."""

    server_name: str = "127.0.0.1"
    """Interface on which Gradio listens."""
    server_port: int = 7873
    """Gradio HTTP port."""


class _RunSlot:
    """One run on the GPU at a time. The slot is taken before a run's worker starts and released by that worker only
    when its run has returned: the worker subprocess reaped, the prefetch closed and the observers detached. A Stop
    ends the request at once, so the slot, not Gradio's queue, keeps the next run out until then."""

    def __init__(self) -> None:
        self.lock: threading.Lock = threading.Lock()
        self.worker: threading.Thread | None = None
        """The worker of the latest run; Stop reports "Stopping…" while it is alive."""


SLOT: _RunSlot = _RunSlot()


def run_app(
    video: str | None, views_per_layer: int | float, seed: int | float, train: bool, steps: int | float
) -> Generator[tuple[bytes | None, str], None, None]:
    """Run 4DAnyone on ``video`` (then train its 4D person) in a worker thread and pump the recording into the viewer until it ends."""
    if video is None:
        yield None, "Choose a video first."
        return
    run_id: str = uuid.uuid4().hex[:12]
    config: RunConfig = RunConfig(
        video=Path(video), output_dir=RUNS_DIR / run_id, layout=ViewLayout(views_per_layer=int(views_per_layer)), seed=int(seed)
    )
    person: PersonConfig | None = dataclasses.replace(APP_PERSON, train=dataclasses.replace(APP_PERSON.train, steps=int(steps))) if train else None
    recording: rr.RecordingStream = rr.RecordingStream(APPLICATION_ID, recording_id=run_id)
    stream: rr.BinaryStream = recording.binary_stream()
    streamer: RunStreamer = RunStreamer(config, recording, previews=True, person=person)
    errors: list[BaseException] = []

    def work() -> None:
        try:
            streamer.run()
        except BaseException as error:  # re-raised by the generator below, beartype violations included
            errors.append(error)
        finally:
            SLOT.lock.release()

    while not SLOT.lock.acquire(timeout=PUMP_SECONDS):
        yield b"", "Waiting for the previous run to stop…"
    worker: threading.Thread = threading.Thread(target=work, name=f"fdanyone-{run_id}", daemon=True)
    started: float = time.perf_counter()
    try:
        worker.start()
    except BaseException:
        SLOT.lock.release()
        raise
    SLOT.worker = worker
    try:
        while worker.is_alive():
            worker.join(PUMP_SECONDS)
            yield stream.read(), f"Running… {time.perf_counter() - started:.0f} s"
        recording.flush()
        if errors and not isinstance(errors[0], RunCancelled):
            raise errors[0]
        person_line: str = " and the 4D person" if person is not None else ""
        yield stream.read(), f"Done in {time.perf_counter() - started:.0f} s: {config.layout.num_views} views{person_line} in {config.output_dir}"
    finally:
        # A Stop (or a closed page) ends this generator: the worker stops its subprocess at once, the rest at the next
        # stage or denoising step, and frees the slot when it has.
        streamer.stop.set()


def stop_status() -> Iterator[str]:
    """Stop's status line: "Stopping…" until the stopped run's worker has exited."""
    worker: threading.Thread | None = SLOT.worker
    while worker is not None and worker.is_alive():
        yield "Stopping…"
        worker.join(PUMP_SECONDS)
    yield "Stopped."


def build(examples: list[str] | None = None) -> gr.Blocks:
    """The input, settings, Run and Stop on the left; the streaming Rerun viewer on the right."""
    with gr.Blocks(title="4DAnyone × Rerun", fill_height=True) as demo:
        gr.Markdown(DESCRIPTION, elem_id="app-description")
        with gr.Row(elem_id="main-row", equal_height=False, scale=1):
            with gr.Column(scale=1, min_width=340, elem_id="left-column"):
                # A fixed height: a portrait 4K upload would otherwise stretch the column past the window.
                video = gr.Video(label="Source clip", sources=["upload"], format=None, height=260, elem_id="source-video")
                gr.Examples(
                    examples=[[path] for path in (examples if examples is not None else [str(EXAMPLE_VIDEO)])], inputs=[video], cache_examples=False
                )
                with gr.Accordion("Settings", open=False):
                    views = gr.Slider(
                        label="Views (one ring, 15° above the subject)", minimum=6, maximum=24, step=6, value=ViewLayout().views_per_layer
                    )
                    seed = gr.Number(label="Seed", value=RunConfig().seed, precision=0)
                    train = gr.Checkbox(label="Then train the 4D person on the views", value=True)
                    steps = gr.Number(
                        label="Training steps (5,000 take about 2 min on an RTX 5090)", value=APP_PERSON.train.steps, precision=0, minimum=100
                    )
                with gr.Row():
                    run_button = gr.Button("Run", variant="primary")
                    stop = gr.Button("Stop", variant="stop")
                status = gr.Markdown(READY, elem_id="run-status")
            with gr.Column(scale=3, elem_id="viewer-column"):
                viewer = Rerun(
                    streaming=True, panel_states={"time": "collapsed", "blueprint": "hidden", "selection": "hidden"}, height="100%", elem_id="viewer"
                )

        run_event = run_button.click(run_app, inputs=[video, views, seed, train, steps], outputs=[viewer, status])
        stop.click(stop_status, outputs=[status], cancels=[run_event])
    return demo


def launch(demo: gr.Blocks, config: Config) -> None:
    """Serve ``demo``, one run at a time (the GPU holds one pipeline)."""
    demo.queue(max_size=1, default_concurrency_limit=1)
    demo.launch(
        server_name=config.server_name,
        server_port=config.server_port,
        ssr_mode=False,
        allowed_paths=[str(EXAMPLE_VIDEO.parent)],
        css=APP_CSS,
        show_error=True,
    )
