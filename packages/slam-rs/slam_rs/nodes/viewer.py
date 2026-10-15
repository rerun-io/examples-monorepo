"""The VIO run in the viewer as ``replay --stage vio`` draws it (``vio_log``), fed by the ``vio`` node's ``view`` port."""

import threading
from collections.abc import Callable, Mapping
from typing import cast

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from dagkit.node import Emit, PortType, RunContext, Tick

from slam_rs.catalog_feed import CameraCalib
from slam_rs.nodes.vio import VioView
from slam_rs.types import empty_trajectory
from slam_rs.vio_log import FrameMode, VioLogger, log_calibration, log_frameset_inputs


class VioViewer:
    """Draw the VIO run as replay does.

    For each tracked frameset (on the estimator clock, as replay): its images and IMU batch, then the estimate, the window, the
    landmarks, the keypoints and the counters; at the end each whole path. The rig and the blueprint go out with the first view.
    ``vio_log`` draws on the thread's recording: this sink's ``recording`` is bound to its thread from the first view to the flush
    (once: a swap per call would end rerun's batch every time, a chunk per row)."""

    inputs: Mapping[str, PortType] = {"view": VioView}
    outputs: Mapping[str, PortType] = {}

    def __init__(self, recording: rr.RecordingStream, blueprint: Callable[[tuple[CameraCalib, ...]], rrb.Blueprint], frames: FrameMode = "jpeg") -> None:
        self.recording, self.blueprint, self.frames = recording, blueprint, frames
        self.logger: VioLogger | None = None
        self.thread: int | None = None  # the thread the recording is bound to, and what that thread had before
        self.previous: rr.RecordingStream | None = None

    def reset(self, ctx: RunContext) -> None:
        self.logger = None

    def _draw(self, draw: Callable[[], None]) -> None:
        if self.thread != threading.get_ident():
            self.previous, self.thread = rr.set_thread_local_data_recording(self.recording), threading.get_ident()
        draw()

    def step(self, tick: Tick, out: Emit) -> None:
        view: VioView = cast(VioView, tick.value)

        def draw() -> None:
            if self.logger is None:
                log_calibration(view.cameras)
                self.recording.send_blueprint(self.blueprint(view.cameras))
                self.logger = VioLogger(view.cameras, empty_trajectory(), np.zeros(0, np.int64))
            log_frameset_inputs(view.cameras, view.frames, self.frames)
            self.logger.log(view.result, view.snapshot, view.flow, view.elapsed_ms)

        self._draw(draw)

    def flush(self, out: Emit) -> None:
        if self.logger is not None:
            self._draw(self.logger.log_complete_paths)
            rr.set_thread_local_data_recording(self.previous)
            self.thread = None
        self.recording.flush()
