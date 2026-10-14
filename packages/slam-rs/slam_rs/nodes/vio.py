"""The ``vio`` node: slam-rs's estimator (``_core.Vio`` behind :class:`~slam_rs.tracking.Lockstep`) on the one time-ordered stream.

Each frameset's IMU batch arrives on ``imu`` at the frameset's time, before its gray8 images on ``frames`` (port order), so the
estimator sees exactly the calls ``Lockstep.push`` makes: the batch, then ``track``. Poses leave late (a refused frameset is
held, lag mode returns the previous frameset's pose); ``flush`` drains the estimator and writes the segment's ``slam_rs`` layer.
"""

import time
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
from dagkit.node import Emit, PortType, RunContext, Tick
from dagkit.sources import SegmentRef
from dagkit.time import STATIC, TimeNs
from dagkit.types import LayerFile
from jaxtyping import Float64
from numpy import ndarray
from serde import serde

from slam_rs import _core
from slam_rs.catalog_feed import Frameset, ImuStream
from slam_rs.catalog_layer import POSE_SOURCE, write_layer
from slam_rs.config import SlamConfig, config_text_sha256, load_slam_config
from slam_rs.tracking import Lockstep, SegmentRun
from slam_rs.trajectory import Trajectory, empty_trajectory
from slam_rs.types import RigPose, VioRig


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class VioParams:
    """The estimator's profile, frontend and decoder."""

    profile: Literal["fast", "reference"] = "fast"
    """Estimator configuration profile."""
    backend: Literal["auto", "gpu", "cpu"] = "auto"
    """Auto uses the GPU frontend when this core has one and an adapter answers; gpu requires it; cpu disables it."""
    decode_device: Literal["auto", "cuda", "cpu"] = "auto"
    """CUDA (NVDEC, the slam-rs-cuda env) when a device answers; cpu keeps the reference PyAV gray8 conversion."""

    def decoder(self) -> Literal["cpu", "cuda"]:
        """``decode_device`` resolved; raises ``ValueError`` when CUDA is asked for and absent."""
        import torch

        if self.decode_device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA decoding requires an available NVIDIA GPU and the slam-rs-cuda environment")
        return "cuda" if self.decode_device == "cuda" or (self.decode_device == "auto" and torch.cuda.is_available()) else "cpu"


class VioNode:
    """Track the rig pose with VIO.

    ``reset`` builds the estimator from the STATIC rig and the dataset's profiled config; ``step`` pushes an IMU batch or holds
    a frameset and tracks what the samples cover; ``flush`` drains it and writes the ``slam_rs`` layer file (``run`` keeps the
    run's counts)."""

    inputs: Mapping[str, PortType] = {"imu": ImuStream, "frames": Frameset, "rig": VioRig}
    outputs: Mapping[str, PortType] = {"pose": RigPose, "layer": LayerFile}
    emits_late: bool = True

    def __init__(self, segment: SegmentRef, params: VioParams, output_dir: Path, decoder: Literal["cpu", "cuda"]) -> None:
        """``output_dir/<segment>/`` holds the layer file; ``decoder`` is the one the source decodes with (layer metadata)."""
        self.segment, self.params, self.output_dir, self.decoder = segment, params, output_dir, decoder
        self.lockstep: Lockstep | None = None
        self.run: SegmentRun | None = None
        self.offset_ns, self.export_ns = 0, 0  # video_time -> estimator clock; estimator clock -> device clock

    def reset(self, ctx: RunContext) -> None:
        rig = ctx.statics.get("rig")
        if not isinstance(rig, VioRig):
            raise ValueError(f"{self.segment.segment}: no rig calibration in the base layer")
        settings: SlamConfig = load_slam_config()
        dataset: str = self.segment.dataset
        self.config_text: str = (
            settings.robocap_config_text(self.params.profile) if dataset == "robocap" else settings.vio_config_text(dataset, self.params.profile)
        )
        calibration: _core.Calibration = _core.Calibration.from_catalog(rig.cameras, rig.imu)
        flow: _core.VioConfig = _core.VioConfig.from_json(self.config_text)
        try:
            vio = _core.Vio(calibration, flow, gpu=self.params.backend == "gpu" or (self.params.backend == "auto" and _core.gpu_backend is not None))
        except ValueError as error:  # only an absent adapter permits the auto fallback
            if self.params.backend != "auto" or "wgpu found no" not in str(error):
                raise
            print(f"GPU unavailable: {error}; using CPU", flush=True)
            vio = _core.Vio(calibration, flow, gpu=False)
        self.lockstep, self.run, self.framesets, self.t_ns, self.wm = Lockstep(vio), None, 0, STATIC, STATIC
        self.offset_ns, self.export_ns = rig.imu.cam_time_offset_ns, rig.export_offset_ns
        self.poses: list[tuple[int, Float64[ndarray, " 3"], Float64[ndarray, " 4"]]] = []  # device clock, position, wxyz
        self.started: float = time.monotonic()

    def _emit(self, tracked: Iterator[tuple[Frameset, _core.VioResult]], out: Emit) -> None:
        for _, result in tracked:
            pose: Float64[ndarray, " 7"] | None = result.world_from_rig
            assert pose is not None, f"tracking result at {result.t_ns} has no pose"
            self.poses.append((result.t_ns + self.export_ns, pose[0:3].copy(), np.roll(pose[3:7], 1)))
            out.put("pose", result.t_ns - self.offset_ns, RigPose(*self.poses[-1][1:]))

    def step(self, tick: Tick, out: Emit) -> None:
        assert self.lockstep is not None, "step before reset"
        if tick.port == "imu":
            self.lockstep.push_imu(cast(ImuStream, tick.value))
        else:
            self.framesets, self.t_ns = self.framesets + 1, tick.t_ns
            self.lockstep.hold(cast(Frameset, tick.value))
        self._emit(self.lockstep.track_pending(), out)
        held: list[int] = [f.t_ns - self.offset_ns for f in [*self.lockstep.pending, *self.lockstep.lagged]]
        wm: TimeNs = min([*held, tick.t_ns]) - 1  # a frameset follows its imu at t, and the layer leaves at the last one
        if wm > self.wm:
            self.wm = wm
            out.watermark(wm)

    def flush(self, out: Emit) -> None:
        if self.lockstep is None:
            return
        self._emit(self.lockstep.flush(), out)
        t_ns, positions, quaternions = zip(*self.poses, strict=True) if self.poses else ((), (), ())
        estimate = Trajectory(np.array(t_ns, dtype=np.int64), np.array(positions).reshape(-1, 3), np.array(quaternions).reshape(-1, 4))
        elapsed: list[float] = self.lockstep.elapsed_ms
        self.run = SegmentRun(estimate, self.framesets, self.framesets - len(self.poses), time.monotonic() - self.started, empty_trajectory(),
                              float(np.median(elapsed)) if elapsed else float("nan"), config_text_sha256(self.config_text))  # fmt: skip
        backend: str = str(_core.gpu_backend) if self.lockstep.vio.gpu else "cpu"
        partial: Path = self.output_dir / self.segment.segment / f".{POSE_SOURCE}.{uuid.uuid4().hex[:12]}.partial"
        partial.parent.mkdir(parents=True, exist_ok=True)
        write_layer(partial, self.segment.segment, self.run, clock_offset_ns=self.export_ns + self.offset_ns,
                    profile=self.params.profile, backend=backend, decoder=self.decoder)  # fmt: skip
        print(
            f"{self.segment.segment}: {self.run.framesets} framesets, {len(estimate)} poses, {self.run.lost} lost, {self.run.wall_s:.1f} s ({backend})"
        )
        self.lockstep = None
        out.put("layer", self.t_ns, LayerFile(partial))
