"""The ``hands`` node (SPEC 5.3): ``_core.HandsLayer``, the runtime's own scheduler as ``robocap-live --source replay --slam
reference`` runs it (DetNet and KeyNet on ONNX Runtime, the ROBUST tracker on handfit), writing the hands layer file itself.

``reset`` builds a fresh core from the STATIC rig and whole ``slam_rs`` pose table (a frameset takes the pose within 2 ms, one
without the newest earlier one); ``step`` pushes a frameset; ``flush`` finishes the layer and emits its file and counts.
"""

import importlib.util
import os
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

import numpy as np
from dagkit.node import Emit, PortType, RunContext, Tick
from dagkit.sources import SegmentRef
from dagkit.time import STATIC, TimeNs
from dagkit.types import LayerFile
from serde import serde
from simplecv.types.frames import LumaFrameset

from robocap_live import _core
from robocap_live.types import RIG_ROOT, CapRig, CatalogPoses, LayerRun

LAYER: str = "hands"
"""Catalog layer name of the hands layer."""
CoreFactory: TypeAlias = Callable[..., _core.HandsLayer]


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class HandsParams:
    """The core's networks and overlays."""

    models_dir: Path
    """``detnet_full.onnx`` and ``keynet.onnx`` (packages/robocap-live/models/MODELS.md)."""
    device: Literal["auto", "cpu", "cuda"] = "auto"
    """ONNX Runtime device: CUDA on GPU 0 when its execution provider registers (auto), or forced."""
    ort_dylib: Path | None = None
    """``libonnxruntime.so``; default ``ORT_DYLIB_PATH``, else the library of this env's ``onnxruntime`` package."""
    ort_threads: int = 0
    """ONNX Runtime intra-op threads per session (0 = its default)."""
    overlays: Literal["fit", "debug", "verbose"] = "debug"
    """How much of the hand pipeline the camera panes show."""


def default_ort_dylib() -> Path:
    """``ORT_DYLIB_PATH``, else the env's ``onnxruntime`` library, found without importing it (that would load a second copy).

    Raises:
        ValueError: If neither exists.
    """
    if from_env := os.environ.get("ORT_DYLIB_PATH"):
        return Path(from_env)
    spec = importlib.util.find_spec("onnxruntime")
    for location in (spec.submodule_search_locations or []) if spec is not None else []:
        if libraries := sorted((Path(location) / "capi").glob("libonnxruntime.so*")):
            return libraries[-1]
    raise ValueError("no ONNX Runtime library: pass ort_dylib or set ORT_DYLIB_PATH (or run in the robocap-live env)")


class HandsNode:
    """Track the hands in each frameset.

    The Rust core (DetNet and KeyNet on ONNX Runtime, the ROBUST tracker on handfit) takes the rig and the whole ``slam_rs``
    pose table at reset, each frameset in ``step``, and writes the hands layer file itself; ``flush`` emits that file and the
    run's counts."""

    inputs: Mapping[str, PortType] = {"frames": LumaFrameset, "cap_rig": CapRig, "poses": CatalogPoses}
    outputs: Mapping[str, PortType] = {"layer": LayerFile, "run": LayerRun}

    def __init__(self, segment: SegmentRef, params: HandsParams, output_dir: Path, core: CoreFactory = _core.HandsLayer) -> None:
        """``output_dir/<segment>/`` holds the partial layer file; ``core`` builds the core (tests pass one without networks)."""
        self.segment, self.params, self.output_dir, self.make_core, self.lock = segment, params, output_dir, core, threading.Lock()
        self.core: _core.HandsLayer | None = None
        self.partial: Path | None = None
        self.t_ns: TimeNs = STATIC
        self.loaded_s: float = 0.0

    def reset(self, ctx: RunContext) -> None:
        self.abort()
        cap, poses = ctx.statics.get("cap_rig"), ctx.statics.get("poses")
        if not isinstance(cap, CapRig) or not isinstance(poses, CatalogPoses):
            raise ValueError(f"{self.segment.segment}: needs the base layer's rig and the slam_rs poses of {RIG_ROOT}; register the SLAM poses first")
        intrinsics = cap.rig.intrinsics  # the core takes C-contiguous float64 arrays
        rig = _core.Rig(
            names=list(cap.names), resolution_wh=cap.rig.resolution_wh.astype(np.float64), cam_from_rig=np.ascontiguousarray(cap.rig.cam_T_world),
            focal=np.ascontiguousarray(intrinsics[:, [0, 1], [0, 1]]), principal=np.ascontiguousarray(intrinsics[:, :2, 2]), fisheye62=np.ascontiguousarray(cap.fisheye62),
            source=f"catalog {self.segment.dataset} {self.segment.segment}", device=self.segment.segment,  # provenance only
        )  # fmt: skip
        self.partial = self.output_dir / self.segment.segment / f".{LAYER}.{uuid.uuid4().hex[:12]}.partial"
        self.partial.parent.mkdir(parents=True, exist_ok=True)
        started: float = time.perf_counter()
        p: HandsParams = self.params
        self.core = self.make_core(
            rig, self.partial, self.segment.segment, poses.t_ns, poses.world_from_rig, models_dir=p.models_dir, device=p.device,
            ort_dylib=p.ort_dylib or default_ort_dylib(), ort_threads=p.ort_threads, overlays=p.overlays,
        )  # fmt: skip
        self.loaded_s, self.t_ns = time.perf_counter() - started, STATIC
        print(f"{self.segment.segment}: nets {self.core.nets} (loaded in {self.loaded_s:.1f} s), {len(poses.t_ns)} slam_rs poses", flush=True)

    def step(self, tick: Tick, out: Emit) -> None:
        frames: LumaFrameset = cast(LumaFrameset, tick.value)
        with self.lock:
            assert self.core is not None, "step before reset (or after abort)"
            self.core.push(frames.t_ns, list(frames.luma), [frames.t_ns if t is None else t for t in frames.cam_t_ns])
        self.t_ns = frames.t_ns

    def flush(self, out: Emit) -> None:
        started: float = time.perf_counter()
        with self.lock:
            if self.core is None or self.partial is None:
                return
            summary: _core.LayerSummary = self.core.finish()
            layer, self.core, self.partial = LayerFile(self.partial), None, None
        stages: _core.HandStageTimes = summary.hand_stages
        run = LayerRun(
            summary.framesets, summary.with_pose, summary.held_pose, *summary.tracked, *summary.reported, summary.scale, summary.scale_final,
            nets_load_s=self.loaded_s, detnet_s=stages.detnet_ms / 1e3, crops_s=stages.crops_ms / 1e3, keynet_s=stages.keynet_ms / 1e3,
            fit_s=stages.fit_ms / 1e3, tracker_s=stages.tracker_ms / 1e3, finish_s=time.perf_counter() - started,
        )  # fmt: skip
        print(f"{self.segment.segment}: {run.framesets} framesets ({run.with_pose} with a pose, {run.held_pose} held), reported "
              f"left {run.reported_left} / right {run.reported_right}, scale {run.scale:.4f}{'' if run.scale_final else ' (not calibrated)'}")  # fmt: skip
        out.put("layer", self.t_ns, layer)
        out.put("run", self.t_ns, run)

    def abort(self) -> None:
        """Stop a running core and delete its unfinished file (a finished one belongs to the publish sink)."""
        with self.lock:  # a failing runner thread calls this at once, maybe during push (PyO3 refuses a call into a borrowed core)
            if self.core is not None:
                self.core.abort()
                self.core = None
            if self.partial is not None:
                self.partial.unlink(missing_ok=True)
                self.partial = None
