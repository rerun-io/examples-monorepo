"""Triangulate every person's landmarks and fit SMPL-X over a sliding window, one fitter per person (emits late).

A fitter emits a frame ``emit_lag`` ticks after it saw it, and a person whose fitter starts later sends its first window's
head frames up to ``window_size - 1`` ticks back. Fits are therefore held per frame and released, in time order, once the
frame is older than the newest ``window_size`` ticks (a later result for a closed frame, possible only for a person seen
in fewer than two views on some ticks, is dropped and counted); ``flush`` drains every fitter's lag and head.
"""

from collections import deque
from collections.abc import Callable, Mapping
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from dagkit.node import Emit, Fenced, PortType, RunContext, Tick
from dagkit.time import TIME_MAX, TimeNs
from jaxtyping import Bool, Float32, Int32, Int64
from numpy import ndarray
from simplecv.gpu import consume
from simplecv.types.rig import RigCameras
from torch import Tensor

from mamma.ops.triangulation import triangulate_gated, triangulate_landmarks
from mamma.ops.window_fit import FitResult, FitterConfig
from mamma.types import BodyTopology, Landmarks3d, MultiViewLandmarks, SmplxFits

CONTACT_MIN: float = 0.25
"""Fused floor-contact probability below which a landmark is not pulled to the floor."""
CONTACT_HEIGHT_M: float = 0.05
"""Height scale of the floor pull: only landmarks the geometry also places near z = 0 are pulled (a jump predicts contact)."""


@runtime_checkable
class WindowFitter(Protocol):
    """What the node needs from ``ops.window_fit.SlidingWindowFitter`` (tests inject a fake)."""

    def push(
        self, frame_idx: int, pts2d_per_cam: list[Tensor], vis_per_cam: list[Tensor], anchors3d: Tensor, anchors_valid: Tensor, floor_contact: Tensor | None = None, optimize: bool = True
    ) -> FitResult | None: ...
    def drain_head(self) -> list[FitResult]: ...
    def drain(self) -> list[FitResult]: ...


class FitNode:
    inputs: Mapping[str, PortType] = {"landmarks": Fenced[MultiViewLandmarks], "rig": RigCameras}
    outputs: Mapping[str, PortType] = {"fit": SmplxFits, "kp3d": Landmarks3d, "topology": BodyTopology}
    emits_late: bool = True

    def __init__(self, config: FitterConfig, faces: Int32[ndarray, "f 3"], new_fitter: Callable[[list[Tensor], list[Tensor]], WindowFitter]) -> None:
        """``new_fitter(K per view, cam_T_world per view)`` starts one person's fitter (engine-size intrinsics, float32 on the device)."""
        self.config, self.faces, self.new_fitter = config, faces, new_fitter

    def reset(self, ctx: RunContext) -> None:
        self.fitters: dict[int, WindowFitter] = {}
        self.held: dict[TimeNs, dict[int, FitResult]] = {}
        self.times: dict[int, TimeNs] = {}
        """Frame index -> its tick time."""
        self.pushed: dict[int, FitResult] = {}
        """Each person's newest result: with ``emit_stride`` > 1 a fitter repeats it on skipped ticks."""
        self.recent: deque[TimeNs] = deque(maxlen=self.config.window_size)
        self.closed: TimeNs | None = None
        """Every fit at or before this time has been put (or dropped as late)."""
        self.late: int = 0
        self.cameras: tuple[list[Tensor], list[Tensor]] | None = None

    def step(self, tick: Tick, out: Emit) -> None:
        assert isinstance(tick.value, Fenced) and isinstance(tick.value.value, MultiViewLandmarks)
        landmarks: MultiViewLandmarks = consume(tick.value)
        if self.cameras is None:
            rig = tick.asof["rig"]
            assert isinstance(rig, RigCameras), "fit needs the rig (a STATIC AsOf value)"
            selected: RigCameras = rig.select(landmarks.views)
            engine_wh: Int64[ndarray, "v 2"] = np.rint(selected.resolution_wh / landmarks.scale_xy).astype(np.int64)
            engine: RigCameras = selected.scaled_to(int(engine_wh[0, 1]), int(engine_wh[0, 0]))
            device: torch.device = landmarks.xy.device
            self.cameras = (
                [torch.as_tensor(K, dtype=torch.float32, device=device) for K in engine.intrinsics],
                [torch.as_tensor(T, dtype=torch.float32, device=device) for T in engine.cam_T_world],
            )
            self.P: Float32[Tensor, "v 3 4"] = torch.stack(self.cameras[0]) @ torch.stack(self.cameras[1])[:, :3]
        self.times[landmarks.frame_index] = tick.t_ns
        self.recent.append(tick.t_ns)
        new: list[int] = []
        points: list[tuple[int, Float32[Tensor, "n 3"], Bool[Tensor, " n"]]] = []
        for person in sorted(set(landmarks.person.tolist())):
            rows: Int64[ndarray, " r"] = np.flatnonzero(landmarks.person == person)
            if rows.size < 2:
                continue
            result, cloud = self._fit_person(landmarks, person, rows, new)
            points.append((person, *cloud))
            fits: list[FitResult] = [*self.fitters[person].drain_head()]
            if result is not None and result is not self.pushed.get(person):
                fits.append(result)
            for fit in fits:
                self.held.setdefault(self.times[fit.frame_idx], {})[person] = self.pushed[person] = fit
        if new:
            out.put("topology", tick.t_ns, BodyTopology(tuple(new), self.faces))
        if points:
            out.put("kp3d", tick.t_ns, Landmarks3d(
                landmarks.frame_index, tuple(p for p, _, _ in points), np.stack([c.cpu().numpy() for _, c, _ in points]), np.stack([v.cpu().numpy() for _, _, v in points]),
            ))  # fmt: skip
        if len(self.recent) == self.recent.maxlen:
            self._release(out, self.recent[0])
            out.watermark(self.recent[0])

    def _fit_person(
        self, landmarks: MultiViewLandmarks, person: int, rows: Int64[ndarray, " r"], new: list[int]
    ) -> tuple[FitResult | None, tuple[Float32[Tensor, "n 3"], Bool[Tensor, " n"]]]:
        """One person's tick: gated triangulation, the dense cloud for display, then a push into the person's fitter (the
        first full window also leaves its head frames in ``drain_head``)."""
        assert self.cameras is not None
        views: int = len(landmarks.views)
        device: torch.device = landmarks.xy.device
        index: Tensor = torch.as_tensor(landmarks.view[rows], device=device)
        row_index: Int64[Tensor, " r"] = torch.as_tensor(rows, device=device)

        def per_view(values: Tensor) -> Tensor:  # views without this person stay zero (excluded by visibility)
            return torch.zeros((views, *values.shape[1:]), device=device).index_copy_(0, index, values[row_index])

        xy, log_variance, visibility = per_view(landmarks.xy), per_view(landmarks.log_variance), per_view(landmarks.visibility)
        gated, anchors, valid = triangulate_gated(xy, log_variance, visibility, self.P, reproj_thresh_px=self.config.reproj_gate_px)
        cloud: tuple[Float32[Tensor, "n 3"], Bool[Tensor, " n"]] = triangulate_landmarks(xy, log_variance, visibility, self.P, vis_thresh=-1.0)
        contact: Float32[Tensor, " n"] = landmarks.floor_contact[row_index].mean(dim=0)  # fused over the views that see the person
        contact = torch.where(contact < CONTACT_MIN, 0.0, contact) * valid.float() * torch.exp(-((anchors[:, 2] / CONTACT_HEIGHT_M) ** 2))
        if person not in self.fitters:
            self.fitters[person] = self.new_fitter(*self.cameras)
            new.append(person)
        observed: list[Float32[Tensor, "n 3"]] = list(torch.cat([xy, log_variance[..., None]], dim=-1).unbind(0))
        optimize: bool = self.config.fit_stride <= 1 or landmarks.frame_index % self.config.fit_stride == 0
        return self.fitters[person].push(landmarks.frame_index, observed, list(gated.unbind(0)), anchors, valid, floor_contact=contact, optimize=optimize), cloud

    def _release(self, out: Emit, until: TimeNs) -> None:
        """Put every held frame with time <= ``until``, in time order, and close everything up to ``until``."""
        for t_ns in sorted(t for t in self.held if t <= until):
            fits: dict[int, FitResult] = self.held.pop(t_ns)
            if self.closed is not None and t_ns <= self.closed:
                self.late += len(fits)
                continue
            persons: list[int] = sorted(fits)
            out.put("fit", t_ns, SmplxFits(
                fits[persons[0]].frame_idx, tuple(persons), *(np.stack([getattr(fits[p], name) for p in persons]) for name in ("vertices", "pose", "betas", "trans")),
            ))  # fmt: skip
        self.closed = until

    def flush(self, out: Emit) -> None:
        for person, fitter in self.fitters.items():
            for result in fitter.drain():
                self.held.setdefault(self.times[result.frame_idx], {})[person] = result
        self.fitters.clear()
        self._release(out, TIME_MAX)
        if self.late:
            print(f"fit: dropped {self.late} late person-frames (their frame was already released)")
            self.late = 0
