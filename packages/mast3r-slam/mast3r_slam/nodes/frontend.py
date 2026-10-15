"""Frontend node (SPEC 5.5): tracks every input frame against the last keyframe and hands keyframes / relocalisation to the backend."""

from collections.abc import Mapping
from typing import Literal

import lietorch
import numpy as np
import torch
from dagkit.node import Emit, PortType, RunContext, Tick
from jaxtyping import Float, Float32, Int
from mast3r.model import AsymmetricMASt3R
from numpy import ndarray
from simplecv.types.frames import MultiViewFrames
from simplecv.types.rig import RigCameras
from torch import Tensor

from mast3r_slam.models.mast3r import mast3r_inference_mono
from mast3r_slam.nodes.shared import SharedSlam
from mast3r_slam.nodes.tracking import FrameTracker
from mast3r_slam.ops.image import ResizedImage, RgbNormalized, frame_intrinsics, resize_img
from mast3r_slam.types import BackendTask, Frame, KeyframeAdded, Mode, RelocRequest, SlamConfig, TrackedFrame


def create_frame(frame_id: int, rgb: RgbNormalized, world_sim3_cam: lietorch.Sim3, img_size: Literal[224, 512], device: str) -> Frame:
    """A Frame of one RGB image (resized and cropped for MASt3R) at an initial pose."""
    resized: ResizedImage = resize_img(rgb, img_size)
    img_shape: Int[Tensor, "1 2"] = torch.tensor(resized.true_shape, device=device)
    rgb_cropped: Float[Tensor, "h w 3"] = torch.from_numpy(resized.rgb_uint8) / 255.0
    return Frame(frame_id, resized.rgb_tensor.to(device=device), img_shape, img_shape.clone(), rgb_cropped, world_sim3_cam)


class Frontend:
    """Track each frame against the keyframes.

    Per frame, by mode: INIT (the first frame gets a mono point map and becomes keyframe 0), TRACKING (pose solve
    against the last keyframe; a low-overlap frame becomes a keyframe), RELOC (mono point map; the backend relocalises it).

    Lossless (SPEC 5.5, today's ``single_thread``): before each frame and at flush the frontend waits until the backend
    acknowledged every task it handed over, so every frame sees one fixed map. Realtime: lost frames that arrive before
    the backend acknowledged the relocalisation request are skipped, so the one shared current frame stays the requested
    one. A frame's ``tracked`` pose leaves after that wait, when the backend's relocalisation verdict is known (hence
    ``emits_late``). ``abort`` ends a wait when the run fails.
    """

    outputs: Mapping[str, PortType] = {"tracked": TrackedFrame, "task": BackendTask}
    emits_late: bool = True

    def __init__(self, params: SlamConfig, device: str = "cuda:0") -> None:
        self.params, self.device = params, device
        self.inputs: dict[str, PortType] = {"frames": MultiViewFrames} | ({"rig": RigCameras} if params.use_calib else {})
        self.held: tuple[int, int] | None = None
        """(time, frame index) of the frame whose tracked verdict waits for the backend."""
        self.aborted: bool = False
        self.issued: int = 0
        """Tasks handed to the backend (the number of the last one)."""
        self.reloc_task: int = 0
        """Number of the last relocalisation request."""

    def reset(self, ctx: RunContext) -> None:
        self.model: AsymmetricMASt3R = ctx.resources["mast3r"]  # type: ignore[assignment]
        self.shared: SharedSlam = ctx.resources["shared"]  # type: ignore[assignment]
        self.shared.reset()
        self.lossless, self.held, self.aborted, self.issued, self.reloc_task = ctx.mode == "lossless", None, False, 0, 0
        self.tracker: FrameTracker = FrameTracker(self.model, self.shared.keyframes, self.params.tracking, self.params.matching, self.params.use_calib)
        torch.backends.cuda.matmul.allow_tf32 = True  # as the old CLI set it for tracking (main process); the backend keeps the default
        if self.params.use_calib:
            rig: RigCameras | None = ctx.statics.get("rig")  # type: ignore[assignment]
            if rig is None:
                raise ValueError("use_calib needs the source's rig intrinsics (a catalog segment with a calibrated camera)")
            width, height = (int(v) for v in rig.resolution_wh[0])
            K: Float32[ndarray, "3 3"] = frame_intrinsics(rig.intrinsics[0], height, width, self.params.img_size)
            self.shared.keyframes.set_intrinsics(torch.from_numpy(K).to(self.device))

    def _settle(self, out: Emit) -> None:
        """Lossless: wait for the backend; then emit the held frame's pose if the system is (again) TRACKING."""
        if self.held is None:
            return
        if self.lossless and not self.shared.wait(self.issued, lambda: self.aborted):
            raise RuntimeError("the run failed while the frontend waited for the backend")
        (t_ns, index), self.held = self.held, None
        if self.shared.states.get_mode() == Mode.TRACKING:
            pose: Float32[ndarray, " 8"] = self.shared.states.get_frame().world_sim3_cam.data[0].cpu().numpy()
            out.put("tracked", t_ns, TrackedFrame(index, pose))
        out.watermark(t_ns)

    def _hand_over(self, task: BackendTask, t_ns: int, out: Emit) -> None:
        """Publish the GPU writes the task refers to, then hand it to the backend as task number ``issued``."""
        self.shared.publish()
        self.issued += 1
        out.put("task", t_ns, task)

    def _queue_keyframe(self, frame: Frame, t_ns: int, out: Emit) -> None:
        self.shared.keyframes.append(frame)
        self._hand_over(KeyframeAdded(len(self.shared.keyframes) - 1), t_ns, out)

    @torch.no_grad()
    def step(self, tick: Tick, out: Emit) -> None:
        self._settle(out)
        frames: MultiViewFrames = tick.value  # type: ignore[assignment]
        states = self.shared.states
        mode: Mode = states.get_mode()
        if mode == Mode.RELOC and self.shared.completed.value < self.reloc_task:  # realtime: the backend still owns the requested shared frame
            out.watermark(tick.t_ns)
            return
        rgb: RgbNormalized = frames.rgb[0].permute(1, 2, 0).cpu().numpy().astype(np.float32) / 255.0
        pose: lietorch.Sim3 = lietorch.Sim3.Identity(1, device=self.device) if mode == Mode.INIT else states.get_frame().world_sim3_cam
        frame: Frame = create_frame(frames.frame_index, rgb, pose, self.params.img_size, self.device)
        if mode == Mode.TRACKING:
            new_kf, try_reloc = self.tracker.track(frame)
            if try_reloc:
                states.set_mode(Mode.RELOC)
            states.set_frame(frame)
            if new_kf:
                self._queue_keyframe(frame, tick.t_ns, out)
        elif mode in (Mode.INIT, Mode.RELOC):  # a mono point map; INIT bootstraps the map, RELOC asks the backend
            X, C = mast3r_inference_mono(self.model, frame)
            frame.update_pointmap(X, C, self.params.tracking)
            if mode == Mode.INIT:
                self._queue_keyframe(frame, tick.t_ns, out)
                states.set_mode(Mode.TRACKING)
            states.set_frame(frame)
            if mode == Mode.RELOC:
                self._hand_over(RelocRequest(frames.frame_index), tick.t_ns, out)
                self.reloc_task = self.issued
        else:
            raise RuntimeError(f"frontend stepped in mode {mode}")
        self.held = (tick.t_ns, frames.frame_index)

    def flush(self, out: Emit) -> None:
        """Lossless: every task applied; then TERMINATED (realtime: at once, the backend drops what is still queued)."""
        self._settle(out)
        self.shared.states.set_mode(Mode.TERMINATED)

    def abort(self) -> None:
        self.aborted = True
        self.shared.wake()
