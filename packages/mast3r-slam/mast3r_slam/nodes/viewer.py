"""The App's viewer sink: every value draws itself (its Type); the run's history is drawn here, as the CLI drew it."""

import time
from collections.abc import Callable, Mapping

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from dagkit.node import Emit, PortType, RunContext, Tick
from dagkit.sinks import RerunSink
from jaxtyping import Float32, Float64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from simplecv.camera_orient_utils import auto_orient_and_center_poses
from simplecv.ops import conventions

from mast3r_slam.types import FrameView, MapUpdate


class SlamViewer(RerunSink):
    """Draw the run in the viewer.

    Each value draws itself; from the run's history this sink adds what the CLI drew: the input video under the current
    camera, the camera path, each keyframe's frustum following the optimised poses (its cloud at the keyframe's Sim3
    scale), the factor graph, the world turned upright from the keyframe poses, the text log, and the blueprint again when
    a keyframe arrives (its depth image is hidden from the 3D view by name)."""

    def __init__(
        self,
        recording: rr.RecordingStream,
        inputs: Mapping[str, PortType],
        roots: Mapping[str, str],
        *,
        root: str,
        blueprint: Callable[[int], rrb.Blueprint],
        about: list[str],
        video: Callable[[rr.RecordingStream], None] | None = None,
    ) -> None:
        super().__init__(recording, inputs, roots)
        self.root, self.blueprint, self.about, self.video = root, blueprint, about, video
        """``blueprint(n)`` = the viewer's blueprint for ``n`` keyframes; ``about`` = the log's first lines; ``video`` logs the input video."""

    def reset(self, ctx: RunContext) -> None:
        self.path: list[Float32[ndarray, " 3"]] = []
        self.poses: Float32[ndarray, "k 8"] = np.zeros((0, 8), np.float32)
        self.frames, self.started = 0, time.perf_counter()
        self.t_ns: int | None = None
        if self.video is not None:
            self.video(self.recording)

    def _text(self, text: str) -> None:
        self.recording.log(f"{self.root}/logs", rr.TextLog(text, level="INFO"))

    def step(self, tick: Tick, out: Emit) -> None:
        if self.t_ns is None:  # the log's first lines, at the run's first time
            self.recording.set_time(self.timeline, duration=np.timedelta64(tick.t_ns, "ns"))
            for line in self.about:
                self._text(line)
        self.t_ns = max(self.t_ns if self.t_ns is not None else tick.t_ns, tick.t_ns)  # final keyframe poses come late
        super().step(tick, out)
        if isinstance(tick.value, FrameView):
            self.path.append(tick.value.current.world_sim3_cam[:3])
            self.recording.log(f"{self.root}/path", rr.LineStrips3D([np.stack(self.path)], colors=(255, 0, 0), labels="Camera Path"))
            self.frames += 1
            if self.frames % 30 == 0:
                self._text(f"FPS: {self.frames / (time.perf_counter() - self.started):.1f}")
        elif isinstance(tick.value, MapUpdate):
            self._map(tick.value)

    def _map(self, update: MapUpdate) -> None:
        """Keyframe frusta (rigid) and clouds (Sim3 scale) where the optimiser moved them, the factor graph, and on a new
        keyframe the upright world (the CLI's gravity alignment of the keyframe poses, OpenGL camera axes) and the blueprint."""
        poses: Float32[ndarray, "k 8"] = update.world_sim3_cam
        old: int = len(self.poses)
        moved: list[int] = [i for i in range(len(poses)) if i >= old or not np.array_equal(poses[i], self.poses[i])]
        for i in moved:
            keyframe: str = f"{self.root}/keyframes/keyframe-{i}"
            self.recording.log(keyframe, rr.Transform3D(translation=poses[i, :3], quaternion=rr.Quaternion(xyzw=poses[i, 3:7])))
            self.recording.log(f"{keyframe}/pointcloud", rr.Transform3D(scale=float(poses[i, 7])))
        centres: Float32[ndarray, "k 3"] = poses[:, :3]
        if len(update.ii):
            edges = np.stack([centres[update.ii], centres[update.jj]], axis=1)
            self.recording.log(f"{self.root}/edges", rr.LineStrips3D(edges, colors=(0, 255, 0), labels="Factor Graph"))
        if len(poses) > old:
            world_from_cam: Float64[ndarray, "k 4 4"] = np.tile(np.eye(4), (len(poses), 1, 1))
            world_from_cam[:, :3, :3] = Rotation.from_quat(poses[:, 3:7]).as_matrix()
            world_from_cam[:, :3, 3] = centres
            gl: Float64[ndarray, "k 4 4"] = conventions.convert_pose(world_from_cam, conventions.CC.CV, conventions.CC.GL)
            orient: Float64[ndarray, "3 4"] = auto_orient_and_center_poses(gl, method="up", center_method="poses").transform
            self.recording.log(self.root, rr.Transform3D(mat3x3=orient[:, :3], translation=orient[:, 3]))
            self.recording.send_blueprint(self.blueprint(len(poses)))
        self.poses = poses

    def flush(self, out: Emit) -> None:
        if self.t_ns is not None:
            self.recording.set_time(self.timeline, duration=np.timedelta64(self.t_ns, "ns"))
            minutes, seconds = divmod(int(time.perf_counter() - self.started), 60)
            for line in (f"Inference time: {minutes:02d}:{seconds:02d}", f"Processed {len(self.poses)} keyframes", "Done"):
                self._text(line)
            self.t_ns = None
        super().flush(out)
