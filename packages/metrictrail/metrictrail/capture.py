"""Log MetricTrail's live intermediates to Rerun while ViPE runs, through DPVO's observer hooks and by wrapping ViPE methods
(no ViPE file is edited).

DPVO (``dpvo.metric.MetricDPVO``, which ViPE builds itself through ``vipe.metrictrail.dpvo.dpvo.DPVO``) is observed,
not wrapped: :meth:`CaptureSession.install` registers :meth:`CaptureSession.observe` with
:func:`dpvo.hooks.add_global_observer`, so every DPVO built while the session is installed

* opens one recording file per DPVO instance (pass);
* after each tracked frame (:meth:`CaptureObserver.on_frame`), logs the frame DPVO saw, the MoGe-2 metric depth it was
  given, the frame's MoGe-2 normals when DPVO's keyframe graph grew on it, the newest keyframe's patches (boxes of
  their P x P footprint on the shown image), the local bundle-adjustment window and its patch points;
* counts global bundle adjustments (:meth:`CaptureObserver.on_global_ba`).

Wrapped (class attributes of whichever ``vipe`` is imported; :meth:`CaptureSession.uninstall` restores them):

* ``vipe.metrictrail.slam.DPVOSLAMSystem.run``: notes the frame count of the tracking pass that starts;
* :meth:`metrictrail.moge2.MoGe2NormalProvider.predict_metric` and ``DPVOSLAMSystem._prepare``: keep the camera-space
  normals of the forward that gave each tracked frame its depth (keyed by the tracked-frame index, as 8-bit RGB), so a
  keyframe's normals are logged and the fused cloud gets per-point normals (:meth:`CaptureSession.kept_normals`);
* with :attr:`CaptureSession.live_surface` (the app), whenever DPVO's keyframe graph grows on a frame (DPVO adds
  every frame as a keyframe; the graph grows when keyframe culling keeps it), that frame's depth is integrated at
  DPVO's current estimate of its pose into the :class:`~metrictrail.live_surface.LiveSurface`, so the surface grows
  while DPVO tracks;
* ``vipe.metrictrail.dino_fusion.FrozenDinoV3.forward``: projects the dinov3_vits16 patch features of the frame being
  patchified on a fixed PCA basis (fit on the first frame) to RGB.

A run can make several DPVO passes (stage A, FP32 recovery). Each pass is written to its own ``.rrd``;
:meth:`CaptureSession.kept_pass` is the last of the longest passes, the one whose poses ViPE writes. With
:attr:`CaptureSession.live` set (the Gradio app), every pass logs straight into that recording instead, so the viewer
sees tracking as it happens; a later pass (FP32 recovery) then overwrites the earlier one frame by frame. The hooks read
tensors and add host syncs, so a captured run's FPS is not a timing result; they never change a value DPVO uses. An
unmodified upstream ViPE (``--label upstream``) runs its own vendored DPVO, which has no observer hooks: its passes are
not captured.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import torch
from dpvo.dpvo import DPVO
from dpvo.hooks import DPVOObserver, FrameEvent, add_global_observer
from jaxtyping import Bool, Float32, Float64, Int64, UInt8
from numpy import ndarray
from scipy.spatial.transform import Rotation
from simplecv.camera_orient_utils import auto_orient_and_center_poses
from torch import Tensor

from metrictrail import entities
from metrictrail.live_surface import LiveSurface
from metrictrail.palette import BA_WINDOW_RGB, KEYFRAME_RGB, depth_colors
from metrictrail.prior_log import PriorLog
from metrictrail.rerun_log import FrameVideo, SourceRaster, log_pinholes, moving_camera, set_frame_time


def dpvo_world_T_cam(poses: Float32[ndarray, "n 7"]) -> Float32[ndarray, "n 4 4"]:
    """Camera-to-world matrices from DPVO poses, which store world-to-camera SE(3) as ``tx ty tz qx qy qz qw``."""
    cam_R_world: Float32[ndarray, "n 3 3"] = Rotation.from_quat(poses[:, 3:7].astype(np.float64)).as_matrix().astype(np.float32)
    world_R_cam: Float32[ndarray, "n 3 3"] = np.transpose(cam_R_world, (0, 2, 1))
    world_T_cam: Float32[ndarray, "n 4 4"] = np.tile(np.eye(4, dtype=np.float32), (poses.shape[0], 1, 1))
    world_T_cam[:, :3, :3] = world_R_cam
    world_T_cam[:, :3, 3] = -np.einsum("nij,nj->ni", world_R_cam, poses[:, :3])
    return world_T_cam


RDF_TO_RUB: Float32[ndarray, "4 4"] = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
"""Right-multiplied onto an OpenCV (RDF) camera pose, gives the OpenGL (RUB) pose ``auto_orient_and_center_poses`` takes."""
ORIENT_EVERY_KEYFRAMES: int = 10
"""Recompute the gravity alignment of the live DPVO scene every this many keyframes (as dpvo's app does)."""


def dpvo_orient(world_T_cam: Float32[ndarray, "n 4 4"]) -> tuple[Float32[ndarray, "3 3"], Float32[ndarray, "3"]]:
    """The transform that turns DPVO's world (the first camera's RDF frame, y down) upright for a z-up viewer and centres
    it on the cameras: the cameras' mean up becomes +z (simplecv ``auto_orient_and_center_poses``, method ``up``), as
    the dpvo and MASt3R-SLAM apps orient their live scenes."""
    orient: Float64[ndarray, "3 4"] = auto_orient_and_center_poses(world_T_cam @ RDF_TO_RUB, method="up", center_method="poses").transform
    return orient[:, :3].astype(np.float32), orient[:, 3].astype(np.float32)


def host_rgb(image: UInt8[Tensor, "3 h w"]) -> UInt8[ndarray, "h w 3"]:
    """A tracked frame (DPVO's BGR, channels first, on its device) as host RGB."""
    return image.flip(0).permute(1, 2, 0).contiguous().cpu().numpy()


def normals_rgb(normals: Float32[Tensor, "h w 3"]) -> UInt8[Tensor, "h w 3"]:
    """Unit normals as 8-bit RGB, ``(n + 1) / 2`` per axis (monoprior's normals colouring)."""
    return ((normals.float() + 1.0) * 127.5).round().clamp(0.0, 255.0).to(torch.uint8)


def rgb_normals(rgb: UInt8[ndarray, "*batch 3"]) -> Float32[ndarray, "*batch 3"]:
    """Unit normals back from :func:`normals_rgb` (renormalised; 8-bit steps of about 0.008)."""
    normals: Float32[ndarray, "*batch 3"] = rgb.astype(np.float32) / 127.5 - 1.0
    length: Float32[ndarray, "*batch 1"] = np.linalg.norm(normals, axis=-1, keepdims=True)
    return (normals / np.maximum(length, 1e-6)).astype(np.float32)


def frustum_strips(
    world_T_cam: Float32[ndarray, "n 4 4"], intrinsics_px: Float32[ndarray, "4"], image_hw: tuple[int, int], depth_m: float
) -> list[Float32[ndarray, "k 3"]]:
    """Wireframe frustums ``depth_m`` deep: per camera one closed far rectangle and four rays from the centre."""
    height, width = image_hw
    fx, fy, cx, cy = (float(value) for value in intrinsics_px)
    corners_px: Float32[ndarray, "4 2"] = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    rays: Float32[ndarray, "4 3"] = np.stack([(corners_px[:, 0] - cx) / fx, (corners_px[:, 1] - cy) / fy, np.ones(4, dtype=np.float32)], -1)
    strips: list[Float32[ndarray, "k 3"]] = []
    for pose in world_T_cam:
        corners_world: Float32[ndarray, "4 3"] = ((rays * depth_m) @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)
        centre: Float32[ndarray, "3"] = pose[:3, 3]
        strips.append(np.vstack([corners_world, corners_world[:1]]).astype(np.float32))
        strips.extend(np.stack([centre, corner]).astype(np.float32) for corner in corners_world)
    return strips


@dataclass(slots=True)
class DinoPca:
    """A fixed PCA basis of DINOv3 patch features, so the colours of a run stay stable over time."""

    mean: Float32[Tensor, "c"]
    """Feature mean of the fitting frame."""
    basis: Float32[Tensor, "c 3"]
    """First three principal directions."""
    low: Float32[Tensor, "3"]
    """1st percentile of each projected component (maps to 0)."""
    high: Float32[Tensor, "3"]
    """99th percentile of each projected component (maps to 255)."""

    @staticmethod
    def fit(features: Float32[Tensor, "c h w"]) -> "DinoPca":
        """Fit the basis on one frame's features."""
        tokens: Float32[Tensor, "hw c"] = features.flatten(1).T.float()
        mean: Float32[Tensor, "c"] = tokens.mean(0)
        centred: Float32[Tensor, "hw c"] = tokens - mean
        right_vectors: Float32[Tensor, "k c"] = torch.linalg.svd(centred, full_matrices=False).Vh
        basis: Float32[Tensor, "c 3"] = right_vectors[:3].T.contiguous()
        projected: Float32[Tensor, "hw 3"] = centred @ basis
        return DinoPca(mean=mean, basis=basis, low=torch.quantile(projected, 0.01, dim=0), high=torch.quantile(projected, 0.99, dim=0))

    def rgb(self, features: Float32[Tensor, "c h w"]) -> UInt8[ndarray, "h w 3"]:
        """Project one frame's features to RGB."""
        tokens: Float32[Tensor, "hw c"] = features.flatten(1).T.float()
        unit: Float32[Tensor, "hw 3"] = (((tokens - self.mean) @ self.basis - self.low) / (self.high - self.low)).clamp(0.0, 1.0)
        return (unit * 255.0).round().to(torch.uint8).reshape(features.shape[1], features.shape[2], 3).cpu().numpy()


@dataclass(frozen=True, slots=True)
class CapturedPass:
    """One DPVO instance's capture recording."""

    path: Path | None
    """Its ``.rrd`` file; None when the pass logged into :attr:`CaptureSession.live`."""
    num_frames: int
    """Frames of the tracking pass it ran over."""


@dataclass(slots=True)
class CaptureSession:
    """Hooks state for one pipeline run; install before the pipeline is built."""

    out_dir: Path
    """Directory for the per-pass recordings ``pass_<k>.rrd``."""
    application_id: str
    """Application id of the run recording (the passes are merged into it)."""
    recording_id: str
    """Recording id of the run recording."""
    depth_range_m: tuple[float, float]
    """Turbo range of the depth-coloured patches, metres."""
    dino_patch_px: int = 16
    """DINOv3 patch size, pixels (dinov3_vits16)."""
    frustum_depth_m: float = 0.08
    """Depth of the BA-window frustums, metres."""
    live: rr.RecordingStream | None = None
    """Recording every pass logs into instead of its own file (the app's streamed recording); None: one file per pass."""
    priors: PriorLog | None = None
    """Logs MoGe-2's depth and normals of every tracked frame off the tracking thread (made by :meth:`install`)."""
    video_time_ns: Int64[ndarray, "n"] | None = None
    """``video_time`` of each source frame (the input video's PTS, or the benchmark's stamps); None: no ``video_time``."""
    source_raster: Callable[[], SourceRaster | None] | None = None
    """How the input reached the tracking raster, once ViPE prepared it (:class:`metrictrail.gpu_video.GpuVideoSession`);
    a known raster means the input video is logged (``input/video``): the patches go onto it and ``cam/rgb`` is not
    logged. None or a None result: ``cam/rgb`` is the tracked frames as a video."""
    fps: float = 30.0
    """Frame rate the derived videos are encoded at (the rate does not set their times; the timelines do)."""
    on_frame: Callable[[int, int | None, int, int], None] | None = None
    """Called after each tracked frame with (frame index, its source frame or None, frames of the pass, keyframes in the
    graph); may raise to stop."""
    live_surface: LiveSurface | None = None
    """TSDF mesh grown while tracking from each frame on which DPVO's keyframe graph grew, reset per pass (logging into
    :attr:`live`); None: none."""
    passes: list[CapturedPass] = field(default_factory=list)
    """Every DPVO instance captured, in creation order."""
    pass_frames: int = 0
    """Frame count of the tracking pass that is running."""
    global_ba_calls: int = 0
    """Global bundle adjustments of the current DPVO instance."""
    oriented_at: int = 0
    """Keyframe count of the pass when the live scene was last oriented (0: not yet)."""
    pca: DinoPca | None = None
    """DINOv3 PCA basis, fit on the first frame DINOv3 sees."""
    last_dino_rgb: UInt8[ndarray, "h w 3"] | None = None
    """PCA image of the latest DINOv3 call (the frame DPVO is patchifying)."""
    source_frames: dict[int, int] = field(default_factory=dict)
    """Source frame of each tracked-frame index (ViPE's ``PreparedFrame.source_frame_id``); the same for every pass."""
    videos: dict[str, FrameVideo] = field(default_factory=dict)
    """The derived videos of the current pass, by entity."""
    streams: dict[int, rr.RecordingStream] = field(default_factory=dict)
    """Open recording per DPVO instance, keyed by ``id(instance)``."""
    restore: list[tuple[type, str, Callable[..., Any]]] = field(default_factory=list)
    """Wrapped methods and their originals."""
    remove_observer: Callable[[], None] | None = None
    """Unregisters :meth:`observe` from :mod:`dpvo.hooks`; None while not installed."""
    pending_normals: dict[int, UInt8[Tensor, "h w 3"]] = field(default_factory=dict)
    """Normals (8-bit RGB, on the GPU) of frames the depth producer prepared and DPVO has not tracked yet, by index."""
    pass_normals: list[dict[int, UInt8[Tensor, "h w 3"]]] = field(default_factory=list)
    """Per pass (same order as :attr:`passes`), the normals of every tracked frame in page-locked host memory (copied
    without a synchronisation); superseded passes are emptied."""
    producer_local: threading.local = field(default_factory=threading.local)
    """The normals of the latest MoGe-2 prediction of the calling thread (the depth producer runs in its own)."""

    def install(self) -> None:
        """Observe every DPVO built from now on and wrap the ViPE classes of the imported ``vipe`` tree."""
        self.priors = PriorLog(self.depth_range_m, self.video_time_ns, self.fps)
        import vipe.metrictrail.dino_fusion as dino_fusion
        import vipe.metrictrail.slam as slam

        from metrictrail.moge2 import MoGe2NormalProvider

        self.out_dir.mkdir(parents=True, exist_ok=True)
        session: CaptureSession = self
        run_orig: Callable[..., Any] = slam.DPVOSLAMSystem.run
        forward_orig: Callable[..., Any] = dino_fusion.FrozenDinoV3.forward
        predict_orig: Callable[..., Any] = MoGe2NormalProvider.predict_metric
        prepare_orig: Callable[..., Any] = slam.DPVOSLAMSystem._prepare

        def run(self: Any, video_stream: Any, **kwargs: Any) -> Any:
            session.pass_frames = len(video_stream)
            return run_orig(self, video_stream, **kwargs)

        def predict(self: Any, rgb_bchw: Tensor, *, focal_px: float) -> Any:
            prediction: Any = predict_orig(self, rgb_bchw, focal_px=focal_px)
            if prediction.normals.shape[0] == 1:
                # Queued on the producer's stream before the frame's ``ready`` event, so DPVO's wait covers it.
                session.producer_local.normals = normals_rgb(prediction.normals[0])
            return prediction

        def prepare(self: Any, index: int, *args: Any, **kwargs: Any) -> Any:
            session.producer_local.normals = None
            item: Any = prepare_orig(self, index, *args, **kwargs)
            source_frame: Any = getattr(item, "source_frame_id", None)
            if source_frame is not None:
                session.source_frames[int(index)] = int(source_frame)
            normals: Tensor | None = getattr(session.producer_local, "normals", None)
            if normals is not None:
                session.pending_normals[int(index)] = normals
            session.producer_local.normals = None
            return item

        def forward(self: Any, dpvo_images: Tensor) -> Tensor:
            features: Tensor = forward_orig(self, dpvo_images)
            if features.shape[0] * features.shape[1] == 1:
                with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
                    single: Float32[Tensor, "c h w"] = features[0, 0].float()
                    if session.pca is None:
                        session.pca = DinoPca.fit(single)
                    session.last_dino_rgb = session.pca.rgb(single)
            return features

        self.restore = [
            (slam.DPVOSLAMSystem, "run", run_orig),
            (dino_fusion.FrozenDinoV3, "forward", forward_orig),
            (MoGe2NormalProvider, "predict_metric", predict_orig),
            (slam.DPVOSLAMSystem, "_prepare", prepare_orig),
        ]
        slam.DPVOSLAMSystem.run = run
        dino_fusion.FrozenDinoV3.forward = forward
        MoGe2NormalProvider.predict_metric = predict
        slam.DPVOSLAMSystem._prepare = prepare
        self.remove_observer = add_global_observer(self.observe)

    def observe(self, dpvo: DPVO) -> "CaptureObserver":
        """The global observer factory: open the pass of a new DPVO instance and observe it."""
        self.open_pass(id(dpvo))
        return CaptureObserver(self)

    def uninstall(self) -> None:
        """Stop observing new DPVO instances, restore the wrapped methods and close every pass recording."""
        if self.remove_observer is not None:
            self.remove_observer()
            self.remove_observer = None
        for owner, name, original in self.restore:
            setattr(owner, name, original)
        self.restore = []
        if self.live_surface is not None:
            self.live_surface.finish()  # the whole surface as one mesh at the pass's last integrated frame
        if self.priors is not None:
            self.priors.close()
        self.close_videos()
        for rec in self.streams.values():
            rec.flush()
            if rec is not self.live:
                rec.disconnect()
        self.streams = {}

    def open_pass(self, instance_id: int) -> None:
        """Start the recording file of a new DPVO instance."""
        self.global_ba_calls = 0
        self.oriented_at = 0
        if self.priors is not None:
            self.priors.drain()  # the previous pass's depth and normals, before its recording closes
        self.close_videos()
        path: Path | None = None
        if self.live is not None:
            rec: rr.RecordingStream = self.live
        else:
            path = self.out_dir / f"pass_{len(self.passes)}.rrd"
            rec = rr.RecordingStream(self.application_id, recording_id=self.recording_id, send_properties=False)
            rec.save(path)
        self.streams[instance_id] = rec
        rec.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)  # the overview's up; dpvo_orient makes it the scene's
        # A pass at least as long supersedes the earlier ones (kept_pass), so their normals can go.
        for earlier, captured in zip(self.pass_normals, self.passes, strict=True):
            if captured.num_frames <= self.pass_frames:
                earlier.clear()
        self.passes.append(CapturedPass(path=path, num_frames=self.pass_frames))
        if self.live_surface is not None:
            self.live_surface.open_pass()
        self.pass_normals.append({})
        self.pending_normals.clear()

    def close_videos(self) -> None:
        """Flush the derived videos of the pass that ends."""
        for video in self.videos.values():
            video.close()
        self.videos = {}

    def video(self, rec: rr.RecordingStream, entity: str) -> FrameVideo:
        """The pass's derived video at ``entity``, started on first use."""
        if entity not in self.videos:
            self.videos[entity] = FrameVideo(rec, entity, self.fps)
        return self.videos[entity]

    def kept_pass(self) -> CapturedPass | None:
        """The last of the longest passes (the tracking pass whose poses ViPE writes), or None if nothing ran."""
        if not self.passes:
            return None
        longest: int = max(captured.num_frames for captured in self.passes)
        return [captured for captured in self.passes if captured.num_frames == longest][-1]

    def kept_normals(self) -> dict[int, UInt8[ndarray, "h w 3"]]:
        """The kept pass's normals (8-bit RGB) by tracked-frame index; empty without a pass or without normals."""
        kept: CapturedPass | None = self.kept_pass()
        if kept is None:
            return {}
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # the asynchronous copies into the pinned buffers are complete
        return {frame: normals.numpy() for frame, normals in self.pass_normals[self.passes.index(kept)].items()}

    def log_frame(
        self,
        rec: rr.RecordingStream,
        dpvo: DPVO,
        frame: int,
        image: Tensor,
        intrinsics: Tensor,
        depth: Tensor | None,
        added: bool,
    ) -> str | None:
        """Log one tracked frame of a DPVO instance (see the module docstring); returns the patches entity when it
        logged patches. ``depth``: MoGe-2's metric depth (0 = none) on its device, for the live surface, on frames the
        keyframe graph grew on; None otherwise (:attr:`priors` logs every frame's)."""
        set_frame_time(rec, frame, self.source_frames.get(frame), self.video_time_ns)
        height, width = int(image.shape[-2]), int(image.shape[-1])
        intrinsics_px: Float32[ndarray, "4"] = intrinsics.detach().float().cpu().numpy().reshape(4)
        raster: SourceRaster | None = self.source_raster() if self.source_raster is not None else None
        log_pinholes(rec, intrinsics_px.astype(np.float64), (height, width), raster)
        if raster is None:
            self.video(rec, entities.CAM_RGB).log(host_rgb(image))
        if self.last_dino_rgb is not None:
            upsampled: UInt8[ndarray, "H W 3"] = self.last_dino_rgb.repeat(self.dino_patch_px, 0).repeat(self.dino_patch_px, 1)[:height, :width]
            self.video(rec, entities.CAM_DINO_PCA).log(upsampled)
        if not added:
            return None
        num_keyframes: int = int(dpvo.n)
        patches_per_frame: int = int(dpvo.M)
        resolution: int = int(dpvo.RES)
        centre: int = int(dpvo.P) // 2
        newest: Float32[ndarray, "m 3"] = (
            dpvo.patches[0, (num_keyframes - 1) * patches_per_frame : num_keyframes * patches_per_frame, :, centre, centre].float().cpu().numpy()
        )
        inverse_depth: Float32[ndarray, "m"] = newest[:, 2]
        patch_depth_m: Float32[ndarray, "m"] = np.where(inverse_depth > 1e-6, 1.0 / np.maximum(inverse_depth, 1e-6), 0.0).astype(np.float32)
        patches_px: Float32[ndarray, "m 2"] = newest[:, :2] * resolution
        # A patch is P x P feature cells at DPVO's stride, centred on its coordinate: a box, not a point.
        half_sizes: Float32[ndarray, "m 2"] = np.full_like(patches_px, int(dpvo.P) * resolution / 2.0)
        if raster is not None:
            patches_px = raster.source_px(patches_px)
            half_sizes = (half_sizes * np.array(raster.scale_xy(), dtype=np.float32)).astype(np.float32)
        patches_entity: str = entities.CAM_PATCHES if raster is None else entities.INPUT_PATCHES
        rec.log(
            patches_entity,
            rr.Boxes2D(
                centers=patches_px,
                half_sizes=half_sizes,
                radii=rr.Radius.ui_points(1.5),
                colors=depth_colors(patch_depth_m, self.depth_range_m),
            ),
        )
        rec.log(f"{entities.DPVO_METRICS}/patch_depth_median_m", rr.Scalars(float(np.median(patch_depth_m))))
        if not dpvo.is_initialized:
            return patches_entity
        window_start: int = max(num_keyframes - dpvo.cfg.optimization_window, 1)
        world_T_cam: Float32[ndarray, "n 4 4"] = dpvo_world_T_cam(dpvo.poses[0, :num_keyframes].float().cpu().numpy())
        centres: Float32[ndarray, "n 3"] = world_T_cam[:, :3, 3]
        in_window: Bool[ndarray, "n"] = np.arange(num_keyframes) >= window_start
        rec.log(
            entities.DPVO_KEYFRAMES,
            rr.Points3D(
                centres,
                colors=np.where(in_window[:, None], BA_WINDOW_RGB, KEYFRAME_RGB).astype(np.uint8),
                radii=np.where(in_window, 0.012, 0.006).astype(np.float32),
            ),
        )
        rec.log(entities.DPVO_KEYFRAME_PATH, rr.LineStrips3D([centres], colors=[KEYFRAME_RGB], radii=rr.Radius.ui_points(1.0)))
        full_px: Float32[ndarray, "4"] = dpvo.intrinsics[0, num_keyframes - 1].float().cpu().numpy() * resolution
        rec.log(
            entities.DPVO_WINDOW,
            rr.LineStrips3D(
                frustum_strips(world_T_cam[window_start:num_keyframes], full_px, (height, width), self.frustum_depth_m),
                colors=[BA_WINDOW_RGB],
                radii=rr.Radius.ui_points(1.5),
            ),
        )
        newest_world_T_cam: Float32[ndarray, "4 4"] = world_T_cam[num_keyframes - 1]
        if self.live_surface is not None and depth is not None:
            self.live_surface.integrate(depth.cpu().numpy(), host_rgb(image), intrinsics_px, newest_world_T_cam, frame)
        if num_keyframes >= 3 and (self.oriented_at == 0 or num_keyframes - self.oriented_at >= ORIENT_EVERY_KEYFRAMES):
            orient_rotation, orient_translation = dpvo_orient(world_T_cam)
            rec.log(entities.DPVO_ROOT, rr.Transform3D(mat3x3=orient_rotation, translation=orient_translation))
            self.oriented_at = num_keyframes
        rec.log(entities.DPVO_CURRENT, rr.Transform3D(translation=newest_world_T_cam[:3, 3], mat3x3=newest_world_T_cam[:3, :3]))
        if raster is None:  # a video input's frustum is the source camera, logged with the frame (log_pinholes)
            rec.log(entities.DPVO_CURRENT_PINHOLE, moving_camera(full_px.astype(np.float64), (height, width)))
        num_patches: int = int(dpvo.m)
        patch_frame: Int64[ndarray, "m"] = dpvo.ix[:num_patches].cpu().numpy().astype(np.int64)
        keep: Bool[ndarray, "m"] = patch_frame >= window_start
        points_world: Float32[ndarray, "k 3"] = dpvo.pg.points_[:num_patches].float().cpu().numpy()[keep]
        colors_rgb: UInt8[ndarray, "k 3"] = dpvo.pg.colors_.view(-1, 3)[:num_patches].cpu().numpy()[keep][:, ::-1]
        rec.log(entities.DPVO_WINDOW_POINTS, rr.Points3D(points_world, colors=np.ascontiguousarray(colors_rgb), radii=rr.Radius.ui_points(3.0)))
        rec.log(f"{entities.DPVO_METRICS}/keyframes_in_graph", rr.Scalars(float(num_keyframes)))
        rec.log(f"{entities.DPVO_METRICS}/global_ba_calls", rr.Scalars(float(self.global_ba_calls)))
        return patches_entity

    def clear_after(self, rec: rr.RecordingStream, frame: int, entity: str) -> None:
        """Clear ``entity`` at the next tracked frame's time (its source frame when ViPE has prepared it, else the next
        source frame), so a viewer playing past the tracked progress shows the video without stale patches; the next
        frame's own log at that time wins (logged later)."""
        source: int | None = self.source_frames.get(frame)
        next_source: int | None = self.source_frames.get(frame + 1, source + 1 if source is not None else None)
        set_frame_time(rec, frame + 1, next_source, self.video_time_ns)
        rec.log(entity, rr.Clear(recursive=False))


class CaptureObserver(DPVOObserver):
    """Logs one DPVO pass into its :class:`CaptureSession` recording (built by :meth:`CaptureSession.observe`)."""

    def __init__(self, session: CaptureSession) -> None:
        self.session: CaptureSession = session

    def on_frame(self, dpvo: DPVO, event: FrameEvent) -> None:
        """Log the frame DPVO just tracked, then call :attr:`CaptureSession.on_frame` (which may raise to stop the run)."""
        session: CaptureSession = self.session
        frame: int = event.frame_index
        rec: rr.RecordingStream | None = session.streams.get(id(dpvo))
        normals: Tensor | None = session.pending_normals.pop(frame, None)
        if normals is not None and session.pass_normals:
            # A CUDA frame: an asynchronous copy into page-locked memory, the tracking loop never waits for it; fusion
            # reads it after the run, long after the copy finished. A CPU frame (no GPU): a plain copy.
            if normals.is_cuda:
                pinned: UInt8[Tensor, "h w 3"] = torch.empty(normals.shape, dtype=normals.dtype, pin_memory=True)
                session.pass_normals[-1][frame] = pinned.copy_(normals, non_blocking=True)
            else:
                session.pass_normals[-1][frame] = normals.clone()
        if rec is not None:
            # DPVO appends every frame to its keyframe graph, then may cull an older keyframe: the graph grew
            # when culling kept them all. The frame itself is always the newest keyframe after the call.
            graph_grew: bool = dpvo.n > event.keyframes_before
            added: bool = dpvo.is_initialized or graph_grew
            depth: Tensor | None = None
            if event.depth.depth is not None:
                metric: Tensor = event.depth.depth.float()
                valid: Tensor = torch.isfinite(metric) if event.depth.mask is None else event.depth.mask.bool()
                depth = torch.where(valid, metric, torch.zeros_like(metric))
            # The live surface integrates only frames the graph grew on (a TSDF integration and a device-to-host copy
            # per frame slowed tracking by a fifth); the priors below log every frame.
            surface_depth: Tensor | None = depth if graph_grew else None
            patches_entity: str | None = session.log_frame(rec, dpvo, frame, event.image, event.intrinsics, surface_depth, added)
            if session.priors is not None:  # every frame: MoGe-2 ran on it for DPVO anyway
                session.priors.submit(rec, frame, session.source_frames.get(frame), depth, normals)
            if patches_entity is not None:
                session.clear_after(rec, frame, patches_entity)
        session.last_dino_rgb = None
        if session.on_frame is not None:
            session.on_frame(frame, session.source_frames.get(frame), session.pass_frames, dpvo.n)

    def on_global_ba(self, dpvo: DPVO) -> None:
        """Count the global bundle adjustment that is about to run."""
        self.session.global_ba_calls += 1
