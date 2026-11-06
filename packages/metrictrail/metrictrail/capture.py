"""Log MetricTrail's live intermediates to Rerun while ViPE runs, by wrapping ViPE methods (no ViPE file is edited).

Wrapped (class attributes of whichever ``vipe`` is imported; :meth:`CaptureSession.uninstall` restores them):

* ``vipe.metrictrail.slam.DPVOSLAMSystem.run``: notes the frame count of the tracking pass that starts;
* ``vipe.metrictrail.dpvo.dpvo.DPVO.__init__``: opens one recording file per DPVO instance (pass);
* ``DPVO.__call__``: after each tracked frame, logs the frame DPVO saw, the MoGe-2 metric depth it was given, the
  MoGe-2 normals of a new keyframe, the newest keyframe's patches, the local bundle-adjustment window and its patch points;
* :meth:`metrictrail.moge2.MoGe2NormalProvider.predict_metric` and ``DPVOSLAMSystem._prepare``: keep the camera-space
  normals of the forward that gave each tracked frame its depth (keyed by the tracked-frame index, as 8-bit RGB), so a
  keyframe's normals are logged and the fused cloud gets per-point normals (:meth:`CaptureSession.kept_normals`);
* ``DPVO._DPVO__run_global_BA``: counts global bundle adjustments;
* ``vipe.metrictrail.dino_fusion.FrozenDinoV3.forward``: projects the dinov3_vits16 patch features of the frame being
  patchified on a fixed PCA basis (fit on the first frame) to RGB.

A run can make several DPVO passes (stage A, FP32 recovery). Each pass is written to its own ``.rrd``;
:meth:`CaptureSession.kept_pass` is the last of the longest passes, the one whose poses ViPE writes. The hooks read
tensors and add host syncs, so a captured run's FPS is not a timing result; they never change a value DPVO uses.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import torch
from jaxtyping import Bool, Float32, Int64, UInt8
from numpy import ndarray
from scipy.spatial.transform import Rotation
from torch import Tensor

from metrictrail import entities
from metrictrail.palette import BA_WINDOW_RGB, KEYFRAME_RGB, depth_colors


def dpvo_world_T_cam(poses: Float32[ndarray, "n 7"]) -> Float32[ndarray, "n 4 4"]:
    """Camera-to-world matrices from DPVO poses, which store world-to-camera SE(3) as ``tx ty tz qx qy qz qw``."""
    cam_R_world: Float32[ndarray, "n 3 3"] = Rotation.from_quat(poses[:, 3:7].astype(np.float64)).as_matrix().astype(np.float32)
    world_R_cam: Float32[ndarray, "n 3 3"] = np.transpose(cam_R_world, (0, 2, 1))
    world_T_cam: Float32[ndarray, "n 4 4"] = np.tile(np.eye(4, dtype=np.float32), (poses.shape[0], 1, 1))
    world_T_cam[:, :3, :3] = world_R_cam
    world_T_cam[:, :3, 3] = -np.einsum("nij,nj->ni", world_R_cam, poses[:, :3])
    return world_T_cam


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

    path: Path
    """Its ``.rrd`` file."""
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
    passes: list[CapturedPass] = field(default_factory=list)
    """Every DPVO instance captured, in creation order."""
    pass_frames: int = 0
    """Frame count of the tracking pass that is running."""
    global_ba_calls: int = 0
    """Global bundle adjustments of the current DPVO instance."""
    pca: DinoPca | None = None
    """DINOv3 PCA basis, fit on the first frame DINOv3 sees."""
    last_dino_rgb: UInt8[ndarray, "h w 3"] | None = None
    """PCA image of the latest DINOv3 call (the frame DPVO is patchifying)."""
    streams: dict[int, rr.RecordingStream] = field(default_factory=dict)
    """Open recording per DPVO instance, keyed by ``id(instance)``."""
    restore: list[tuple[type, str, Callable[..., Any]]] = field(default_factory=list)
    """Wrapped methods and their originals."""
    pending_normals: dict[int, UInt8[Tensor, "h w 3"]] = field(default_factory=dict)
    """Normals (8-bit RGB, on the GPU) of frames the depth producer prepared and DPVO has not tracked yet, by index."""
    pass_normals: list[dict[int, UInt8[ndarray, "h w 3"]]] = field(default_factory=list)
    """Per pass (same order as :attr:`passes`), the normals of every tracked frame; superseded passes are emptied."""
    producer_local: threading.local = field(default_factory=threading.local)
    """The normals of the latest MoGe-2 prediction of the calling thread (the depth producer runs in its own)."""

    def install(self) -> None:
        """Wrap the ViPE classes of the imported ``vipe`` tree."""
        import vipe.metrictrail.dino_fusion as dino_fusion
        import vipe.metrictrail.dpvo.dpvo as dpvo_module
        import vipe.metrictrail.slam as slam

        from metrictrail.moge2 import MoGe2NormalProvider

        self.out_dir.mkdir(parents=True, exist_ok=True)
        session: CaptureSession = self
        run_orig: Callable[..., Any] = slam.DPVOSLAMSystem.run
        init_orig: Callable[..., Any] = dpvo_module.DPVO.__init__
        call_orig: Callable[..., Any] = dpvo_module.DPVO.__call__
        gba_orig: Callable[..., Any] = dpvo_module.DPVO._DPVO__run_global_BA
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
            normals: Tensor | None = getattr(session.producer_local, "normals", None)
            if normals is not None:
                session.pending_normals[int(index)] = normals
            session.producer_local.normals = None
            return item

        def init(self: Any, *args: Any, **kwargs: Any) -> None:
            init_orig(self, *args, **kwargs)
            session.open_pass(id(self))

        def call(
            self: Any, tstamp: Any, image: Tensor, intrinsics: Tensor, metric_depth: Tensor | None = None, metric_mask: Tensor | None = None
        ) -> Any:
            frame: int = int(self.counter)
            keyframes_before: int = int(self.n)
            result: Any = call_orig(self, tstamp, image, intrinsics, metric_depth=metric_depth, metric_mask=metric_mask)
            rec: rr.RecordingStream | None = session.streams.get(id(self))
            normals: Tensor | None = session.pending_normals.pop(frame, None)
            normals_host: UInt8[ndarray, "h w 3"] | None = None
            if normals is not None:
                normals_host = normals.cpu().numpy()
                if session.pass_normals:
                    session.pass_normals[-1][frame] = normals_host
            if rec is not None:
                new_keyframe: bool = int(self.n) > keyframes_before
                added: bool = bool(self.is_initialized) or new_keyframe
                session.log_frame(rec, self, frame, image, intrinsics, metric_depth, metric_mask, added)
                if new_keyframe and normals_host is not None:
                    rec.log(entities.CAM_NORMALS, rr.Image(normals_host, color_model="RGB").compress(jpeg_quality=92))
            session.last_dino_rgb = None
            return result

        def gba(self: Any, *args: Any, **kwargs: Any) -> Any:
            session.global_ba_calls += 1
            return gba_orig(self, *args, **kwargs)

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
            (dpvo_module.DPVO, "__init__", init_orig),
            (dpvo_module.DPVO, "__call__", call_orig),
            (dpvo_module.DPVO, "_DPVO__run_global_BA", gba_orig),
            (dino_fusion.FrozenDinoV3, "forward", forward_orig),
            (MoGe2NormalProvider, "predict_metric", predict_orig),
            (slam.DPVOSLAMSystem, "_prepare", prepare_orig),
        ]
        slam.DPVOSLAMSystem.run = run
        dpvo_module.DPVO.__init__ = init
        dpvo_module.DPVO.__call__ = call
        dpvo_module.DPVO._DPVO__run_global_BA = gba
        dino_fusion.FrozenDinoV3.forward = forward
        MoGe2NormalProvider.predict_metric = predict
        slam.DPVOSLAMSystem._prepare = prepare

    def uninstall(self) -> None:
        """Restore the wrapped methods and close every pass recording."""
        for owner, name, original in self.restore:
            setattr(owner, name, original)
        self.restore = []
        for rec in self.streams.values():
            rec.flush()
            rec.disconnect()
        self.streams = {}

    def open_pass(self, instance_id: int) -> None:
        """Start the recording file of a new DPVO instance."""
        self.global_ba_calls = 0
        path: Path = self.out_dir / f"pass_{len(self.passes)}.rrd"
        rec: rr.RecordingStream = rr.RecordingStream(self.application_id, recording_id=self.recording_id, send_properties=False)
        rec.save(path)
        self.streams[instance_id] = rec
        # A pass at least as long supersedes the earlier ones (kept_pass), so their normals can go.
        for earlier, captured in zip(self.pass_normals, self.passes, strict=True):
            if captured.num_frames <= self.pass_frames:
                earlier.clear()
        self.passes.append(CapturedPass(path=path, num_frames=self.pass_frames))
        self.pass_normals.append({})
        self.pending_normals.clear()

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
        return self.pass_normals[self.passes.index(kept)]

    def log_frame(
        self,
        rec: rr.RecordingStream,
        dpvo: Any,
        frame: int,
        image: Tensor,
        intrinsics: Tensor,
        metric_depth: Tensor | None,
        metric_mask: Tensor | None,
        added: bool,
    ) -> None:
        """Log one tracked frame of a DPVO instance (see the module docstring)."""
        rec.set_time(entities.FRAME_TIMELINE, sequence=frame)
        rgb: UInt8[ndarray, "h w 3"] = image.flip(0).permute(1, 2, 0).contiguous().cpu().numpy()
        height, width = rgb.shape[:2]
        intrinsics_px: Float32[ndarray, "4"] = intrinsics.detach().float().cpu().numpy().reshape(4)
        rec.log(entities.CAM, rr.Pinhole(focal_length=intrinsics_px[:2], principal_point=intrinsics_px[2:], resolution=[width, height]))
        rec.log(entities.CAM_RGB, rr.Image(rgb, color_model="RGB").compress(jpeg_quality=92))
        if metric_depth is not None:
            valid: Tensor = torch.isfinite(metric_depth) if metric_mask is None else metric_mask.bool()
            depth_m: Float32[ndarray, "h w"] = torch.where(valid, metric_depth.float(), torch.zeros_like(metric_depth.float())).cpu().numpy()
            depth_mm: ndarray = np.clip(np.round(depth_m * 1000.0), 0, 65535).astype(np.uint16)
            rec.log(entities.CAM_DEPTH, rr.DepthImage(depth_mm, meter=1000.0, colormap="turbo", depth_range=[r * 1000.0 for r in self.depth_range_m]))
        if self.last_dino_rgb is not None:
            upsampled: UInt8[ndarray, "H W 3"] = self.last_dino_rgb.repeat(self.dino_patch_px, 0).repeat(self.dino_patch_px, 1)[:height, :width]
            rec.log(entities.CAM_DINO_PCA, rr.Image(np.ascontiguousarray(upsampled), color_model="RGB"))
        if not added:
            return
        num_keyframes: int = int(dpvo.n)
        patches_per_frame: int = int(dpvo.M)
        resolution: int = int(dpvo.RES)
        centre: int = int(dpvo.P) // 2
        newest: Float32[ndarray, "m 3"] = (
            dpvo.patches[0, (num_keyframes - 1) * patches_per_frame : num_keyframes * patches_per_frame, :, centre, centre].float().cpu().numpy()
        )
        inverse_depth: Float32[ndarray, "m"] = newest[:, 2]
        patch_depth_m: Float32[ndarray, "m"] = np.where(inverse_depth > 1e-6, 1.0 / np.maximum(inverse_depth, 1e-6), 0.0).astype(np.float32)
        rec.log(
            entities.CAM_PATCHES,
            rr.Points2D(newest[:, :2] * resolution, radii=rr.Radius.ui_points(4.0), colors=depth_colors(patch_depth_m, self.depth_range_m)),
        )
        rec.log(f"{entities.DPVO_METRICS}/patch_depth_median_m", rr.Scalars(float(np.median(patch_depth_m))))
        if not dpvo.is_initialized:
            return
        window_start: int = max(num_keyframes - int(dpvo.cfg.OPTIMIZATION_WINDOW), 1)
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
        rec.log(entities.DPVO_CURRENT, rr.Transform3D(translation=newest_world_T_cam[:3, 3], mat3x3=newest_world_T_cam[:3, :3]))
        rec.log(
            entities.DPVO_CURRENT,
            rr.Pinhole(
                focal_length=full_px[:2],
                principal_point=full_px[2:],
                resolution=[width, height],
                image_plane_distance=self.frustum_depth_m,
                color=BA_WINDOW_RGB,
                line_width=-2.5,  # negative = UI points (rr.Radius.ui_points); a positive value is metres
            ),
        )
        num_patches: int = int(dpvo.m)
        patch_frame: Int64[ndarray, "m"] = dpvo.ix[:num_patches].cpu().numpy().astype(np.int64)
        keep: Bool[ndarray, "m"] = patch_frame >= window_start
        points_world: Float32[ndarray, "k 3"] = dpvo.pg.points_[:num_patches].float().cpu().numpy()[keep]
        colors_rgb: UInt8[ndarray, "k 3"] = dpvo.pg.colors_.view(-1, 3)[:num_patches].cpu().numpy()[keep][:, ::-1]
        rec.log(entities.DPVO_WINDOW_POINTS, rr.Points3D(points_world, colors=np.ascontiguousarray(colors_rgb), radii=rr.Radius.ui_points(3.0)))
        rec.log(f"{entities.DPVO_METRICS}/keyframes_in_graph", rr.Scalars(float(num_keyframes)))
        rec.log(f"{entities.DPVO_METRICS}/global_ba_calls", rr.Scalars(float(self.global_ba_calls)))
