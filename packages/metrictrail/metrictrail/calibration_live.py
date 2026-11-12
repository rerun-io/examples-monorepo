"""SharedK calibration, live: what ViPE's calibration estimator does before DPVO tracks, as it does it.

With unknown intrinsics MetricTrail first estimates one pinhole K (``vipe.metrictrail.calibration``): MoGe-2 predicts a
focal on each of the first 32 frames and their median starts a calibration-only DROID-style keyframe SLAM, which reads
the clip from its first frame until it keeps its keyframe budget (128 by default) and bundle-adjusts the focal jointly
with throwaway keyframe poses and disparities, pulled towards MoGe-2 metric depth at the current focal. DPVO then
tracks the clip from the start with that K.

:func:`calibration_live_view` wraps the calibration engine ``ViPESharedFocalEngine`` and its MoGe-2 wrapper
``AuditedDepthModel`` (ViPE stays unedited); :class:`CalibrationLive` logs what they produce on
:data:`entities.CALIBRATION_TIMELINE` (the index of the frame the sweep read) and on ``video_time`` / ``source_frame``:

* MoGe-2's focal on each head frame, the focal after each backend solve, the keyframes kept per frame read (scalars);
* the DROID map as ViPE's own visualiser draws it (``GraphBuffer.log``): per keyframe its pose, its frustum and the
  bundle-adjusted disparities as RGB points (:func:`keyframe_points`), plus the keyframe path and the factor-graph
  edges; keyframes the frontend drops are cleared. No per-keyframe image: a 3D view reports every image whose pinhole
  is not there at the cursor's time (keyframes not made yet, or dropped);
* the newest keyframe with a MoGe-2 prior in 2D: its RGB, that prior and its bundle-adjusted depth, in one depth range.

The map is read from the GPU at most every ``min_interval_s`` (each read waits for the GPU) and once when the sweep
ends; only keyframes the frontend or backend changed since the last read (ViPE's ``dirty`` flags) are re-logged.

The calibration turns any ``Exception`` raised inside its sweep into its MoGe-2 focal fallback, so a failure of this
view would silently change the calibrated K: every hook runs its own code through :func:`guarded`, which fails the run
with :class:`CalibrationLiveError` instead.
"""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import rerun as rr
from jaxtyping import Float32, Float64, Int64, UInt8
from numpy import ndarray

from metrictrail import entities
from metrictrail.capture import dpvo_world_T_cam
from metrictrail.palette import BA_WINDOW_RGB, KEYFRAME_RGB
from metrictrail.rerun_log import set_source_time

RunPhase = Literal["calibration", "tracking"]
"""Which stage moved the cursor: SharedK's sweep (a ``calibration_step``) or DPVO (a ``video_time``, ns)."""
KEYFRAME_PLANE_M: float = 0.05
"""Image-plane distance of each keyframe's frustum (DROID's scale, roughly metric through MoGe-2)."""
EDGE_RGB: tuple[int, int, int] = (90, 140, 200)
"""Factor-graph edges."""


class CalibrationLiveError(BaseException):
    """The live calibration view failed. Not an ``Exception``, so ViPE's calibration cannot turn it into its focal
    fallback: the run fails instead of tracking with a K the view changed."""


def guarded(action: Callable[[], None]) -> None:
    """Run the live view's ``action`` inside a calibration hook; any failure becomes :class:`CalibrationLiveError` (the
    original, a beartype violation included, is its cause)."""
    try:
        action()
    except Exception as error:
        raise CalibrationLiveError(f"the live SharedK view failed: {error!r}") from error


def keyframe_points(disps: Any, masks: Any, samples: Any, k4_internal: Any) -> tuple[Any, Any, Any]:
    """DROID keyframes as camera-space points, as ViPE's ``extract_slam_map(is_local=True)`` back-projects them.

    Args:
        disps: ``(d, h/8, w/8)`` bundle-adjusted inverse depth (torch).
        masks: ``(d, h/8, w/8)`` bool, True = invalid.
        samples: ``(d, 3, h/8, w/8)`` RGB in [0, 1]: the keyframe images at ``[..., 3::8, 3::8]``, one pixel per
            disparity, as ViPE's visualiser samples them (the caller slices before gathering the keyframes).
        k4_internal: ``fx fy cx cy`` at DROID's internal raster; divided by 8 like ``GraphBuffer.K_dense_disp``.

    Returns:
        Points ``(d, h/8, w/8, 3)`` float32, validity ``(d, h/8, w/8)`` (ViPE's single-view rule: disparity above half
        its keyframe's mean and not masked; vipe_mojo has no port of the multi-view ``depth_filter``) and colours
        ``(d, h/8, w/8, 3)`` uint8.
    """
    import torch

    fx, fy, cx, cy = (k4_internal.float() / 8.0).unbind()
    count, height8, width8 = disps.shape
    v, u = torch.meshgrid(torch.arange(height8, device=disps.device), torch.arange(width8, device=disps.device), indexing="ij")
    depth = 1.0 / disps.float().clamp_min(1e-6)
    points = torch.stack([(u - cx) / fx * depth, (v - cy) / fy * depth, depth], dim=-1).float()
    valid = (disps > 0.5 * disps.mean(dim=(1, 2), keepdim=True)) & ~masks
    colors = (samples.float().clamp(0, 1) * 255).byte().moveaxis(1, -1)
    return points, valid, colors


def depth_range_m(prior_m: Float32[ndarray, "h w"]) -> tuple[float, float]:
    """The colour range both depth panels share: the 2nd to 98th percentile of the MoGe-2 prior (0 = no prior), so the
    bundle-adjusted depth reads against it and a few far disparities cannot wash it out."""
    valid: Float32[ndarray, "k"] = prior_m[prior_m > 0]
    if valid.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(valid, [2.0, 98.0])
    return float(lo), float(max(hi, lo + 1e-3))


@dataclass(frozen=True, slots=True)
class KeyframeCloud:
    """One changed keyframe: its filtered bundle-adjusted points, in its own camera."""

    step: int
    """The sweep step it was made at (DROID's ``tstamp``)."""
    points_cam: Float32[ndarray, "m 3"]
    colors: UInt8[ndarray, "m 3"]


@dataclass(frozen=True, slots=True)
class MapSnapshot:
    """The calibration's DROID map on the host, as read after a frontend step or at the end of the sweep."""

    steps: Int64[ndarray, "n"]
    """Sweep step of every kept keyframe (DROID's ``tstamp``), in buffer order."""
    world_to_cam: Float32[ndarray, "n 7"]
    """Their lietorch SE3 world-to-camera poses."""
    edges: Int64[ndarray, "e 2"]
    """Factor-graph edges, as buffer indices."""
    k_eighth: Float64[ndarray, "3 3"]
    """The current K at 1/8 of DROID's internal raster (the resolution of its disparities)."""
    changed: list[KeyframeCloud]
    """Keyframes changed since the last snapshot."""
    newest_rgb: UInt8[ndarray, "h w 3"]
    """The newest keyframe with a MoGe-2 prior, at DROID's internal raster."""
    newest_prior_m: Float32[ndarray, "h8 w8"]
    """Its MoGe-2 metric depth at the current focal (0 = none)."""
    newest_depth_m: Float32[ndarray, "h8 w8"]
    """Its bundle-adjusted depth (0 where the points are dropped, :func:`keyframe_points`)."""


@dataclass(slots=True)
class CalibrationLive:
    """Logs the calibration into ``rec`` and reports it through the run's callbacks."""

    rec: rr.RecordingStream
    """The live recording."""
    video_time_ns: Int64[ndarray, "s"] | None
    """``video_time`` of each source frame (the input video's PTS); None for a frame directory."""
    target_keyframes: int
    """The sweep's keyframe budget (``MODES[calibration.budget]``)."""
    head_frames: int
    """Frames MoGe-2's initial focal is the median of (``calibration.initializer_frames``)."""
    on_stage: Callable[[str], None] | None
    """The run's status callback; it may raise to stop the run, so it must raise a ``BaseException`` that is not an
    ``Exception``: the calibration turns any ``Exception`` into the MoGe-2 focal fallback and carries on."""
    on_cursor: Callable[[int, RunPhase], None] | None
    """Called with the ``calibration_step`` of every frame the calibration reads."""
    min_interval_s: float = 0.5
    """Shortest wall time between two map reads."""
    step: int = -1
    """``calibration_step`` of the newest frame read."""
    source_frame: int | None = None
    """Its source frame."""
    head_focals_px: list[float] = field(default_factory=list)
    """MoGe-2's focal on each head frame so far."""
    kept: int = 0
    """Keyframes the sweep keeps now."""
    read_at: float = 0.0
    """``time.perf_counter()`` of the last map read."""
    logged: set[int] = field(default_factory=set)
    """Source frames of the keyframes in the viewer."""
    focal_px: float | None = None
    """The newest focal."""

    def __post_init__(self) -> None:
        self.rec.log(entities.CALIBRATION, rr.ViewCoordinates.RDF, static=True)
        self.rec.log(entities.CALIBRATION_FOCAL, rr.SeriesLines(names="SharedK focal (bundle adjustment)", colors=BA_WINDOW_RGB, widths=2.0), static=True)
        self.rec.log(
            entities.CALIBRATION_HEAD_FOCAL, rr.SeriesPoints(names="MoGe-2 focal per frame", colors=KEYFRAME_RGB, markers="circle", marker_sizes=3.0), static=True
        )
        self.rec.log(entities.CALIBRATION_KEPT, rr.SeriesLines(names=f"keyframes kept (budget {self.target_keyframes})", colors=KEYFRAME_RGB), static=True)

    def status(self) -> str:
        """The status line: where the calibration is and its focal against MoGe-2's."""
        if self.step < 0:
            return f"SharedK calibration: MoGe-2 focal on frame {len(self.head_focals_px)} / {self.head_frames}"
        text: str = f"SharedK calibration: {self.step + 1} frames read, {self.kept} / {self.target_keyframes} keyframes"
        if self.focal_px is not None and self.head_focals_px:  # the sweep starts from the median of the head focals
            text += f", focal {self.focal_px:.1f} px (MoGe-2: {float(np.median(self.head_focals_px)):.1f} px)"
        return text

    @contextmanager
    def at(self, step: int, source_frame: int | None) -> Iterator[None]:
        """Log on every timeline of ``step`` (``calibration_step``, and its source frame's), then reset the time."""
        self.rec.set_time(entities.CALIBRATION_TIMELINE, sequence=step)
        set_source_time(self.rec, source_frame, self.video_time_ns)
        try:
            yield
        finally:
            self.rec.reset_time()

    def head_focal(self, k4: Float64[ndarray, "4"]) -> None:
        """MoGe-2's focal on the next head frame (the head is the clip's first frames, so its index is the step)."""
        with self.at(len(self.head_focals_px), len(self.head_focals_px)):
            self.rec.log(entities.CALIBRATION_HEAD_FOCAL, rr.Scalars(float(k4[0])))
        self.head_focals_px.append(float(k4[0]))
        if self.on_stage is not None:
            self.on_stage(self.status())

    def read(self, step: int, source_frame: int) -> None:
        """The sweep read frame ``step`` (``source_frame``): move the cursor there and report (the stage callback may
        stop the run)."""
        self.step = step
        self.source_frame = source_frame
        if self.on_cursor is not None:
            self.on_cursor(step, "calibration")
        if self.on_stage is not None:
            self.on_stage(self.status())

    def frontend(self, keyframes: int) -> bool:
        """The frontend ran on the newest frame and keeps ``keyframes``; returns whether to read the map now."""
        self.kept = keyframes
        with self.at(self.step, self.source_frame):
            self.rec.log(entities.CALIBRATION_KEPT, rr.Scalars(float(keyframes)))
        return time.perf_counter() - self.read_at >= self.min_interval_s

    def focal(self, k4: Float64[ndarray, "4"]) -> None:
        """The focal (raster px) after a backend solve."""
        self.focal_px = float(k4[0])
        with self.at(max(self.step, 0), self.source_frame):
            self.rec.log(entities.CALIBRATION_FOCAL, rr.Scalars(self.focal_px))

    def map(self, snapshot: MapSnapshot, k4_raster: Float64[ndarray, "4"]) -> None:
        """Log a map read: the changed keyframes' poses, frustums and points, the path, the
        edges and the newest keyframe; clear the keyframes the frontend dropped."""
        self.read_at = time.perf_counter()
        self.kept = int(snapshot.steps.shape[0])
        self.focal_px = float(k4_raster[0])
        with self.at(max(self.step, 0), self.source_frame):
            frames: list[int] = [int(frame) for frame in snapshot.steps]
            for dropped in sorted(self.logged - set(frames)):
                self.rec.log(entities.calibration_keyframe(dropped), rr.Clear(recursive=True))
            self.logged = set(frames)
            if frames:
                world_T_cam: Float32[ndarray, "n 4 4"] = dpvo_world_T_cam(snapshot.world_to_cam)
                centres: Float32[ndarray, "n 3"] = world_T_cam[:, :3, 3]
                height8, width8 = snapshot.newest_prior_m.shape
                index_of: dict[int, int] = {frame: index for index, frame in enumerate(frames)}
                for cloud in snapshot.changed:  # ViPE's dirty keyframes: the ones whose pose, disparities or focal moved
                    index: int = index_of[cloud.step]
                    entity: str = entities.calibration_keyframe(cloud.step)
                    self.rec.log(entity, rr.Transform3D(translation=centres[index], mat3x3=world_T_cam[index, :3, :3]))
                    self.rec.log(
                        f"{entity}/camera",
                        rr.Pinhole(image_from_camera=snapshot.k_eighth, resolution=[width8, height8], image_plane_distance=KEYFRAME_PLANE_M, color=KEYFRAME_RGB),
                    )
                    self.rec.log(f"{entity}/points", rr.Points3D(cloud.points_cam, colors=cloud.colors, radii=-2.0))
                self.rec.log(entities.CALIBRATION_PATH, rr.LineStrips3D([centres], colors=BA_WINDOW_RGB, radii=-1.5))
                self.rec.log(entities.CALIBRATION_EDGES, rr.LineStrips3D(centres[snapshot.edges], colors=EDGE_RGB, radii=-0.5))
                lo, hi = depth_range_m(snapshot.newest_prior_m)
                self.rec.log(entities.CALIBRATION_NEWEST_RGB, rr.Image(snapshot.newest_rgb).compress(jpeg_quality=85))
                self.rec.log(entities.CALIBRATION_NEWEST_PRIOR, rr.DepthImage(snapshot.newest_prior_m, meter=1.0, colormap="turbo", depth_range=[lo, hi]))
                self.rec.log(entities.CALIBRATION_NEWEST_DEPTH, rr.DepthImage(snapshot.newest_depth_m, meter=1.0, colormap="turbo", depth_range=[lo, hi]))


def read_map(engine: Any) -> MapSnapshot:
    """Read the calibration engine's DROID map to the host (waits for the GPU); clears ViPE's ``dirty`` flags of what it
    read, as ``GraphBuffer.log`` does."""
    import torch

    buffer: Any = engine.system.buffer
    count: int = int(buffer.n_frames)
    with torch.no_grad():
        dirty: Any = torch.where(buffer.dirty[:count])[0]
        buffer.dirty[:count] = False
        steps: Int64[ndarray, "n"] = buffer.tstamp[:count].long().cpu().numpy()
        world_to_cam: Float32[ndarray, "n 7"] = buffer.poses[:count].float().cpu().numpy()
        graph: Any = engine.system.frontend.graph
        edges: Int64[ndarray, "e 2"] = torch.stack([graph.ii, graph.jj], dim=-1).long().cpu().numpy().reshape(-1, 2)
        edges = edges[(edges < count).all(axis=1) & (edges[:, 0] < edges[:, 1])]
        changed: list[KeyframeCloud] = []
        if dirty.numel():
            samples: Any = buffer.images[:count, 0, :, 3::8, 3::8][dirty]  # gather the sampled pixels, not every full raster
            points, valid, colors = keyframe_points(buffer.disps[dirty, 0], buffer.masks[dirty, 0], samples, buffer.intrinsics[0])
            # Every second row and column: a quarter of the points, still a dense map, a quarter of the bytes per read.
            points_host: Float32[ndarray, "d h16 w16 3"] = points[:, ::2, ::2].cpu().numpy()
            valid_host = valid[:, ::2, ::2].cpu().numpy()
            colors_host: UInt8[ndarray, "d h16 w16 3"] = colors[:, ::2, ::2].cpu().numpy()
            for index, slot in enumerate(dirty.tolist()):
                keep = valid_host[index]
                changed.append(KeyframeCloud(int(steps[slot]), points_host[index][keep], colors_host[index][keep]))
        k_eighth: Float64[ndarray, "3 3"] = np.asarray(buffer.K_dense_disp[0], dtype=np.float64)
        # The newest keyframe that has its MoGe-2 prior: depth is attached once the frontend keeps a keyframe, so the
        # newest one usually has none yet.
        with_prior = torch.nonzero(buffer.disps_sens[:count, 0].amax(dim=(1, 2)) > 0).flatten()
        shown: int = int(with_prior[-1]) if with_prior.numel() else count - 1
        _, shown_valid, _ = keyframe_points(
            buffer.disps[shown : shown + 1, 0], buffer.masks[shown : shown + 1, 0], buffer.images[shown : shown + 1, 0, :, 3::8, 3::8], buffer.intrinsics[0]
        )
        rgb_full: UInt8[ndarray, "h w 3"] = (buffer.images[shown, 0].float().clamp(0, 1) * 255).byte().moveaxis(0, -1).cpu().numpy()
        prior: Float32[ndarray, "h8 w8"] = torch.where(buffer.disps_sens[shown, 0] > 0, 1.0 / buffer.disps_sens[shown, 0], 0.0).float().cpu().numpy()
        depth: Float32[ndarray, "h8 w8"] = torch.where(shown_valid[0], 1.0 / buffer.disps[shown, 0].clamp_min(1e-6), 0.0).float().cpu().numpy()
    return MapSnapshot(
        steps=steps,
        world_to_cam=world_to_cam,
        edges=edges,
        k_eighth=k_eighth,
        changed=changed,
        newest_rgb=rgb_full,
        newest_prior_m=prior,
        newest_depth_m=depth,
    )


@contextmanager
def calibration_live_view(live: CalibrationLive | None) -> Iterator[None]:
    """Wrap ``AuditedDepthModel.estimate_free_fov_k4`` and ``ViPESharedFocalEngine.candidate`` / ``run_frontend`` /
    ``run_backend`` / ``end_benchmark`` to report to ``live`` while the context is open; None: nothing to wrap."""
    if live is None:
        yield
        return
    view: CalibrationLive = live  # the hooks below close over it (a closure loses the narrowing)
    from vipe.metrictrail.calibration.engine import AuditedDepthModel, ViPESharedFocalEngine

    head_orig = AuditedDepthModel.estimate_free_fov_k4
    candidate_orig = ViPESharedFocalEngine.candidate
    frontend_orig = ViPESharedFocalEngine.run_frontend
    backend_orig = ViPESharedFocalEngine.run_backend
    end_orig = ViPESharedFocalEngine.end_benchmark

    def log_map(engine: Any) -> None:
        if int(engine.system.buffer.n_frames) > 0:
            view.map(read_map(engine), engine.recovered_k().astype(np.float64))

    def estimate_free_fov_k4(self: Any, rgb_bchw: Any) -> Any:
        k4: Any = head_orig(self, rgb_bchw)
        guarded(lambda: view.head_focal(np.asarray(k4, dtype=np.float64)))
        return k4

    def candidate(self: Any, frame_index: int, frame: Any) -> bool:
        guarded(lambda: view.read(int(frame_index), int(frame.raw_frame_idx)))  # a Stop passes through: not an Exception
        return candidate_orig(self, frame_index, frame)

    def run_frontend(self: Any) -> None:
        frontend_orig(self)

        def report() -> None:
            if view.frontend(int(self.system.buffer.n_frames)):
                log_map(self)

        guarded(report)

    def run_backend(self: Any, steps: int, *, update_depth: bool) -> None:
        backend_orig(self, steps, update_depth=update_depth)
        guarded(lambda: view.focal(self.recovered_k().astype(np.float64)))

    def end_benchmark(self: Any) -> dict[str, float]:
        timings: dict[str, float] = end_orig(self)
        guarded(lambda: log_map(self))  # after the benchmark interval, so the calibration's own timing leaves this read out
        return timings

    patches: list[tuple[type, str, Callable[..., Any], Callable[..., Any]]] = [
        (AuditedDepthModel, "estimate_free_fov_k4", head_orig, estimate_free_fov_k4),
        (ViPESharedFocalEngine, "candidate", candidate_orig, candidate),
        (ViPESharedFocalEngine, "run_frontend", frontend_orig, run_frontend),
        (ViPESharedFocalEngine, "run_backend", backend_orig, run_backend),
        (ViPESharedFocalEngine, "end_benchmark", end_orig, end_benchmark),
    ]
    for owner, name, _, wrapper in patches:
        setattr(owner, name, wrapper)
    try:
        yield
    finally:
        for owner, name, original, _ in patches:
            setattr(owner, name, original)
