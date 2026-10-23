"""mast3r-slam Types: the typed SLAM config (``SlamConfig``), the working ``Frame`` and the values on the graph's edges.

Edge Types (host, picklable: the backend runs in its own process):

- ``TrackedFrame``: one camera pose the SLAM estimated (frontend ``tracked``: per input frame left in TRACKING; backend
  ``keyframes``: the final keyframe poses, emitted at flush at each keyframe's own frame time).
- ``KeyframeAdded`` / ``RelocRequest``: the frontend's tasks for the backend (``BackendTask``).
- ``MapUpdate``: every keyframe pose and factor-graph edge after the backend applied a task.
- ``LogLine``: one line of the run's text log.
- ``FrameView`` / ``KeyframeView``: what the viewer draws of the current frame and of a new keyframe (``CameraView``: pose,
  image, point map, confidence; viewer only, never resumed).

Poses are lietorch Sim3 rows ``[t_xyz, q_xyzw, s]`` (world_from_cam); Rerun gets the rigid part (Transform3D) and keeps
the scale and indices as ``AnyValues`` beside it, so ``from_layer`` restores the value exactly.
"""

import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, TypeAlias

import lietorch
import numpy as np
import pyarrow as pa
import rerun as rr
import torch
from dagkit.time import TimeNs
from dagkit.types import Log, column, row_times
from dust3r.post_process import estimate_focal_knowing_depth
from jaxtyping import Bool, Float, Float32, Int, Int64, UInt8
from numpy import ndarray
from serde import coerce, serde
from torch import Tensor

FilteringMode: TypeAlias = Literal["first", "recent", "best_score", "indep_conf", "weighted_pointmap", "weighted_spherical"]
"""How a frame's canonical point map absorbs a new observation (see ``ops.pointmap.update_pointmap``)."""


class Mode(Enum):
    """Operating mode of the SLAM system (shared by the frontend and the backend)."""

    INIT = 0
    TRACKING = 1
    RELOC = 2
    TERMINATED = 3


@serde(deny_unknown_fields=True, type_check=coerce)
@dataclass(frozen=True, slots=True)
class MatchingConfig:
    """Iterative projection matching (``ops.matching``)."""

    max_iter: int = 10
    """Levenberg-Marquardt iterations of the projection search."""
    lambda_init: float = 1e-8
    """Initial damping of the projection search."""
    convergence_thresh: float = 1e-6
    """Step norm below which the projection search stops."""
    dist_thresh: float = 0.1
    """3D distance (canonical units) above which a match counts as occluded."""
    radius: int = 3
    """Descriptor refinement radius in pixels (0 disables the refinement)."""
    dilation_max: int = 5
    """Largest refinement dilation; it decreases to 1."""


@serde(deny_unknown_fields=True, type_check=coerce)
@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """Frame-to-keyframe pose tracking (frontend)."""

    min_match_frac: float = 0.05
    """Fraction of valid matches below which tracking is lost (relocalisation)."""
    max_iters: int = 50
    """Gauss-Newton iterations of the pose solve."""
    C_conf: float = 0.0
    """Point confidence threshold."""
    Q_conf: float = 1.5
    """Match confidence threshold."""
    rel_error: float = 1e-3
    """Relative cost decrease that counts as converged."""
    delta_norm: float = 1e-3
    """Step norm that counts as converged."""
    huber: float = 1.345
    """Huber threshold of the robust cost."""
    match_frac_thresh: float = 0.333
    """Overlap below which the frame becomes a new keyframe."""
    sigma_ray: float = 0.003
    """Ray residual sigma (uncalibrated)."""
    sigma_dist: float = 10.0
    """Distance residual sigma (uncalibrated)."""
    sigma_pixel: float = 1.0
    """Pixel residual sigma (calibrated)."""
    sigma_depth: float = 10.0
    """Log-depth residual sigma (calibrated)."""
    pixel_border: int = -10
    """Calibrated: pixels may project this far outside the image (negative = outside allowed)."""
    depth_eps: float = 1e-6
    """Calibrated: smallest valid depth."""
    filtering_mode: FilteringMode = "weighted_pointmap"
    """How point maps absorb new observations."""
    filtering_score: Literal["median", "mean"] = "median"
    """Score of ``best_score`` filtering."""


@serde(deny_unknown_fields=True, type_check=coerce)
@dataclass(frozen=True, slots=True)
class LocalOptConfig:
    """Global factor-graph optimisation (backend)."""

    pin: int = 1
    """Keyframes held fixed (the first ones)."""
    C_conf: float = 0.0
    """Point confidence threshold."""
    Q_conf: float = 1.5
    """Match confidence threshold."""
    min_match_frac: float = 0.1
    """Fraction of valid matches an edge needs (both directions)."""
    pixel_border: int = -10
    """Calibrated: pixels may project this far outside the image."""
    depth_eps: float = 1e-6
    """Calibrated: smallest valid depth."""
    max_iters: int = 10
    """Gauss-Newton iterations per solve."""
    sigma_ray: float = 0.003
    """Ray residual sigma (uncalibrated)."""
    sigma_dist: float = 10.0
    """Distance residual sigma (uncalibrated)."""
    sigma_pixel: float = 1.0
    """Pixel residual sigma (calibrated)."""
    sigma_depth: float = 10.0
    """Log-depth residual sigma (calibrated)."""
    delta_norm: float = 1e-8
    """Step norm that stops the solve."""


@serde(deny_unknown_fields=True, type_check=coerce)
@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Image retrieval of loop-closure candidates."""

    k: int = 3
    """Candidates per query."""
    min_thresh: float = 5e-3
    """Smallest retrieval score of a candidate."""


@serde(deny_unknown_fields=True, type_check=coerce)
@dataclass(frozen=True, slots=True)
class RelocConfig:
    """Relocalisation after tracking is lost."""

    min_match_frac: float = 0.3
    """Fraction of valid matches a relocalisation edge needs."""
    strict: bool = True
    """Fail the relocalisation when any candidate edge is invalid."""


@serde(deny_unknown_fields=True, type_check=coerce)
@dataclass(frozen=True, slots=True)
class SlamConfig:
    """Everything the frontend and the backend read (profile presets override fields; TOML is coerced)."""

    use_calib: bool = False
    """Calibrated tracking and optimisation (needs the source's rig intrinsics and the CUDA backend)."""
    img_size: Literal[224, 512] = 512
    """MASt3R input size."""
    subsample: int = 1
    """Input stride: every n-th frame of the source."""
    keyframe_buffer: int = 512
    """Keyframe slots in the shared buffer; a run that needs more fails with a clear error."""
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    local_opt: LocalOptConfig = field(default_factory=LocalOptConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reloc: RelocConfig = field(default_factory=RelocConfig)


@dataclasses.dataclass
class Frame:
    """One image with its canonical point map, confidence, MASt3R features and world_from_cam Sim3 pose (working state)."""

    frame_id: int
    """Index of this frame in the input sequence."""
    rgb_tensor: Float[Tensor, "*batch 3 h w"]
    """ImageNet-normalized RGB tensor, (1, 3, h, w) or (3, h, w) when read from the shared buffer."""
    img_shape: Int[Tensor, "*batch 2"]
    """(height, width) of the processed image after optional downsampling."""
    img_true_shape: Int[Tensor, "*batch 2"]
    """(height, width) of the image before any downsampling."""
    rgb: Float[Tensor, "*batch h w 3"]
    """RGB image in [0, 1], HWC on the CPU."""
    world_sim3_cam: lietorch.Sim3 = lietorch.Sim3.Identity(1)
    """World-from-camera Sim3 pose."""
    X_canon: Float[Tensor, "hw 3"] | None = None
    """Canonical 3D point map."""
    C: Float[Tensor, "hw 1"] | None = None
    """Per-point confidence."""
    feat: Float[Tensor, "1 n_patches feat_dim"] | None = None
    """Encoded MASt3R feature tokens."""
    pos: Int[Tensor, "1 n_patches 2"] | None = None
    """Positional encodings of the feature patches."""
    N: int = 0
    """Accumulated point map observations."""
    N_updates: int = 0
    """Point map update calls."""
    K: Float[Tensor, "3 3"] | None = None
    """Camera intrinsics (calibrated mode only)."""
    score: Float[Tensor, ""] | None = None
    """Scalar filtering score (``best_score`` filtering)."""

    def get_average_conf(self) -> Float[Tensor, "hw 1"] | None:
        """Confidence divided by the observation count, or None without confidence."""
        return self.C / self.N if self.C is not None else None


def _value(cell: object) -> float:
    return float(np.asarray(cell, dtype=np.float64).reshape(-1)[0])


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class TrackedFrame:
    """A camera pose the SLAM estimated for one input frame."""

    frame_index: int
    """Input frame index (the source's own frame numbering)."""
    world_sim3_cam: Float32[ndarray, " 8"]
    """World-from-camera Sim3 ``[t_xyz, q_xyzw, s]``."""

    def to_rerun(self, root: str) -> list[Log]:
        pose: Float32[ndarray, " 8"] = self.world_sim3_cam
        values = rr.AnyValues(frame_index=self.frame_index, sim3_scale=float(pose[7]))
        return [Log(root, rr.Transform3D(translation=pose[:3], quaternion=rr.Quaternion(xyzw=pose[3:7]))), Log(root, values)]

    @classmethod
    def contents(cls, root: str) -> list[str]:
        return [root]

    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "TrackedFrame"]]:
        cells = zip(*(column(table, f"{root}:{c}") for c in ("Transform3D:translation", "Transform3D:quaternion", "sim3_scale", "frame_index")), strict=True)
        for t_ns, (xyz, xyzw, scale, frame_index) in zip(row_times(table, timeline), cells, strict=True):
            if xyz is not None and frame_index is not None:
                pose: Float32[ndarray, " 8"] = np.concatenate([np.ravel(xyz), np.ravel(xyzw), [_value(scale)]]).astype(np.float32)
                yield t_ns, cls(int(_value(frame_index)), pose)


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class KeyframeAdded:
    """The frontend appended keyframe ``index`` to the shared buffer: connect it and optimise the map."""

    index: int


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class RelocRequest:
    """Tracking is lost: relocalise the shared current frame against the retrieval database."""

    frame_index: int


BackendTask: TypeAlias = KeyframeAdded | RelocRequest


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class MapUpdate:
    """The map after the backend applied one task: every keyframe pose and every factor-graph edge ``ii[e] - jj[e]``."""

    world_sim3_cam: Float32[ndarray, "k 8"]
    """Keyframe ``i``'s world-from-camera Sim3 in row ``i``."""
    ii: Int64[ndarray, " e"]
    """First keyframe of each matching factor."""
    jj: Int64[ndarray, " e"]
    """Second keyframe of each matching factor."""

    def to_rerun(self, root: str) -> list[Log]:
        """Keyframe centres (``<root>/keyframes``) and factor edges (``<root>/edges``); the exact values ride along as AnyValues."""
        centres: Float32[ndarray, "k 3"] = self.world_sim3_cam[:, :3]
        keyframes = rr.AnyValues(world_sim3_cam=self.world_sim3_cam.reshape(-1))
        lines = rr.LineStrips3D(np.stack([centres[self.ii], centres[self.jj]], axis=1), colors=(0, 200, 0), radii=rr.Radius.ui_points(1.0))
        logs: list[Log] = [Log(f"{root}/keyframes", rr.Points3D(centres, colors=(255, 140, 0), radii=rr.Radius.ui_points(4.0))), Log(f"{root}/keyframes", keyframes)]
        if len(self.ii):  # Rerun refuses an AnyValues of empty batches; edges never disappear once logged
            logs += [Log(f"{root}/edges", lines), Log(f"{root}/edges", rr.AnyValues(edge_ii=self.ii, edge_jj=self.jj))]
        return logs

    @classmethod
    def contents(cls, root: str) -> list[str]:
        return [f"{root}/keyframes", f"{root}/edges"]

    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "MapUpdate"]]:
        columns = (f"{root}/keyframes:world_sim3_cam", f"{root}/edges:edge_ii", f"{root}/edges:edge_jj")
        for t_ns, (poses, ii, jj) in zip(row_times(table, timeline), zip(*(column(table, c) for c in columns), strict=True), strict=True):
            if poses is not None:
                first, second = (np.asarray(e if e is not None else [], dtype=np.int64).reshape(-1) for e in (ii, jj))
                yield t_ns, cls(np.asarray(poses, dtype=np.float32).reshape(-1, 8), first, second)


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class LogLine:
    """One line of the run's text log (the backend's retrieval and relocalisation verdicts)."""

    text: str
    level: str = "INFO"

    def to_rerun(self, root: str) -> list[Log]:
        return [Log(root, rr.TextLog(self.text, level=self.level))]

    @classmethod
    def contents(cls, root: str) -> list[str]:
        return [root]

    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "LogLine"]]:
        for t_ns, text, level in zip(row_times(table, timeline), column(table, f"{root}:TextLog:text"), column(table, f"{root}:TextLog:level"), strict=True):
            if text is not None:
                yield t_ns, cls(str(text[0]), str(level[0]) if level is not None else "INFO")


CONFIDENT: float = 7.0
"""Accumulated confidence a point needs to enter a keyframe's cloud and the drawn depth (the CLI's threshold)."""


def _gray(values: Float32[ndarray, "..."]) -> UInt8[ndarray, "..."]:
    """Min-max over every finite value to 0-255 (a point map's xyz as RGB, a confidence as grey)."""
    finite: Float32[ndarray, "..."] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = float(finite.min()), float(finite.max())
    return np.zeros(finite.shape, np.uint8) if high <= low else np.clip((finite - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)


class Drawing:
    """Base of the viewer's values: drawn, never resumed (a run re-derives them from the SLAM state)."""

    __slots__ = ()

    @classmethod
    def contents(cls, root: str) -> list[str]:
        raise NotImplementedError(f"{cls.__name__} is drawn, not resumed")

    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "Drawing"]]:
        raise NotImplementedError(f"{cls.__name__} is drawn, not resumed")


@dataclass(frozen=True, slots=True)
class CameraView:
    """One camera as the viewer draws it (host copies): its pose, image, canonical point map and the map's confidence."""

    index: int
    """Input frame index (the current frame) or keyframe index."""
    world_sim3_cam: Float32[ndarray, " 8"]
    rgb: UInt8[ndarray, "h w 3"]
    points: Float32[ndarray, "h w 3"]
    """Canonical points in the camera frame."""
    confidence: Float32[ndarray, "h w"]

    @classmethod
    def of(cls, frame: "Frame", index: int) -> "CameraView":
        """Host copies (owned: a keyframe's buffers change after the copy) of a frame with a point map, the RGB as the CLI
        quantised it."""
        assert frame.X_canon is not None and frame.C is not None
        h, w = frame.rgb.shape[-3:-1]
        pose: Float32[ndarray, " 8"] = frame.world_sim3_cam.data.reshape(-1).to("cpu", copy=True).numpy()
        rgb: UInt8[ndarray, "h w 3"] = (frame.rgb.reshape(h, w, 3).numpy() * 255).astype(np.uint8)
        points: Float32[ndarray, "h w 3"] = frame.X_canon.reshape(h, w, 3).to("cpu", copy=True).numpy()
        return cls(index, pose, rgb, points, frame.C.reshape(h, w).to("cpu", copy=True).numpy())

    def frustum(self, path: str, image_plane: float) -> list[Log]:
        """The rigid pose at ``path`` (the viewer draws no frustum under a scaled transform) and its frustum, with the focal
        length the point map implies and the principal point at the image centre, as the CLI estimated it."""
        h, w = self.rgb.shape[:2]
        focal = float(estimate_focal_knowing_depth(torch.from_numpy(self.points)[None], torch.tensor([w / 2, h / 2]), focal_mode="weiszfeld"))
        pose: Float32[ndarray, " 8"] = self.world_sim3_cam
        pinhole = rr.Pinhole(focal_length=focal, principal_point=(w / 2, h / 2), width=w, height=h, camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=image_plane)
        return [Log(path, rr.Transform3D(translation=pose[:3], quaternion=rr.Quaternion(xyzw=pose[3:7]))), Log(f"{path}/pinhole", pinhole)]

    def images(self, image: str, maps: str) -> list[Log]:
        """The image (JPEG) at ``image``; at ``maps``/ the point map as colour, its confident depth in mm and the confidence."""
        points: Float32[ndarray, "h w 3"] = np.nan_to_num(self.points, nan=0.0, posinf=0.0, neginf=0.0)
        depth_m: Float32[ndarray, "h w"] = np.where(self.confidence > CONFIDENT, np.maximum(points[..., 2], 0.0), 0.0)
        return [
            Log(image, rr.Image(self.rgb, color_model=rr.ColorModel.RGB).compress(jpeg_quality=75)),
            Log(f"{maps}/pointmap", rr.Image(_gray(points), color_model=rr.ColorModel.RGB).compress()),
            Log(f"{maps}/pointmap_depth", rr.DepthImage(np.clip(depth_m * 1000.0, 0.0, 65535.0).astype(np.uint16), meter=1000.0).compress()),
            Log(f"{maps}/confidence", rr.Image(_gray(self.confidence), color_model=rr.ColorModel.L).compress()),
        ]


@dataclass(frozen=True, slots=True)
class FrameView(Drawing):
    """The current frame and the last keyframe, whose point map the tracker refines with every frame (viewer only)."""

    current: CameraView
    last_keyframe: CameraView | None

    def to_rerun(self, root: str) -> list[Log]:
        """``current_camera``: pose, frustum (image plane 0.4), image and maps; ``last_keyframe``: image and maps."""
        camera: str = f"{root}/current_camera"
        logs: list[Log] = [*self.current.frustum(camera, 0.4), *self.current.images(f"{camera}/pinhole/image", f"{camera}/pinhole")]
        return logs + (self.last_keyframe.images(f"{root}/last_keyframe", f"{root}/last_keyframe") if self.last_keyframe is not None else [])


@dataclass(frozen=True, slots=True)
class KeyframeView(Drawing):
    """A new keyframe (viewer only): its frustum with its image and maps, and its confident points, coloured."""

    keyframe: CameraView

    def to_rerun(self, root: str) -> list[Log]:
        """``keyframe-<index>``: pose, frustum (image plane 0.2), image, maps and ``pointcloud`` at the keyframe's Sim3 scale
        (the pose above it is rigid)."""
        kf: CameraView = self.keyframe
        path: str = f"{root}/keyframe-{kf.index}"
        keep: Bool[ndarray, "h w"] = kf.confidence > CONFIDENT
        cloud = rr.Points3D(kf.points[keep], colors=kf.rgb[keep])
        return [*kf.frustum(path, 0.2), *kf.images(f"{path}/pinhole/image", f"{path}/pinhole"), Log(f"{path}/pointcloud", rr.Transform3D(scale=float(kf.world_sim3_cam[7]))), Log(f"{path}/pointcloud", cloud)]
