"""mast3r-slam Types: the typed SLAM config (``SlamConfig``), the working ``Frame`` and the values on the graph's edges.

Edge Types (host, picklable: the backend runs in its own process):

- ``TrackedFrame``: one camera pose the SLAM estimated (frontend ``tracked``: per input frame left in TRACKING; backend
  ``keyframes``: the final keyframe poses, emitted at flush at each keyframe's own frame time).
- ``KeyframeAdded`` / ``RelocRequest``: the frontend's tasks for the backend (``BackendTask``).
- ``MapUpdate``: every keyframe pose and factor-graph edge after the backend applied a task.
- ``KeyframeCloud``: a new keyframe's confident points in its own camera frame (viewer only).

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
from jaxtyping import Bool, Float, Float32, Int, Int64, UInt8
from numpy import ndarray
from serde import coerce, serde
from torch import Tensor

FilteringMode: TypeAlias = Literal["first", "recent", "best_score", "indep_conf", "weighted_pointmap", "weighted_spherical"]
"""How a frame's canonical point map absorbs a new observation (see ``Frame.update_pointmap``)."""


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


def _cartesian_to_spherical(P: Float[Tensor, "hw 3"]) -> Float[Tensor, "hw 3"]:
    r: Float[Tensor, "hw 1"] = torch.linalg.norm(P, dim=-1, keepdim=True)
    x, y, z = torch.tensor_split(P, 3, dim=-1)
    return torch.cat((r, torch.atan2(y, x), torch.acos(z / r)), dim=-1)


def _spherical_to_cartesian(spherical: Float[Tensor, "hw 3"]) -> Float[Tensor, "hw 3"]:
    r, phi, theta = torch.tensor_split(spherical, 3, dim=-1)
    return torch.cat((r * torch.sin(theta) * torch.cos(phi), r * torch.sin(theta) * torch.sin(phi), r * torch.cos(theta)), dim=-1)


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

    def update_pointmap(self, X: Float[Tensor, "hw 3"], C: Float[Tensor, "hw 1"], cfg: TrackingConfig) -> None:
        """Absorb a new observation into the canonical point map with ``cfg.filtering_mode``."""
        mode: FilteringMode = cfg.filtering_mode
        score = torch.median if cfg.filtering_score == "median" else torch.mean
        if self.N == 0:  # the first observation is always accepted
            self.X_canon, self.C, self.N, self.N_updates = X.clone(), C.clone(), 1, 1
            if mode == "best_score":
                self.score = score(C)
            return
        assert self.C is not None and self.X_canon is not None
        if mode == "first" and self.N_updates == 1 or mode == "recent":  # first: only the second observation
            self.X_canon, self.C, self.N = X.clone(), C.clone(), 1
        elif mode == "best_score":
            new_score: Float[Tensor, ""] = score(C)
            assert self.score is not None
            if new_score > self.score:
                self.X_canon, self.C, self.N, self.score = X.clone(), C.clone(), 1, new_score
        elif mode == "indep_conf":  # per pixel: keep the more confident point
            new_mask: Bool[Tensor, "hw 1"] = C > self.C
            self.X_canon[new_mask.repeat(1, 3)] = X[new_mask.repeat(1, 3)]
            self.C[new_mask] = C[new_mask]
            self.N = 1
        elif mode == "weighted_pointmap":  # confidence-weighted running mean in Cartesian space
            self.X_canon = ((self.C * self.X_canon) + (C * X)) / (self.C + C)
            self.C, self.N = self.C + C, self.N + 1
        elif mode == "weighted_spherical":  # the same in spherical (r, phi, theta) space
            spherical = ((self.C * _cartesian_to_spherical(self.X_canon)) + (C * _cartesian_to_spherical(X))) / (self.C + C)
            self.X_canon = _spherical_to_cartesian(spherical)
            self.C, self.N = self.C + C, self.N + 1
        self.N_updates += 1

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
        for i, pose in enumerate(self.world_sim3_cam):  # each keyframe's frame, so its KeyframeCloud follows the corrected pose
            logs.append(Log(f"{root}/keyframes/{i:04d}", rr.Transform3D(translation=pose[:3], quaternion=rr.Quaternion(xyzw=pose[3:7]), scale=float(pose[7]))))
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
class KeyframeCloud:
    """The confident points of a new keyframe in its own camera frame, coloured (logged under the keyframe's pose)."""

    index: int
    """Keyframe index (``MapUpdate`` row)."""
    points: Float32[ndarray, "n 3"]
    """Canonical points in the keyframe's camera frame (at its Sim3 scale)."""
    colors: UInt8[ndarray, "n 3"]
    """RGB of each point."""

    def to_rerun(self, root: str) -> list[Log]:
        return [Log(f"{root}/{self.index:04d}/points", rr.Points3D(self.points, colors=self.colors, radii=rr.Radius.ui_points(1.5)))]

    @classmethod
    def contents(cls, root: str) -> list[str]:
        return [f"{root}/**"]

    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "KeyframeCloud"]]:
        names: list[str] = [c.removesuffix(":Points3D:positions") for c in table.column_names if c.startswith(f"{root}/") and c.endswith("/points:Points3D:positions")]
        for row, t_ns in enumerate(row_times(table, timeline)):
            for name in names:
                points = table[f"{name}:Points3D:positions"][row].as_py()
                if points is not None:
                    colors = np.asarray(table[f"{name}:Points3D:colors"][row].as_py(), dtype=np.uint32)  # packed RGBA
                    rgb: UInt8[ndarray, "n 3"] = np.stack([(colors >> s) & 0xFF for s in (24, 16, 8)], axis=1).astype(np.uint8)
                    index: int = int(name.removeprefix(f"{root}/").removesuffix("/points"))
                    yield t_ns, cls(index, np.asarray(points, dtype=np.float32).reshape(-1, 3), rgb)
