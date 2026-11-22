"""`TrigsMotion3D`: a TRiGS motion model logged once beside a static `GaussianSplats3D` (compact 4D playback).

The custom viewer evaluates the motion at the time cursor (`gsplat_core::trigs`, tested against the trainer's
`_deform`). Stock viewers draw the static canonical rows only: every pose of the clip at once (a ghost trail).

Contract (one entity, every component static):
  GaussianSplats3D:centers/quaternions(xyzw)/scales(linear)/colors  canonical rows; colour RGB = clip(0.5 + SH_C0 * sh0),
                                                                    A = sigmoid(opacity logit)
  TrigsMotion3D:opacities          f32    sigmoid(opacity logit), unquantised
  TrigsMotion3D:times              f32    centre time mu, seconds
  TrigsMotion3D:log_durations      f32    log sigma, seconds
  TrigsMotion3D:twist_base/c0/c1/c2 f32x6 (nu xyz, omega xyz): base twist and quadratic Bezier control points
  TrigsMotion3D:anchors            f32x3  rotation anchor
  TrigsMotion3D:is_static          f32    1 = static row (no motion, gamma = 1), else 0
  TrigsMotion3D:fps                f64    one value: timeline frames per second
  TrigsMotion3D:frame_origin       f64    one value: timeline frame at t = 0 s
  TrigsMotion3D:active_threshold   f32    one value: smooth temporal cull threshold (0 = off)
Time: t [s] = (frame_index - frame_origin) / fps on a sequence timeline.
The motion fields have no Rerun component type (like the untyped `Hexels3D` fields).
"""

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Float32, UInt8
from numpy import ndarray

from gsplat_rust_renderer.gaussians3d import SH_C0

ARCHETYPE: str = "TrigsMotion3D"


def descriptor(field: str) -> rr.ComponentDescriptor:
    """`TrigsMotion3D:<field>` without a component type."""
    return rr.ComponentDescriptor(f"{ARCHETYPE}:{field}", archetype=ARCHETYPE, component_type=None)


def untyped(field: str, values: ndarray, dtype: pa.DataType) -> rr.DescribedComponentBatch:
    """Rows (scalars or fixed-size lists) of `dtype` under an untyped `TrigsMotion3D` descriptor."""
    flat: pa.Array = pa.array(np.ascontiguousarray(values).reshape(-1), type=dtype)
    array: pa.Array = flat if values.ndim == 1 else pa.FixedSizeListArray.from_arrays(flat, values.shape[1])
    field_descriptor: rr.ComponentDescriptor = descriptor(field)
    return rr.DescribedComponentBatch(rr.AnyBatchValue(field_descriptor, array), field_descriptor)


@dataclass(frozen=True)
class TrigsMotion3D(rr.AsComponents):
    """Per-row TRiGS motion (activated where the viewer needs it) and the clip's time mapping."""

    opacities: Float32[ndarray, "n"]
    """Activated opacity sigmoid(logit) in [0, 1]."""
    times: Float32[ndarray, "n"]
    """Centre time of each row, seconds."""
    log_durations: Float32[ndarray, "n"]
    """Log temporal scale, seconds."""
    twist_base: Float32[ndarray, "n 6"]
    """Base twist (nu xyz, omega xyz)."""
    twist_c0: Float32[ndarray, "n 6"]
    """First quadratic Bezier control point of the twist."""
    twist_c1: Float32[ndarray, "n 6"]
    """Second control point."""
    twist_c2: Float32[ndarray, "n 6"]
    """Third control point."""
    anchors: Float32[ndarray, "n 3"]
    """Rotation anchor."""
    is_static: Float32[ndarray, "n"]
    """1 for a static row, 0 otherwise."""
    fps: float
    """Timeline frames per second."""
    frame_origin: float
    """Timeline frame at t = 0 s."""
    active_threshold: float
    """Smooth temporal cull threshold; 0 turns the cull off."""

    def __post_init__(self) -> None:
        rows: int = len(self.opacities)
        lengths: set[int] = {len(v) for v in (self.times, self.log_durations, self.twist_base, self.twist_c0, self.twist_c1, self.twist_c2, self.anchors, self.is_static)}
        if lengths != {rows}:
            raise ValueError(f"TrigsMotion3D: every per-row field needs {rows} rows, got lengths {sorted(lengths)}")
        if not self.fps > 0.0:
            raise ValueError(f"TrigsMotion3D: fps must be positive, got {self.fps}")
        if not np.isin(self.is_static, (0.0, 1.0)).all():
            raise ValueError("TrigsMotion3D: is_static holds only 0 and 1")

    def as_component_batches(self) -> list[rr.DescribedComponentBatch]:
        """Untyped batches in contract order; the three clip values are one-element batches."""
        f32: pa.DataType = pa.float32()
        return [
            untyped("opacities", self.opacities, f32),
            untyped("times", self.times, f32),
            untyped("log_durations", self.log_durations, f32),
            untyped("twist_base", self.twist_base, f32),
            untyped("twist_c0", self.twist_c0, f32),
            untyped("twist_c1", self.twist_c1, f32),
            untyped("twist_c2", self.twist_c2, f32),
            untyped("anchors", self.anchors, f32),
            untyped("is_static", self.is_static, f32),
            untyped("fps", np.array([self.fps], np.float64), pa.float64()),
            untyped("frame_origin", np.array([self.frame_origin], np.float64), pa.float64()),
            untyped("active_threshold", np.array([self.active_threshold], np.float32), f32),
        ]


@dataclass(frozen=True)
class TrigsParams:
    """Raw TRiGS parameters of a checkpoint (the trainer's names and activations), as f32 arrays."""

    means: Float32[ndarray, "n 3"]
    """Canonical centres."""
    quats: Float32[ndarray, "n 4"]
    """Canonical rotations wxyz, not necessarily unit."""
    scales: Float32[ndarray, "n 3"]
    """Log scales."""
    opacities: Float32[ndarray, "n"]
    """Opacity logits."""
    sh0: Float32[ndarray, "n 3"]
    """DC SH coefficient per RGB channel."""
    times: Float32[ndarray, "n"]
    """Centre times, seconds."""
    log_durations: Float32[ndarray, "n"]
    """Log temporal scales, seconds."""
    zeta_base_t: Float32[ndarray, "n 3"]
    zeta_base_r: Float32[ndarray, "n 3"]
    zeta_c0_t: Float32[ndarray, "n 3"]
    zeta_c0_r: Float32[ndarray, "n 3"]
    zeta_c1_t: Float32[ndarray, "n 3"]
    zeta_c1_r: Float32[ndarray, "n 3"]
    zeta_c2_t: Float32[ndarray, "n 3"]
    zeta_c2_r: Float32[ndarray, "n 3"]
    anchors: Float32[ndarray, "n 3"]
    """Rotation anchors."""
    is_static: Float32[ndarray, "n"] | None = None
    """Static-row flag (> 0.5 = static); None means every row moves."""


def sigmoid(x: Float32[ndarray, "n"]) -> Float32[ndarray, "n"]:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def canonical_splats(params: TrigsParams) -> rr.GaussianSplats3D:
    """Static canonical rows: the motion input of the custom viewer and the stock-viewer fallback."""
    quats: Float32[ndarray, "n 4"] = params.quats / np.linalg.norm(params.quats, axis=1, keepdims=True)
    rgba: Float32[ndarray, "n 4"] = np.clip(np.concatenate([0.5 + np.float32(SH_C0) * params.sh0, sigmoid(params.opacities)[:, None]], axis=1), 0.0, 1.0)
    colors: UInt8[ndarray, "n 4"] = (rgba * 255.0 + 0.5).astype(np.uint8)
    return rr.GaussianSplats3D(centers=params.means, quaternions=np.roll(quats, -1, axis=1), scales=np.exp(params.scales), colors=colors)


def trigs_motion(params: TrigsParams, fps: float, frame_origin: float, active_threshold: float) -> TrigsMotion3D:
    """The `TrigsMotion3D` of a checkpoint's parameters."""
    rows: int = len(params.means)
    is_static: Float32[ndarray, "n"] = np.zeros(rows, np.float32) if params.is_static is None else (params.is_static > 0.5).astype(np.float32)
    return TrigsMotion3D(
        opacities=sigmoid(params.opacities),
        times=params.times,
        log_durations=params.log_durations,
        twist_base=np.concatenate([params.zeta_base_t, params.zeta_base_r], axis=1),
        twist_c0=np.concatenate([params.zeta_c0_t, params.zeta_c0_r], axis=1),
        twist_c1=np.concatenate([params.zeta_c1_t, params.zeta_c1_r], axis=1),
        twist_c2=np.concatenate([params.zeta_c2_t, params.zeta_c2_r], axis=1),
        anchors=params.anchors,
        is_static=is_static,
        fps=fps,
        frame_origin=frame_origin,
        active_threshold=active_threshold,
    )


def log_trigs_motion(entity: str, params: TrigsParams, fps: float, frame_origin: float, active_threshold: float, *, recording: rr.RecordingStream | None = None) -> None:
    """Log the static canonical `GaussianSplats3D` and its `TrigsMotion3D` on one entity."""
    rr.log(entity, canonical_splats(params), trigs_motion(params, fps, frame_origin, active_threshold), static=True, recording=recording)
