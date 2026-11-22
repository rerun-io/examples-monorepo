"""`SogSplats3D`: a SOG v2/v3 scene logged as its stored bytes, beside a typed `GaussianSplats3D` preview.

The custom viewer decodes the WebP planes (`gsplat_core::sog`) and draws the full scene instead of the preview.
Stock viewers draw the preview only: every k-th splat in file (Morton) order, no SH.

Contract (one entity, every component static, no Rerun component type):
  GaussianSplats3D:centers/quaternions(xyzw)/scales(linear)/colors  preview rows; colour RGB = clip(0.5 + SH_C0 * sh0),
                                                                    A = clip(opacity, 0, 1)
  SogSplats3D:format         Utf8 x1  "sog/2" | "sog/3" (meta.json version)
  SogSplats3D:meta           Utf8 x1  meta.json verbatim
  SogSplats3D:means_l, :means_u, :quats, :scales, :sh0 [, :shN_centroids, :shN_labels]
                             UInt8 xN the stored WebP bytes of that plane (a flat batch, one instance per byte)
v3 blind planes are not logged. Data stays in the file frame (3DGS PLY: x right, y down, z forward).
"""

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Float32, UInt8
from numpy import ndarray

from gsplat_rust_renderer.gaussians3d import SH_C0
from gsplat_rust_renderer.sog_decode import DecodedSog, SogFile, decode

ARCHETYPE: str = "SogSplats3D"
PLANES: tuple[str, ...] = ("means_l", "means_u", "quats", "scales", "sh0", "shN_centroids", "shN_labels")
"""Plane fields in contract order; the last two only with SH."""
PREVIEW_ROWS: int = 16_384


def descriptor(field: str) -> rr.ComponentDescriptor:
    """`SogSplats3D:<field>` without a component type."""
    return rr.ComponentDescriptor(f"{ARCHETYPE}:{field}", archetype=ARCHETYPE, component_type=None)


def batch(field: str, array: pa.Array) -> rr.DescribedComponentBatch:
    field_descriptor: rr.ComponentDescriptor = descriptor(field)
    return rr.DescribedComponentBatch(rr.AnyBatchValue(field_descriptor, array), field_descriptor)


@dataclass(frozen=True)
class SogSplats3D(rr.AsComponents):
    """A SOG as stored: meta.json text and the WebP bytes of each plane."""

    sog: SogFile

    def __post_init__(self) -> None:
        missing: list[str] = [p for p in PLANES[:5] if p not in self.sog.planes]
        if missing:
            raise ValueError(f"SogSplats3D: planes {missing} are missing")
        if (self.sog.meta.shN is None) != ("shN_centroids" not in self.sog.planes):
            raise ValueError("SogSplats3D: SH planes must be present exactly when meta.json has shN")

    def as_component_batches(self) -> list[rr.DescribedComponentBatch]:
        out: list[rr.DescribedComponentBatch] = [
            batch("format", pa.array([f"sog/{self.sog.meta.version}"], pa.utf8())),
            batch("meta", pa.array([self.sog.meta_json], pa.utf8())),
        ]
        for plane in PLANES:
            if plane in self.sog.planes:
                out.append(batch(plane, pa.array(np.frombuffer(self.sog.planes[plane], np.uint8), pa.uint8())))
        return out


def preview_splats(decoded: DecodedSog, rows: int = PREVIEW_ROWS) -> rr.GaussianSplats3D | None:
    """Every k-th splat (k = ceil(n / rows)) as a typed preview; None for rows = 0."""
    if rows <= 0:
        return None
    step: int = max(1, -(-len(decoded.means) // rows))
    quats: Float32[ndarray, "m 4"] = decoded.quats_wxyz[::step]
    norms: Float32[ndarray, "m 1"] = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = np.where(norms > 1e-12, quats / np.where(norms > 1e-12, norms, 1.0), np.array([1.0, 0.0, 0.0, 0.0], np.float32)).astype(np.float32)
    rgba: Float32[ndarray, "m 4"] = np.clip(
        np.concatenate([0.5 + np.float32(SH_C0) * decoded.sh0[::step], decoded.opacity[::step, None].astype(np.float32)], axis=1), 0.0, 1.0
    )
    colors: UInt8[ndarray, "m 4"] = (rgba * 255.0 + 0.5).astype(np.uint8)
    return rr.GaussianSplats3D(
        centers=decoded.means[::step], quaternions=np.roll(quats, -1, axis=1), scales=np.exp(decoded.log_scales[::step]), colors=colors
    )


def log_sog(entity: str, sog: SogFile, preview_rows: int = PREVIEW_ROWS, *, recording: rr.RecordingStream | None = None) -> None:
    """Log the `SogSplats3D` payload and its preview (unless `preview_rows` is 0) on one static entity."""
    preview: rr.GaussianSplats3D | None = preview_splats(decode(sog), preview_rows)
    payload: SogSplats3D = SogSplats3D(sog)
    if preview is None:
        rr.log(entity, payload, static=True, recording=recording)
    else:
        rr.log(entity, preview, payload, static=True, recording=recording)
