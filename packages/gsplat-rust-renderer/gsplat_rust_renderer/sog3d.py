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

`SogLodSplats3D`: a Streamed SOG (PlayCanvas LOD: lod-meta.json + one SOG per file, `sog_lod.py`) logged as stored.
The custom viewer uploads every file into one scene and picks a level per leaf and per view. Stock viewers draw the
preview: every k-th splat of the coarsest level with at least `preview_rows` rows. The entity has no `SogSplats3D`.
  SogLodSplats3D:format       Utf8 x1   "sog-lod/1" (lod-meta.json version)
  SogLodSplats3D:lod_meta     Utf8 x1   lod-meta.json verbatim
  SogLodSplats3D:files        Utf8 xF   meta.json path of each file relative to lod-meta.json: `filenames`, then
                                        `environment` when present (file index = lod-meta file index)
  SogLodSplats3D:metas        Utf8 xF   meta.json of each file verbatim, same order
  SogLodSplats3D:<plane>      UInt8 xN  the stored WebP bytes of that plane of every file, concatenated in file order
                                        (plane names as SogSplats3D)
  SogLodSplats3D:plane_sizes  UInt32 x(F*7)  byte size of plane p of file f at f * 7 + p (PLANES order), 0 = absent

`AnimatedSogSplats3D`: an animated-sog `.asog` v1 clip logged as its file bytes (`gsplat_core::asog` decodes it and
evaluates the motion at the time cursor). The same entity's `GaussianSplats3D` preview is ONE pose, the clip middle
(rows with weight > 0, opacity x weight, every k-th of them), never the canonical rows (a ghost trail).
  AnimatedSogSplats3D:format            Utf8 x1    "asog/1"
  AnimatedSogSplats3D:asog              UInt8 xN   the .asog file verbatim
  AnimatedSogSplats3D:frame_origin      Float64 x1 timeline frame at clip time 0: t [s] = (frame - frame_origin) / frameRate,
                                                   frameRate from the manifest (no fps component)
  AnimatedSogSplats3D:active_threshold  Float32 x1 temporal cut the viewer applies (absent = 0.05, the format's constant)
"""

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Float32, UInt8
from numpy import ndarray

from gsplat_rust_renderer.asog_decode import TEMPORAL_THRESHOLD, AsogFile, AsogFrame, DecodedAsog, evaluate
from gsplat_rust_renderer.asog_decode import decode as decode_asog
from gsplat_rust_renderer.gaussians3d import SH_C0
from gsplat_rust_renderer.sog_decode import DecodedSog, SogFile, decode
from gsplat_rust_renderer.sog_lod import SogLodSet

ARCHETYPE: str = "SogSplats3D"
PLANES: tuple[str, ...] = ("means_l", "means_u", "quats", "scales", "sh0", "shN_centroids", "shN_labels")
"""Plane fields in contract order; the last two only with SH."""
PREVIEW_ROWS: int = 16_384


ANIMATED_ARCHETYPE: str = "AnimatedSogSplats3D"
LOD_ARCHETYPE: str = "SogLodSplats3D"


def descriptor(field: str, archetype: str = ARCHETYPE) -> rr.ComponentDescriptor:
    """`<archetype>:<field>` without a component type."""
    return rr.ComponentDescriptor(f"{archetype}:{field}", archetype=archetype, component_type=None)


def batch(field: str, array: pa.Array, archetype: str = ARCHETYPE) -> rr.DescribedComponentBatch:
    field_descriptor: rr.ComponentDescriptor = descriptor(field, archetype)
    return rr.DescribedComponentBatch(rr.AnyBatchValue(field_descriptor, array), field_descriptor)


def validate_sog(sog: SogFile) -> None:
    """Raise ValueError unless the five required planes are present and the SH planes match meta.json `shN`."""
    missing: list[str] = [p for p in PLANES[:5] if p not in sog.planes]
    if missing:
        raise ValueError(f"SogSplats3D: planes {missing} are missing")
    if (sog.meta.shN is None) != ("shN_centroids" not in sog.planes):
        raise ValueError("SogSplats3D: SH planes must be present exactly when meta.json has shN")


@dataclass(frozen=True)
class SogSplats3D(rr.AsComponents):
    """A SOG as stored: meta.json text and the WebP bytes of each plane."""

    sog: SogFile

    def __post_init__(self) -> None:
        validate_sog(self.sog)

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


@dataclass(frozen=True)
class SogLodSplats3D(rr.AsComponents):
    """A Streamed SOG as stored: lod-meta.json, then meta.json and the WebP planes of every file."""

    lod: SogLodSet

    def __post_init__(self) -> None:
        # Per-file plane checks only: files may differ in SH (the core pads absent coefficients with zeros).
        for sog in self.lod.sogs:
            validate_sog(sog)

    def as_component_batches(self) -> list[rr.DescribedComponentBatch]:
        sizes: list[int] = [len(sog.planes.get(plane, b"")) for sog in self.lod.sogs for plane in PLANES]
        out: list[rr.DescribedComponentBatch] = [
            batch("format", pa.array(["sog-lod/1"], pa.utf8()), LOD_ARCHETYPE),
            batch("lod_meta", pa.array([self.lod.lod_meta_json], pa.utf8()), LOD_ARCHETYPE),
            batch("files", pa.array(self.lod.files, pa.utf8()), LOD_ARCHETYPE),
            batch("metas", pa.array([sog.meta_json for sog in self.lod.sogs], pa.utf8()), LOD_ARCHETYPE),
            batch("plane_sizes", pa.array(np.array(sizes, np.uint32), pa.uint32()), LOD_ARCHETYPE),
        ]
        for plane in PLANES:
            data: bytes = b"".join(sog.planes.get(plane, b"") for sog in self.lod.sogs)
            if data:
                out.append(batch(plane, pa.array(np.frombuffer(data, np.uint8), pa.uint8()), LOD_ARCHETYPE))
        return out


def lod_preview_level(lod: SogLodSet, rows: int) -> int:
    """The coarsest level with at least `rows` splats (level 0 when none has)."""
    totals: list[int] = [sum(c for c, level in zip(lod.file_counts, lod.file_levels, strict=True) if level == k) for k in range(lod.levels)]
    return max((k for k, total in enumerate(totals) if total >= rows), default=0)


def log_sog_lod(entity: str, lod: SogLodSet, preview_rows: int = PREVIEW_ROWS, *, recording: rr.RecordingStream | None = None) -> None:
    """Log the `SogLodSplats3D` payload and its preview (every k-th splat of `lod_preview_level`) on one static entity.

    `preview_rows` must be > 0: the custom viewer attaches the payload to the preview rows.
    """
    if preview_rows <= 0:
        raise ValueError("log_sog_lod: preview_rows must be > 0 (the viewer draws the payload of entities with rows)")
    level: int = lod_preview_level(lod, preview_rows)
    parts: list[DecodedSog] = [decode(sog) for sog, file_level in zip(lod.sogs, lod.file_levels, strict=True) if file_level == level]
    merged: DecodedSog = DecodedSog(
        means=np.concatenate([p.means for p in parts]),
        quats_wxyz=np.concatenate([p.quats_wxyz for p in parts]),
        log_scales=np.concatenate([p.log_scales for p in parts]),
        opacity=np.concatenate([p.opacity for p in parts]),
        sh0=np.concatenate([p.sh0 for p in parts]),
        sh_rest=np.zeros((sum(len(p.means) for p in parts), 0, 3), np.float32),
        sh_bands=0,
    )
    preview: rr.GaussianSplats3D | None = preview_splats(merged, preview_rows)
    assert preview is not None, "preview_rows > 0 always gives a preview"
    rr.log(entity, preview, SogLodSplats3D(lod), static=True, recording=recording)


@dataclass(frozen=True)
class AnimatedSogSplats3D(rr.AsComponents):
    """An `.asog` clip as stored, plus what the file lacks: the Rerun frame of clip time 0 and the temporal cut."""

    asog: AsogFile
    frame_origin: float = 0.0
    active_threshold: float = TEMPORAL_THRESHOLD

    def __post_init__(self) -> None:
        if not 0.0 <= self.active_threshold < 1.0:
            raise ValueError(f"AnimatedSogSplats3D: active_threshold {self.active_threshold} is outside [0, 1)")

    def as_component_batches(self) -> list[rr.DescribedComponentBatch]:
        return [
            batch("format", pa.array([f"asog/{self.asog.manifest.version}"], pa.utf8()), ANIMATED_ARCHETYPE),
            batch("asog", pa.array(np.frombuffer(self.asog.data, np.uint8), pa.uint8()), ANIMATED_ARCHETYPE),
            batch("frame_origin", pa.array([self.frame_origin], pa.float64()), ANIMATED_ARCHETYPE),
            batch("active_threshold", pa.array([self.active_threshold], pa.float32()), ANIMATED_ARCHETYPE),
        ]


def asog_preview(
    decoded: DecodedAsog, frame: AsogFrame, rows: int = PREVIEW_ROWS
) -> rr.GaussianSplats3D | None:
    """One evaluated pose as a typed preview: rows with weight > 0, opacity x weight, every k-th; None for rows = 0."""
    if rows <= 0:
        return None
    alive: ndarray = np.nonzero(frame.weights > 0)[0]
    alive = alive[:: max(1, -(-len(alive) // rows))]
    quats: Float32[ndarray, "m 4"] = frame.rotations[alive].astype(np.float32)
    rgba: Float32[ndarray, "m 4"] = np.clip(
        np.concatenate(
            [0.5 + SH_C0 * decoded.sh0[alive], (decoded.opacity[alive] * frame.weights[alive])[:, None]], axis=1
        ).astype(np.float32),
        0.0,
        1.0,
    )
    colors: UInt8[ndarray, "m 4"] = (rgba * 255.0 + 0.5).astype(np.uint8)
    return rr.GaussianSplats3D(
        centers=frame.positions[alive].astype(np.float32),
        quaternions=np.roll(quats, -1, axis=1),
        scales=np.exp(decoded.log_scales[alive]).astype(np.float32),
        colors=colors,
    )


def log_asog(
    entity: str,
    asog: AsogFile,
    frame_origin: float = 0.0,
    active_threshold: float = TEMPORAL_THRESHOLD,
    preview_rows: int = PREVIEW_ROWS,
    *,
    fps: float | None = None,
    recording: rr.RecordingStream | None = None,
) -> None:
    """Log the `AnimatedSogSplats3D` payload and its mid-clip preview on one static entity.

    `fps` is the frame rate of the timeline the caller logs on; it must equal the manifest's frameRate, because the
    viewer maps frames to clip seconds with the manifest's rate.
    """
    rate: float = asog.manifest.timeline.frame_rate
    if fps is not None and fps != rate:
        raise ValueError(f"log_asog: timeline fps {fps} differs from the .asog frameRate {rate}")
    payload: AnimatedSogSplats3D = AnimatedSogSplats3D(asog, frame_origin, active_threshold)
    preview: rr.GaussianSplats3D | None = None
    if preview_rows > 0:
        decoded: DecodedAsog = decode_asog(asog)
        mid: AsogFrame = evaluate(decoded, asog.manifest, asog.manifest.duration / 2, active_threshold)
        preview = asog_preview(decoded, mid, preview_rows)
    if preview is None:
        rr.log(entity, payload, static=True, recording=recording)
    else:
        rr.log(entity, preview, payload, static=True, recording=recording)
