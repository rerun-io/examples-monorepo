"""SOG v2/v3 reader and numpy decoder: the independent oracle of `gsplat_core::sog` and the source of preview rows.

A port of splat-transform `read-sog.ts` `decodeInto` (5d5be2b1), vectorised in f64; codebooks are f32 as there.
One addition from the PlayCanvas engine: a null `codebook[0]` (old tool bug) becomes `cb[1] + (cb[1] - cb[255]) / 255`.
SH bands come from `shN.bands` or, when absent (older v2 files such as PlayCanvas's skull.sog), from the centroid
width {192: 1, 512: 2, 960: 3, 1536: 4} as the engine infers them (splat-transform would drop that SH).
Opacity stays linear and unclamped here (v3 reaches 1.5625); `core_layout` applies the gsplat-core clamps.
v1 files (no `version`) and v3 blind data are out of scope.
"""

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float32, Float64, Int64, UInt8
from numpy import ndarray
from PIL import Image
from serde import SerdeError, field, serde
from serde.core import coerce
from serde.json import from_json

SH_REST_BY_WIDTH: dict[int, int] = {192: 1, 512: 2, 960: 3, 1536: 4}
"""Centroid texture width (64 x coefficients) -> SH bands."""


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class MeansMeta:
    mins: list[float]
    maxs: list[float]
    files: list[str]


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class CodebookMeta:
    codebook: list[float | None]
    files: list[str]


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class FilesMeta:
    files: list[str]


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class ShNMeta:
    codebook: list[float | None]
    files: list[str]
    count: int | None = None
    """Palette entries; absent in older v2 files (then centroid rows x 64)."""
    bands: int | None = None
    """1-4; absent in older v2 files (then inferred from the centroid width)."""


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class SogMeta:
    """meta.json of a v2/v3 SOG (third-party schema: unknown fields are ignored)."""

    version: int
    count: int
    means: MeansMeta
    scales: CodebookMeta
    quats: FilesMeta
    sh0: CodebookMeta
    shN: ShNMeta | None = field(default=None, rename="shN")

    def __post_init__(self) -> None:
        if self.version not in (2, 3):
            raise ValueError(f"SOG version {self.version} is not supported (v2 and v3 only)")


def parse_meta(text: str, source: str) -> SogMeta:
    """Typed meta.json; v1 (no `version`) and other versions raise a ValueError naming `source`."""
    try:
        return from_json(SogMeta, text)
    except (SerdeError, ValueError) as error:
        raise ValueError(f"{source}: not a SOG v2/v3 meta.json ({error})") from error


@dataclass(frozen=True, slots=True)
class SogFile:
    """A SOG as stored: meta.json text and the WebP bytes of each plane, by role (the SogSplats3D field names)."""

    meta_json: str
    meta: SogMeta
    planes: dict[str, bytes]
    """Keys: means_l, means_u, quats, scales, sh0 and, with SH, shN_centroids, shN_labels."""


def plane_roles(meta: SogMeta) -> dict[str, str]:
    """Role -> file name from meta.json (`files[]` order is significant)."""
    roles: dict[str, str] = {
        "means_l": meta.means.files[0],
        "means_u": meta.means.files[1],
        "quats": meta.quats.files[0],
        "scales": meta.scales.files[0],
        "sh0": meta.sh0.files[0],
    }
    if meta.shN is not None:
        roles["shN_centroids"], roles["shN_labels"] = meta.shN.files[0], meta.shN.files[1]
    return roles


def read_sog(path: Path) -> SogFile:
    """A bundled `.sog` (zip with the files at its root), a `meta.json`, or a directory holding one."""
    if path.is_dir():
        path = path / "meta.json"
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names: set[str] = set(archive.namelist())

            def read(name: str) -> bytes:
                if name not in names:
                    raise ValueError(f"{path}: {name} is missing from the archive")
                return archive.read(name)

            meta_json: str = read("meta.json").decode()
            meta: SogMeta = parse_meta(meta_json, str(path))
            return SogFile(meta_json, meta, {role: read(name) for role, name in plane_roles(meta).items()})
    meta_json = path.read_text()
    meta = parse_meta(meta_json, str(path))
    return SogFile(meta_json, meta, {role: (path.parent / name).read_bytes() for role, name in plane_roles(meta).items()})


def decode_webp(data: bytes) -> UInt8[ndarray, "h w 4"]:
    """Lossless WebP -> RGBA8 as stored (Pillow; RGB images get A = 255)."""
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "WEBP":
            raise ValueError(f"SOG plane is {image.format}, not WebP")
        return np.asarray(image.convert("RGBA"), dtype=np.uint8)


def codebook(entries: list[float | None], name: str) -> Float32[ndarray, "256"]:
    """256 f32 entries; a null entry 0 is patched as the PlayCanvas engine does, any other null raises."""
    if len(entries) != 256:
        raise ValueError(f"SOG {name} codebook needs 256 entries, got {len(entries)}")
    rest: list[float | None] = entries[1:]
    if any(v is None for v in rest):
        raise ValueError(f"SOG {name} codebook has a null entry beyond index 0")
    values: list[float] = [v for v in rest if v is not None]
    first: float = entries[0] if entries[0] is not None else values[0] + (values[0] - values[-1]) / 255.0
    return np.array([first, *values], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class DecodedSog:
    """Per-splat attributes as splat-transform decodes them (file frame, quaternion wxyz)."""

    means: Float32[ndarray, "n 3"]
    quats_wxyz: Float32[ndarray, "n 4"]
    """Not renormalised (v2 smallest-three as stored)."""
    log_scales: Float32[ndarray, "n 3"]
    opacity: Float64[ndarray, "n"]
    """Linear opacity; v3 may exceed 1 (up to 1.5625)."""
    sh0: Float32[ndarray, "n 3"]
    """SH DC coefficient per RGB channel."""
    sh_rest: Float32[ndarray, "n c 3"]
    """AC coefficients, coefficient-major then RGB; c = 0 without SH."""
    sh_bands: int


def sh_bands(meta: SogMeta, centroid_width: int) -> int:
    inferred: int | None = SH_REST_BY_WIDTH.get(centroid_width)
    bands: int | None = meta.shN.bands if meta.shN is not None else None
    if inferred is None or (bands is not None and bands != inferred):
        raise ValueError(f"SOG shN centroid width {centroid_width} does not fit {bands} SH bands")
    return inferred


def texels(image: UInt8[ndarray, "h w 4"], count: int, name: str) -> UInt8[ndarray, "n 4"]:
    flat: UInt8[ndarray, "m 4"] = image.reshape(-1, 4)
    if len(flat) < count:
        raise ValueError(f"SOG {name} plane holds {len(flat)} texels, fewer than the {count} splats")
    return flat[:count]


QUAT_SLOTS: tuple[tuple[int, int, int], ...] = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))
"""v2 smallest-three: wxyz slots of the stored components for each index of the dropped largest one."""


def decode(sog: SogFile) -> DecodedSog:
    """Every splat of `sog` (the oracle: same formulas as splat-transform, f64 arithmetic)."""
    meta: SogMeta = sog.meta
    n: int = meta.count
    v3: bool = meta.version == 3

    def plane(role: str) -> UInt8[ndarray, "n 4"]:
        return texels(decode_webp(sog.planes[role]), n, role)

    lo: UInt8[ndarray, "n 4"] = plane("means_l")
    hi: UInt8[ndarray, "n 4"] = plane("means_u")
    code: Float64[ndarray, "n 3"] = (lo[:, :3].astype(np.float64) + hi[:, :3].astype(np.float64) * 256.0) / 65535.0
    mins: Float64[ndarray, "3"] = np.array(meta.means.mins, np.float64)
    span: Float64[ndarray, "3"] = np.array(meta.means.maxs, np.float64) - mins
    span[span == 0.0] = 1.0
    v: Float64[ndarray, "n 3"] = mins + span * code
    means: Float64[ndarray, "n 3"] = np.where(v < 0.0, -1.0, 1.0) * (np.exp(np.abs(v)) - 1.0)

    q: UInt8[ndarray, "n 4"] = plane("quats")
    quats: Float64[ndarray, "n 4"]
    if v3:
        p: Float64[ndarray, "n 3"] = (q[:, :3].astype(np.float64) - 128.0) / 127.0
        r: Float64[ndarray, "n"] = np.linalg.norm(p, axis=1)
        t: Float64[ndarray, "n"] = np.pi / 2.0 * r
        k: Float64[ndarray, "n"] = np.where(r > 1e-12, np.sin(t) / np.where(r > 1e-12, r, 1.0), np.pi / 2.0)
        quats = np.concatenate([np.cos(t)[:, None], p * k[:, None]], axis=1)
    else:
        quats = np.zeros((n, 4), np.float64)
        quats[:, 0] = 1.0
        c: Float64[ndarray, "n 3"] = (q[:, :3].astype(np.float64) / 255.0 * 2.0 - 1.0) / np.sqrt(2.0)
        largest_d: Float64[ndarray, "n"] = np.sqrt(np.maximum(0.0, 1.0 - (c * c).sum(axis=1)))
        for largest, slots in enumerate(QUAT_SLOTS):
            rows: ndarray = q[:, 3] == 252 + largest
            quats[rows] = 0.0
            quats[np.ix_(rows, slots)] = c[rows]
            quats[rows, largest] = largest_d[rows]

    scales_cb: Float32[ndarray, "256"] = codebook(meta.scales.codebook, "scales")
    sh0_cb: Float32[ndarray, "256"] = codebook(meta.sh0.codebook, "sh0")
    s: UInt8[ndarray, "n 4"] = plane("scales")
    c0: UInt8[ndarray, "n 4"] = plane("sh0")
    alpha: Float64[ndarray, "n"] = c0[:, 3].astype(np.float64)
    opacity: Float64[ndarray, "n"] = (alpha / 204.0) ** 2 if v3 else alpha / 255.0

    bands: int = 0
    sh_rest: Float32[ndarray, "n c 3"] = np.zeros((n, 0, 3), np.float32)
    if meta.shN is not None and "shN_centroids" in sog.planes and "shN_labels" in sog.planes:
        centroids: UInt8[ndarray, "h w 4"] = decode_webp(sog.planes["shN_centroids"])
        bands = sh_bands(meta, centroids.shape[1])
        coeffs: int = (bands + 1) ** 2 - 1
        palette: UInt8[ndarray, "m 4"] = centroids.reshape(-1, 4)
        palette_count: int = meta.shN.count if meta.shN.count is not None else centroids.shape[0] * 64
        palette_count = min(palette_count, len(palette) // coeffs)
        lab: Int64[ndarray, "n 4"] = plane("shN_labels").astype(np.int64)
        label: ndarray = lab[:, 0] | (lab[:, 1] << 8) | ((lab[:, 2] << 16) if v3 else 0)
        valid: ndarray = label < palette_count
        index: ndarray = np.where(valid, label, 0)[:, None] * coeffs + np.arange(coeffs)[None, :]
        rest_cb: Float32[ndarray, "256"] = codebook(meta.shN.codebook, "shN")
        sh_rest = rest_cb[palette[index][..., :3]]
        sh_rest[~valid] = 0.0

    return DecodedSog(
        means=means.astype(np.float32),
        quats_wxyz=quats.astype(np.float32),
        log_scales=scales_cb[s[:, :3]],
        opacity=opacity,
        sh0=sh0_cb[c0[:, :3]],
        sh_rest=sh_rest,
        sh_bands=bands,
    )


@dataclass(frozen=True, slots=True)
class CoreSplats:
    """`gsplat_core::Splats` as numbers: transforms (mean, wxyz, log scale), raw logit opacity, SH (DC first)."""

    transforms: Float32[ndarray, "n 10"]
    raw_opacities: Float32[ndarray, "n"]
    sh: Float32[ndarray, "n k 3"]


def core_layout(decoded: DecodedSog) -> CoreSplats:
    """The gsplat-core form: opacity clamped to 1, then to [1e-6, 1 - 1e-6], logit (what `gsplat_core::sog` returns)."""
    alpha: Float64[ndarray, "n"] = np.clip(np.minimum(decoded.opacity, 1.0), 1e-6, 1.0 - 1e-6)
    return CoreSplats(
        transforms=np.concatenate([decoded.means, decoded.quats_wxyz, decoded.log_scales], axis=1),
        raw_opacities=np.log(alpha / (1.0 - alpha)).astype(np.float32),
        sh=np.concatenate([decoded.sh0[:, None, :], decoded.sh_rest], axis=1),
    )
