"""`.asog` v1 reader and numpy decoder: the source of the `AnimatedSogSplats3D` preview pose in the gsplat env.

A port of the animated-sog reference decoder (`asog.container`, `asog.decode.read`, `Asog.evaluate`, `asog.motion`
at the `asog-src` pin bc976cf; format.md sections 1-5), f64. It exists so the gsplat env needs no `asog-src` recipe
(that recipe lives on the open4dgs stack); `tests/test_asog_decode.py` checks it against the reference decode stored
in `crates/gsplat-viewer/tests/fixtures/asog_reference.rrd`. The viewer decodes with `gsplat_core::asog`.
"""

import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray
from serde import SerdeError, field, serde
from serde.core import coerce
from serde.json import from_json

HEADER_SIZE: int = 64
TEMPORAL_THRESHOLD: float = 0.05
"""The format's temporal cut (no manifest field)."""


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class Timeline:
    frame_rate: float = field(rename="frameRate")
    total_frames: int | None = field(rename="totalFrames", default=None)
    duration_seconds: float | None = field(rename="durationSeconds", default=None)


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class ClipGlobal:
    frame_start: float = field(rename="frameStart")
    active_time_start: float = field(rename="activeTimeStart")
    active_time_end: float = field(rename="activeTimeEnd")
    core_time_start: float = field(rename="coreTimeStart")
    core_time_end: float = field(rename="coreTimeEnd")


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class ClipLocal:
    frame_start: float = field(rename="frameStart")


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class ClipRender:
    fade_in_frames: float = field(rename="fadeInFrames", default=0.0)
    fade_out_frames: float = field(rename="fadeOutFrames", default=0.0)


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class Clip:
    global_: ClipGlobal = field(rename="global")
    local: ClipLocal
    render: ClipRender = field(default_factory=ClipRender)


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class Bounds:
    mins: list[float]
    maxs: list[float]


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class Codebook:
    codebook: list[float]


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class ShN:
    codebook: list[float]
    count: int
    coeffs: int


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class MuMeta:
    scale: float


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class LogSigmaMeta:
    min: float
    max: float
    bits: int


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class MotionMeta:
    qp: float
    qa: float
    sm: float
    mid_o: float = field(rename="midO")
    mid_g: float = field(rename="midG")
    mref_samples: int
    r_min: float


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class StreamDesc:
    name: str
    count: int
    channels: int
    planes: int
    delta: bool
    zigzag: bool
    byte_offset: int = field(rename="byteOffset")
    byte_length: int = field(rename="byteLength")


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class AsogManifest:
    """The `.asog` JSON manifest (third-party schema: unknown fields allowed)."""

    format: str
    version: int
    count: int
    timeline: Timeline
    clips: list[Clip]
    means: Bounds
    scales: Codebook
    sh0: Codebook
    mu: MuMeta
    logsigma: LogSigmaMeta
    motion: MotionMeta
    streams: list[StreamDesc]
    sh_n: ShN | None = field(rename="shN", default=None)

    @property
    def duration(self) -> float:
        """Clip duration in seconds (`durationSeconds`, else `totalFrames / frameRate`, else the last active end)."""
        if self.timeline.duration_seconds is not None:
            return self.timeline.duration_seconds
        if self.timeline.total_frames is not None:
            return self.timeline.total_frames / self.timeline.frame_rate
        return max((c.global_.active_time_end for c in self.clips), default=0.0)


@dataclass(frozen=True, slots=True)
class AsogFile:
    """A whole `.asog` file: its bytes (logged verbatim) and the parsed manifest."""

    data: bytes
    manifest: AsogManifest


def parse_asog(data: bytes, source: str) -> AsogFile:
    """Header + manifest of `.asog` bytes; `source` names the file in errors."""
    if len(data) < HEADER_SIZE or data[:4] != b"ASOG":
        raise ValueError(f"{source}: not an .asog file (magic {data[:4]!r})")
    offset, length = struct.unpack("<QQ", data[16:32])
    try:
        manifest: AsogManifest = from_json(AsogManifest, data[offset : offset + length].decode())
    except (SerdeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{source}: bad .asog manifest: {error}") from error
    if manifest.format != "asog" or manifest.version != 1:
        raise ValueError(f"{source}: manifest format {manifest.format!r} v{manifest.version}, expected asog v1")
    return AsogFile(data, manifest)


def read_asog(path: Path) -> AsogFile:
    return parse_asog(path.read_bytes(), str(path))


def decode_streams(asog: AsogFile) -> dict[str, Int64[ndarray, "_ _"]]:
    """Every stream as `[count, channels]` integers (inflate, byte planes, un-zigzag, running sum)."""
    payload_offset: int = struct.unpack("<Q", asog.data[32:40])[0]
    out: dict[str, Int64[ndarray, "_ _"]] = {}
    for s in asog.manifest.streams:
        start: int = payload_offset + s.byte_offset
        raw: bytes = zlib.decompress(asog.data[start : start + s.byte_length])
        if len(raw) != s.count * s.channels * s.planes:
            raise ValueError(f"stream {s.name}: inflated {len(raw)} bytes, expected {s.count * s.channels * s.planes}")
        planes: ndarray = np.frombuffer(raw, np.uint8).reshape(s.planes, s.channels, s.count).astype(np.uint64)
        z: ndarray = np.zeros((s.channels, s.count), np.uint64)
        for p in range(s.planes):
            z |= planes[p] << np.uint64(8 * p)
        x: ndarray = z.astype(np.int64)
        if s.zigzag:
            x = (x >> 1) ^ -(x & 1)
        if s.delta:
            x = np.cumsum(x, axis=1)
        out[s.name] = x.T
    return out


@dataclass(frozen=True, slots=True)
class DecodedAsog:
    """De-quantised splats; motion fields are zero for static rows."""

    means: Float64[ndarray, "n 3"]
    quats_wxyz: Float64[ndarray, "n 4"]
    log_scales: Float64[ndarray, "n 3"]
    sh0: Float64[ndarray, "n 3"]
    opacity: Float64[ndarray, " n"]
    mu: Float64[ndarray, " n"]
    sigma: Float64[ndarray, " n"]
    anchor: Float64[ndarray, "n 3"]
    omega: Float64[ndarray, "3 n 3"]
    g: Float64[ndarray, "5 n"]
    static: Bool[ndarray, " n"]
    clip: Int64[ndarray, " n"]


def bezier2(tau: Float64[ndarray, " n"], c: Float64[ndarray, "3 n 3"]) -> Float64[ndarray, "n 3"]:
    t: Float64[ndarray, "n 1"] = tau[:, None]
    u: Float64[ndarray, "n 1"] = 1 - t
    return u * u * c[0] + 2 * u * t * c[1] + t * t * c[2]


def decode(asog: AsogFile) -> DecodedAsog:
    """Reference `asog.decode.read`, static and motion parts (SH is not needed for a preview and is not decoded)."""
    m: AsogManifest = asog.manifest
    s: dict[str, Int64[ndarray, "_ _"]] = decode_streams(asog)
    n: int = m.count
    mins: Float64[ndarray, " 3"] = np.array(m.means.mins)
    v: Float64[ndarray, "n 3"] = mins + s["means"] / 65535 * (np.array(m.means.maxs) - mins)
    means: Float64[ndarray, "n 3"] = np.sign(v) * np.expm1(np.abs(v))
    rest: Float64[ndarray, "n 3"] = (s["quats"] / 255 - 0.5) * np.sqrt(2)
    big: Int64[ndarray, " n"] = s["quatmode"][:, 0]
    quats: Float64[ndarray, "n 4"] = np.zeros((n, 4))
    largest: Float64[ndarray, " n"] = np.sqrt(np.maximum(0, 1 - (rest * rest).sum(1)))
    for mode in range(4):
        sel: Bool[ndarray, " n"] = big == mode
        quats[np.ix_(sel, [k for k in range(4) if k != mode])] = rest[sel]
        quats[sel, mode] = largest[sel]
    mo: MotionMeta = m.motion
    ls: LogSigmaMeta = m.logsigma
    sigma: Float64[ndarray, " n"] = np.exp(ls.min + s["logsigma"][:, 0] / (2**ls.bits - 1) * (ls.max - ls.min))
    life: Float64[ndarray, " n"] = mo.sm * sigma
    qpv: float = mo.qp / 1000
    anchor: Float64[ndarray, "n 3"] = s["anchor"] * mo.qa / 1000
    step_o: Float64[ndarray, " n"] = qpv / np.maximum(np.linalg.norm(means - anchor, axis=1), mo.r_min)
    w_o: tuple[float, float, float] = (1.0, mo.mid_o, 1.0)
    o_life: Float64[ndarray, "3 n 3"] = np.stack([s["omega"][:, 3 * k : 3 * k + 3] * step_o[:, None] * w_o[k] for k in range(3)])
    norms: Float64[ndarray, "k n"] = np.stack(
        [np.linalg.norm(bezier2(np.full(n, t), o_life), axis=1) for t in np.linspace(0, 1, mo.mref_samples)]
    )
    mref: Float64[ndarray, " n"] = np.maximum(norms.min(0), 1e-9) * qpv
    w_g: tuple[float, ...] = (1.0, mo.mid_g, mo.mid_g, mo.mid_g, 1.0)
    flags: Int64[ndarray, " n"] = s["flags"][:, 0]
    return DecodedAsog(
        means=means,
        quats_wxyz=quats,
        log_scales=np.array(m.scales.codebook)[s["scales"]],
        sh0=np.array(m.sh0.codebook)[s["sh0"]],
        opacity=s["opacity"][:, 0] / 255,
        mu=s["mu"][:, 0] / m.mu.scale,
        sigma=sigma,
        anchor=anchor,
        omega=o_life / life[None, :, None],
        g=np.stack([s["g"][:, j] * mref * w_g[j] / life**2 for j in range(5)]),
        static=(flags & 1) == 1,
        clip=flags >> 1,
    )


@dataclass(frozen=True, slots=True)
class AsogFrame:
    """The splats at one time: centres, unit wxyz rotations (w >= 0) and the opacity factor (0 = not drawn)."""

    positions: Float64[ndarray, "n 3"]
    rotations: Float64[ndarray, "n 4"]
    weights: Float64[ndarray, " n"]


def _smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def clip_weights(manifest: AsogManifest, clip: Int64[ndarray, " n"], t: float) -> tuple[Float64[ndarray, " n"], Float64[ndarray, " n"]]:
    """Per row at timeline time `t`: the clip fade and the clip-local time (reference `Asog.clip_weights`)."""
    fps: float = manifest.timeline.frame_rate
    offset: Float64[ndarray, " c"] = np.zeros(len(manifest.clips))
    fade: Float64[ndarray, " c"] = np.zeros(len(manifest.clips))
    for c, entry in enumerate(manifest.clips):
        g: ClipGlobal = entry.global_
        offset[c] = (g.frame_start - entry.local.frame_start) / fps
        fade_in: float = entry.render.fade_in_frames / fps
        fade_out: float = entry.render.fade_out_frames / fps
        if not g.active_time_start <= t < g.active_time_end:
            fade[c] = 0.0
        elif fade_in > 0 and t < g.core_time_start:
            fade[c] = _smoothstep((t - (g.core_time_start - fade_in)) / fade_in)
        elif fade_out > 0 and t >= g.core_time_end:
            fade[c] = _smoothstep((g.core_time_end + fade_out - t) / fade_out)
        else:
            fade[c] = 1.0
    return fade[clip], t - offset[clip]


def evaluate(decoded: DecodedAsog, manifest: AsogManifest, t: float, threshold: float = TEMPORAL_THRESHOLD) -> AsogFrame:
    """Reference `Asog.evaluate` at timeline time `t` with temporal cut `threshold`."""
    fade, local_t = clip_weights(manifest, decoded.clip, t)
    dt: Float64[ndarray, " n"] = local_t - decoded.mu
    gamma: Float64[ndarray, " n"] = np.exp(-0.5 * (dt / decoded.sigma) ** 2)
    temporal: Float64[ndarray, " n"] = np.maximum((gamma - threshold) / (1 - threshold), 0) if threshold > 0 else gamma
    weights: Float64[ndarray, " n"] = np.where(decoded.static, fade, fade * temporal)
    positions: Float64[ndarray, "n 3"] = decoded.means.copy()
    rotations: Float64[ndarray, "n 4"] = decoded.quats_wxyz.copy()
    rows: Int64[ndarray, " m"] = np.nonzero(~decoded.static & (weights > 0))[0]
    if len(rows):
        sigma: Float64[ndarray, " m"] = decoded.sigma[rows]
        tau: Float64[ndarray, " m"] = np.clip((dt[rows] + 2 * sigma) / (4 * sigma), 0, 1)
        om: Float64[ndarray, "m 3"] = bezier2(tau, decoded.omega[:, rows])
        u: Float64[ndarray, " m"] = 1 - tau
        quartic: Float64[ndarray, "5 m"] = np.stack([u**4, 4 * u**3 * tau, 6 * u * u * tau * tau, 4 * u * tau**3, tau**4])
        g: Float64[ndarray, " m"] = (quartic * decoded.g[:, rows]).sum(0)
        oo: Float64[ndarray, "m 1"] = (om * om).sum(1, keepdims=True)
        nu: Float64[ndarray, "m 3"] = g[:, None] / np.maximum(oo, 1e-24) * om
        ok: Bool[ndarray, "m 1"] = oo >= 1e-8
        anchor: Float64[ndarray, "m 3"] = decoded.anchor[rows]
        a: Float64[ndarray, "m 3"] = np.where(ok, anchor - (anchor * om).sum(1, keepdims=True) / np.where(ok, oo, 1) * om, anchor)
        w: Float64[ndarray, "m 3"] = om * dt[rows, None]
        v: Float64[ndarray, "m 3"] = nu * dt[rows, None]
        th: Float64[ndarray, "m 1"] = np.linalg.norm(w, axis=1, keepdims=True)
        small: Bool[ndarray, "m 1"] = th < 1e-6
        ths: Float64[ndarray, "m 1"] = np.where(small, 1, th)
        sh, ch = np.sin(0.5 * ths), np.cos(0.5 * ths)
        big_a: Float64[ndarray, "m 1"] = np.where(small, 1, 2 * sh * ch / ths)
        big_b: Float64[ndarray, "m 1"] = np.where(small, 0.5, 2 * sh * sh / ths**2)
        big_c: Float64[ndarray, "m 1"] = np.where(small, 1 / 6, (ths - 2 * sh * ch) / ths**3)
        r: Float64[ndarray, "m 3"] = decoded.means[rows] - a
        wr: Float64[ndarray, "m 3"] = np.cross(w, r)
        wv: Float64[ndarray, "m 3"] = np.cross(w, v)
        positions[rows] = r + big_a * wr + big_b * np.cross(w, wr) + a + v + big_b * wv + big_c * np.cross(w, wv)
        q_rel: Float64[ndarray, "m 4"] = np.concatenate([np.cos(0.5 * th), w * np.where(small, 0.5, sh / ths)], 1)
        q: Float64[ndarray, "m 4"] = _quat_mul(q_rel, decoded.quats_wxyz[rows])
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        rotations[rows] = q * np.where(q[:, :1] < 0, -1, 1)
    return AsogFrame(positions, rotations, weights)


def _quat_mul(a: Float64[ndarray, "m 4"], b: Float64[ndarray, "m 4"]) -> Float64[ndarray, "m 4"]:
    aw, ax, ay, az = a.T
    bw, bx, by, bz = b.T
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        1,
    )
