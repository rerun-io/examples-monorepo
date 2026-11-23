"""Animated SOG (``.asog``, recipes/asog-src) for trained TRiGS clips: checkpoint parameters -> ``.asog`` and back.

``.asog`` is a third-party interchange format (arrival-space/animated-sog). Our artifact is the ``.rrd``: the ASOG bytes
go into the recording as the gsplat viewer's ``AnimatedSogSplats3D`` payload (``gsplat_rust_renderer.sog3d.log_asog``),
so ``encode_asog`` returns bytes and ``decode_asog`` reads bytes; upstream's encoder and reader only take paths, so both
go through a temporary file.

Encode: the checkpoint's parameters become the encoder's in-memory input (``asog.trigs.TrigsRuntime``, what upstream
reads from a ``.trigsbin``) and go through ``asog.encode.encode_runtime`` (recipe hook 0001), so no ``.trigsbin`` is
written. One clip covers the whole run, no fades.

Decode: ``decode_asog`` dequantises the file (``asog.read``) and maps it back onto TRiGS parameters, so the trainer's own
renderer (``TrigsTrainer._render``) draws it. ``.asog`` stores the rotation twist control points ``O_k`` (``zeta_base_r``
folded in) and, for the translation, only ``g(tau) = nu(tau) . omega(tau)`` as a quartic (``G_0..G_4``): TRiGS's gauge fix
keeps only the component of ``nu`` along ``omega``, ``nu_par = (nu . omega / |omega|^2) omega``, so ``g`` is all that
moves a splat. The way back picks the minimum-norm translation control points ``N_k`` whose product with the ``O_k`` is
that quartic (``nu_control_points``); the TRiGS motion of the result equals the ``.asog`` reference decoder's motion.

Axis anchor: the encoder quantises ``O_k`` with the step ``qp / |mean - anchor|`` (a rotation that moves the splat by
``qp`` mm). A translation-dominant row has a small ``omega``, so its ``O_k`` round to a few steps or to zero, and the
decoded translation ``g / |omega|^2 omega`` follows that coarse axis or vanishes (run C, band-safe: 13 % of the rows
under 5 steps, 30-60 mm median position error; 1 % rounded to zero). ``axis_anchors`` moves the anchor of such rows onto
their own mean rotation axis, far out: the rotation then barely moves the splat, and the long lever makes the step fine.
It drops the row's small rotation about its old axis (under ``AXIS_ANCHOR_STEPS`` x qp) and costs nothing in bytes;
measured on held-out crop_raw PSNR it cuts the band-safe loss from -0.42 % to -0.09 % (run C) and -0.47 % to -0.09 % (MAMMA).
A row with ``omega = 0`` throughout (TRiGS: pure translation by the full ``nu``) has no axis and still loses its
translation; none of our runs has one.
"""

import dataclasses
import math
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from asog import Asog
from asog import read as read_asog
from asog.encode import MID_O, R_MIN, SM, EncodeStats, encode_runtime
from asog.trigs import TrigsRuntime
from jaxtyping import Float64
from numpy import ndarray
from torch import Tensor

OPACITY_EPS: float = 1e-6
"""Decoded opacities (q / 255) are clamped to [eps, 1 - eps] before the logit."""
AXIS_ANCHOR_STEPS: float = 8.0
"""Rows whose twist control points span fewer omega steps than this over the visible life get the axis anchor."""
AXIS_ANCHOR_TARGET: float = 16.0
"""Omega steps the axis anchor gives such a row (more: finer axis, but a larger swing when the axis turns)."""
AXIS_ANCHOR_MAX_M: float = 50.0
"""Farthest axis anchor from the splat, file units."""


@dataclass(frozen=True, slots=True)
class AsogOptions:
    """Encoder options, all set (what ``encode_runtime`` takes)."""

    qp: float
    """Motion step: displacement over a splat's visible life, file units x 1e-3 (mm for metres)."""
    qa: float
    """Anchor step, file units x 1e-3."""
    prune: float
    """Fraction of the dynamic splats dropped, least important first (opacity x area x sigma)."""
    mbits: int
    """Position bits per axis (of 16, log-mapped)."""
    tbits: int
    """Centre-time resolution 2^-tbits s."""
    lsbits: int
    """Log-sigma bits."""


AsogPreset = Literal["band-safe", "small", "upstream"]

_UPSTREAM: AsogOptions = AsogOptions(qp=2.0, qa=8.0, prune=0.3, mbits=14, tbits=11, lsbits=8)
PRESETS: dict[AsogPreset, AsogOptions] = {
    "band-safe": AsogOptions(qp=0.5, qa=2.0, prune=0.0, mbits=16, tbits=11, lsbits=8),
    "small": _UPSTREAM,
    "upstream": _UPSTREAM,
}
"""``band-safe`` (our default) keeps held-out PSNR within 1 % of the uncompressed clip on 4DAnyone and MAMMA at
about 12 MB (11.9 MB, -0.09 % on 4DAnyone run C); ``small`` = arrival-space/animated-sog's encoder defaults (alias
``upstream``), a smaller file for web streaming (7.3 MB, -0.39 % on run C with the axis anchor)."""


@dataclass(frozen=True, slots=True)
class AsogSettings:
    """A preset, with any option overridden (``None`` = the preset's value)."""

    preset: AsogPreset = "band-safe"
    """``band-safe``: our default (held-out PSNR within 1 % of the uncompressed clip); ``small``: smaller file for web streaming (``upstream`` is an alias)."""
    qp: float | None = None
    """Motion step: displacement over a splat's visible life, file units x 1e-3 (mm for metres)."""
    qa: float | None = None
    """Anchor step, file units x 1e-3."""
    prune: float | None = None
    """Fraction of the dynamic splats dropped, least important first (opacity x area x sigma)."""
    mbits: int | None = None
    """Position bits per axis (of 16, log-mapped)."""
    tbits: int | None = None
    """Centre-time resolution 2^-tbits s."""
    lsbits: int | None = None
    """Log-sigma bits."""

    def options(self) -> AsogOptions:
        """The preset's options with the overrides applied."""
        overrides = {name: getattr(self, name) for name in ("qp", "qa", "prune", "mbits", "tbits", "lsbits") if getattr(self, name) is not None}
        return dataclasses.replace(PRESETS[self.preset], **overrides)


def clip_manifest(fps: float, num_frames: int) -> dict[str, object]:
    """The ``trigs.runtime`` manifest fields the encoder copies: one clip over the whole run, no fades."""
    duration: float = num_frames / fps
    return {
        "format": "trigs.runtime",
        "version": 1,
        "timeline": {"frameRate": float(fps), "totalFrames": int(num_frames), "durationSeconds": duration},
        "clips": [
            {
                "id": "clip_0000",
                "index": 0,
                "global": {
                    "frameStart": 0,
                    "frameEndExclusive": int(num_frames),
                    "activeTimeStart": 0.0,
                    "activeTimeEnd": duration,
                    "coreTimeStart": 0.0,
                    "coreTimeEnd": duration,
                },
                "local": {"frameStart": 0, "frameEndExclusive": int(num_frames), "timeStart": 0.0, "timeEnd": duration},
                "render": {"fadeInFrames": 0, "fadeOutFrames": 0, "fadeCurve": "smoothstep"},
            }
        ],
    }


def trigs_runtime(params: Mapping[str, Tensor], fps: float, num_frames: int) -> TrigsRuntime:
    """The encoder input from TRiGS parameters (quaternions wxyz, opacity after the sigmoid, ``O_k = zeta_base + zeta_ck``)."""

    def arr(name: str) -> ndarray:
        return params[name].detach().to("cpu", torch.float64).numpy()

    n: int = int(params["means"].shape[0])
    static: ndarray = arr("is_static").reshape(n) > 0.5 if "is_static" in params else np.zeros(n, dtype=bool)
    quats: ndarray = arr("quats")
    quats = quats / np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)
    keep_motion: ndarray = (~static)[:, None]
    omega: ndarray = np.stack([np.where(keep_motion, arr("zeta_base_r") + arr(f"zeta_c{k}_r"), 0.0) for k in range(3)])
    nu: ndarray = np.stack([np.where(keep_motion, arr("zeta_base_t") + arr(f"zeta_c{k}_t"), 0.0) for k in range(3)])
    return TrigsRuntime(
        manifest=clip_manifest(fps, num_frames),
        means=arr("means"),
        log_scales=arr("scales"),
        quats=quats,
        opacity=1.0 / (1.0 + np.exp(-arr("opacities").reshape(n))),
        sh0=arr("sh0").reshape(n, 3),
        shN=params["shN"].detach().to("cpu", torch.float32).numpy() if "shN" in params else None,
        times=np.where(static, 0.0, arr("times").reshape(n)),
        log_durations=np.where(static, 0.0, arr("log_durations").reshape(n)),
        anchors=np.where(keep_motion, arr("anchors"), 0.0),
        omega=omega,
        nu=nu,
        static=static,
        clip=np.zeros(n, dtype=np.int64),
    )


def axis_anchors(runtime: TrigsRuntime, qp: float) -> Float64[ndarray, "n 3"]:
    """Anchors for the encoder: translation-dominant rows' anchors moved onto their mean rotation axis (module docstring)."""
    qpv: float = qp / 1000
    life: ndarray = SM * np.exp(runtime.log_durations)
    lever: ndarray = np.maximum(np.linalg.norm(runtime.means - runtime.anchors, axis=1), R_MIN)
    mid: ndarray = np.array([1.0, MID_O, 1.0])[:, None]
    steps: ndarray = (np.abs(runtime.omega).max(axis=2) * life * lever / qpv / mid).max(axis=0)
    axis: ndarray = runtime.omega.sum(axis=0)
    axis_norm: ndarray = np.linalg.norm(axis, axis=1)
    distance: ndarray = np.clip(AXIS_ANCHOR_TARGET * qpv / np.maximum(np.abs(runtime.omega).max(axis=(0, 2)) * life, 1e-12), R_MIN, AXIS_ANCHOR_MAX_M)
    move: ndarray = ~runtime.static & (steps < AXIS_ANCHOR_STEPS) & (axis_norm > 0)
    anchors: ndarray = runtime.anchors.copy()
    anchors[move] = runtime.means[move] + distance[move, None] * axis[move] / axis_norm[move, None]
    return anchors


def encode_asog(params: Mapping[str, Tensor], fps: float, num_frames: int, settings: AsogSettings) -> tuple[bytes, EncodeStats]:
    """The clip as ``.asog`` bytes and the encoder's statistics."""
    options: AsogOptions = settings.options()
    runtime: TrigsRuntime = trigs_runtime(params, fps, num_frames)
    runtime.anchors = axis_anchors(runtime, options.qp)
    with tempfile.TemporaryDirectory(prefix="open4dgs-asog-") as scratch:
        dst: Path = Path(scratch) / "clip.asog"
        stats: EncodeStats = encode_runtime(
            runtime,
            dst,
            qp=options.qp,
            qa=options.qa,
            prune=options.prune,
            mbits=options.mbits,
            tbits=options.tbits,
            lsbits=options.lsbits,
        )
        return dst.read_bytes(), stats


def nu_control_points(omega: Float64[ndarray, "3 n 3"], g: Float64[ndarray, "5 n"]) -> Float64[ndarray, "3 n 3"]:
    """Minimum-norm quadratic translation control points ``N_k`` with ``(sum b_i N_i) . (sum b_j O_j) = sum B_k G_k``.

    The product of two quadratic Bernstein curves has the quartic Bernstein coefficients
    ``G_k = sum_{i+j=k} C(2,i) C(2,j) / C(4,k) N_i . O_j``: five linear equations in the nine values of ``N``.
    """
    n: int = omega.shape[1]
    o0, o1, o2 = omega
    zero: ndarray = np.zeros_like(o0)
    rows: list[list[ndarray]] = [
        [o0, zero, zero],
        [o1 / 2, o0 / 2, zero],
        [o2 / 6, 4 * o1 / 6, o0 / 6],
        [zero, o2 / 2, o1 / 2],
        [zero, zero, o2],
    ]
    system: ndarray = np.stack([np.concatenate(row, axis=1) for row in rows], axis=1)  # (n, 5, 9)
    solution: ndarray = (np.linalg.pinv(system) @ g.T[:, :, None])[:, :, 0]  # (n, 9)
    return solution.reshape(n, 3, 3).transpose(1, 0, 2)


def trigs_params(decoded: Asog, device: torch.device) -> dict[str, Tensor]:
    """TRiGS parameters whose motion is the decoded clip's (``zeta_base_*`` = 0, ``zeta_ck_r = O_k``, ``zeta_ck_t = N_k``)."""
    n: int = decoded.count
    nu: ndarray = nu_control_points(decoded.omega, decoded.g)
    opacity: ndarray = np.clip(decoded.opacity, OPACITY_EPS, 1.0 - OPACITY_EPS)

    def t(a: ndarray) -> Tensor:
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)

    params: dict[str, Tensor] = {
        "means": t(decoded.means),
        "quats": t(decoded.quats),
        "scales": t(decoded.log_scales),
        "opacities": t(np.log(opacity / (1.0 - opacity))),
        "sh0": t(decoded.sh0),
        "times": t(decoded.mu[:, None]),
        "log_durations": t(np.log(decoded.sigma)[:, None]),
        "anchors": t(decoded.anchor),
        "is_static": t(decoded.static[:, None].astype(np.float32)),
        "zeta_base_r": t(np.zeros((n, 3))),
        "zeta_base_t": t(np.zeros((n, 3))),
    }
    if decoded.shN is not None:
        params["shN"] = t(decoded.shN)
    for k in range(3):
        params[f"zeta_c{k}_r"] = t(decoded.omega[k])
        params[f"zeta_c{k}_t"] = t(nu[k])
    return params


def decode_asog(data: bytes, device: torch.device) -> tuple[dict[str, Tensor], float]:
    """Decode ``.asog`` bytes and return (TRiGS parameters on ``device``, decode seconds incl. the mapping)."""
    t0: float = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="open4dgs-asog-") as scratch:
        path: Path = Path(scratch) / "clip.asog"
        path.write_bytes(data)
        decoded: Asog = read_asog(path)
    params: dict[str, Tensor] = trigs_params(decoded, device)
    return params, time.perf_counter() - t0


def tiny_omega_fraction(params: Mapping[str, Tensor], samples: int = 9) -> float:
    """Fraction of dynamic rows whose ``|omega(tau)|^2`` drops below TRiGS's gauge threshold 1e-8 somewhere in the life.

    There TRiGS moves the splat with the full ``nu``; ``.asog`` keeps only ``nu . omega``, so such a splat loses its
    translation.
    """
    o: list[Tensor] = [(params["zeta_base_r"] + params[f"zeta_c{k}_r"]).double() for k in range(3)]
    worst: Tensor = torch.full((o[0].shape[0],), math.inf, dtype=torch.float64, device=o[0].device)
    for s in range(samples):
        tau: float = s / (samples - 1)
        w: Tensor = (1 - tau) ** 2 * o[0] + 2 * (1 - tau) * tau * o[1] + tau**2 * o[2]
        worst = torch.minimum(worst, (w * w).sum(dim=-1))
    dynamic: Tensor = params["is_static"].reshape(-1) < 0.5 if "is_static" in params else torch.ones_like(worst, dtype=torch.bool)
    return float((worst[dynamic] < 1e-8).double().mean())
