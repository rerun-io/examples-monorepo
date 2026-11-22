"""An ``.asog`` clip as a Rerun payload: the custom gsplat viewer's ``AnimatedSogSplats3D`` contract plus a stock preview.

The contract is owned by packages/gsplat-rust-renderer (``gsplat_rust_renderer.sog3d``; the viewer decodes the file in
``gsplat_core::asog`` and evaluates the motion at the time cursor). This is a numpy-only copy of its writer so open4dgs
does not depend on that package; ``tests/test_asog_rrd.py`` pins it against a fixture written by the viewer's writer.
One static entity carries:

  AnimatedSogSplats3D:format            Utf8 x1    ``asog/<manifest version>``
  AnimatedSogSplats3D:asog              UInt8 xN   the ``.asog`` file verbatim
  AnimatedSogSplats3D:frame_origin      Float64 x1 timeline frame at clip time 0: t [s] = (frame - frame_origin) / frameRate
                                                   (frameRate from the manifest; there is no fps component)
  AnimatedSogSplats3D:active_threshold  Float32 x1 temporal cut the viewer applies (absent = 0.05, the format's constant)
  GaussianSplats3D                      the preview for stock viewers: ONE pose (the clip middle), rows with weight > 0,
                                        opacity x weight, every k-th of them; never the canonical rows (a ghost trail)

The preview pose comes from the asog reference decoder (recipes/asog-src) with the cut made a parameter: the reference
fixes it at 0.05, ``--asog-cut trained`` records the trainer's 0.01.
"""

import numpy as np
import pyarrow as pa
import rerun as rr
from asog import Asog, Frame
from asog.motion import bezier2, bezier4, life_tau, quat_mul, screw
from jaxtyping import Float32, Float64, UInt8
from numpy import ndarray

from open4dgs.splat_export import SH_C0

ARCHETYPE: str = "AnimatedSogSplats3D"
PREVIEW_ROWS: int = 16_384
"""Stock-viewer preview rows (the viewer's writer default)."""


def evaluate(clip: Asog, t: float, cut: float) -> Frame:
    """``Asog.evaluate(t)`` with the temporal cut ``cut`` instead of the reference's fixed 0.05 (same operations)."""
    weights, local_t = clip.clip_weights(t)
    dt: Float64[ndarray, " n"] = local_t - clip.mu
    gamma: Float64[ndarray, " n"] = np.exp(-0.5 * (dt / clip.sigma) ** 2)
    weights = np.where(clip.static, weights, weights * np.maximum((gamma - cut) / (1 - cut), 0))
    positions, rotations = clip.means.copy(), clip.quats.copy()
    m = np.nonzero(~clip.static & (weights > 0))[0]
    if len(m):
        tau = life_tau(dt[m], clip.sigma[m])
        om = bezier2(tau, clip.omega[:, m])
        oo = (om * om).sum(1, keepdims=True)
        nu = bezier4(tau, clip.g[:, m]).reshape(-1, 1) / np.maximum(oo, 1e-24) * om
        positions[m], qrel = screw(clip.means[m], clip.anchor[m], om, nu, dt[m])
        q = quat_mul(qrel, clip.quats[m])
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        rotations[m] = q * np.where(q[:, :1] < 0, -1, 1)
    return Frame(positions, rotations, weights)


def mid_clip_seconds(clip: Asog) -> float:
    """Clip time of the preview pose: half the duration (``durationSeconds``, else ``totalFrames / frameRate``)."""
    timeline: dict = clip.manifest["timeline"]
    duration = timeline.get("durationSeconds")
    return float(duration if duration is not None else timeline["totalFrames"] / timeline["frameRate"]) / 2


def preview(clip: Asog, frame: Frame, rows: int = PREVIEW_ROWS) -> rr.GaussianSplats3D | None:
    """One evaluated pose as stock ``GaussianSplats3D`` rows: weight > 0, opacity x weight, every k-th; None for rows <= 0."""
    if rows <= 0:
        return None
    alive: ndarray = np.nonzero(frame.weights > 0)[0]
    alive = alive[:: max(1, -(-len(alive) // rows))]
    rgba: Float32[ndarray, "m 4"] = np.clip(
        np.concatenate([0.5 + SH_C0 * clip.sh0[alive], (clip.opacity[alive] * frame.weights[alive])[:, None]], axis=1).astype(np.float32), 0.0, 1.0
    )
    colors: UInt8[ndarray, "m 4"] = (rgba * 255.0 + 0.5).astype(np.uint8)
    quats: Float32[ndarray, "m 4"] = frame.rotations[alive].astype(np.float32)
    return rr.GaussianSplats3D(
        centers=frame.positions[alive].astype(np.float32),
        quaternions=np.roll(quats, -1, axis=1),  # wxyz -> xyzw
        scales=np.exp(clip.log_scales[alive]).astype(np.float32),
        colors=colors,
    )


def payload(data: bytes, version: int, frame_origin: float, active_threshold: float) -> list[rr.DescribedComponentBatch]:
    """The ``AnimatedSogSplats3D`` component batches (untyped descriptors, as the viewer's writer logs them)."""
    if not 0.0 <= active_threshold < 1.0:
        raise ValueError(f"AnimatedSogSplats3D: active_threshold {active_threshold} is outside [0, 1)")
    arrays: dict[str, pa.Array] = {
        "format": pa.array([f"asog/{version}"], pa.utf8()),
        "asog": pa.array(np.frombuffer(data, np.uint8), pa.uint8()),
        "frame_origin": pa.array([frame_origin], pa.float64()),
        "active_threshold": pa.array([active_threshold], pa.float32()),
    }
    batches: list[rr.DescribedComponentBatch] = []
    for field, array in arrays.items():
        descriptor = rr.ComponentDescriptor(f"{ARCHETYPE}:{field}", archetype=ARCHETYPE, component_type=None)
        batches.append(rr.DescribedComponentBatch(rr.AnyBatchValue(descriptor, array), descriptor))
    return batches


def log_asog(
    recording: rr.RecordingStream,
    entity: str,
    data: bytes,
    clip: Asog,
    frame_origin: float,
    active_threshold: float,
    preview_rows: int = PREVIEW_ROWS,
) -> int:
    """Log the payload of ``data`` (decoded as ``clip``) and its mid-clip preview statically on ``entity``; returns the preview rows."""
    batches = payload(data, int(clip.manifest["version"]), frame_origin, active_threshold)
    rows = preview(clip, evaluate(clip, mid_clip_seconds(clip), active_threshold), preview_rows)
    if rows is None:
        recording.log(entity, batches, static=True)
        return 0
    recording.log(entity, rows, batches, static=True)
    return 0 if rows.centers is None else len(rows.centers)
