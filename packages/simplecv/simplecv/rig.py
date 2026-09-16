"""Generic, system-agnostic rig schema (COLMAP-style), owned by ``simplecv``.

A **rig** is a set of sensors with fixed relative poses; one **reference
sensor** defines the rig origin (its ``rig_T_sensor`` is identity) and every
other sensor is a fixed offset from it. When the rig moves, its pose
(``world_T_rig``) is driven over time on the rig node and **every sensor moves
rigidly with it** — this is exactly the COLMAP rig/sensor model
(<https://colmap.github.io/concepts.html#rigs>).

The entity layout matches ``packages/slam-evals/docs/schema.md`` and
``packages/live-rerun/docs/rig_schema.md`` (``/world/rig_<NN>/cam_<NN>``), so a
recording drops into the same mental model regardless of which system produced
it. ``live_rerun.rig`` re-exports these types for backward compatibility.

This module owns two layers:

- :class:`CameraSensor` / :class:`RigCalibration` — the generic *static*
  descriptor of a rig's cameras and which one is the reference. Reused verbatim
  by ``live-rerun`` (a single stationary OAK rig).
- :class:`Rig` / :class:`RigPoseStream` — the exoego-level wrapper that adds an
  entity ``index`` (``rig_<NN>``) and *optional motion*. A static
  world-anchored rig (exo) leaves ``pose_stream`` ``None`` (implicit identity);
  a moving rig (a worn ego device) carries ``world_T_rig(t)`` plus a validity
  mask so tracking dropouts hide the whole rig at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
from jaxtyping import Bool, Float
from numpy import ndarray

from simplecv.camera_parameters import Extrinsics, Fisheye62Parameters, Intrinsics, PinholeParameters

#: Reserved sensor kinds describing image *content* (not projection model — a
#: fisheye RGB camera is still ``"rgb"``). Only ``rgb``/``grayscale`` cameras are
#: emitted by this package; ``depth``/``imu``/``mag`` name the peer sensors that slot
#: in beside the cameras (e.g. ``/world/rig_00/imu_00``, ``/world/rig_00/mag_00``)
#: without a vocabulary change — dataforge writes the latter two.
SensorKind = Literal["rgb", "grayscale", "depth", "imu", "mag"]

#: Zero-padded width for rig/sensor entity indices (``rig_00``, ``cam_00``,
#: ``imu_00``). Two digits so ids sort lexicographically for any realistic rig
#: (``cam_02`` before ``cam_10``); a single digit would sort ``cam_10`` first.
INDEX_WIDTH: int = 2


def entity_id(prefix: str, index: int) -> str:
    """Format a zero-padded rig/sensor entity id, e.g. ``cam_03`` / ``rig_00``."""
    return f"{prefix}_{index:0{INDEX_WIDTH}d}"


@dataclass
class CameraSensor:
    """One camera in a rig.

    ``pinhole`` carries intrinsics plus extrinsics expressed **relative to the
    rig frame** — i.e. ``rig_T_cam`` — so the reference sensor's extrinsics are
    identity. The type is ``PinholeParameters | Fisheye62Parameters`` because
    ego rigs (Aria, HOT3D, UmeTrack) use Kannala–Brandt fisheye sensors;
    ``log_pinhole`` / ``PinholeWithDistortion.from_camera`` already accept both.
    ``index`` selects the entity path ``cam_<NN>`` (zero-padded via
    :func:`entity_id`); ``name`` is the human role label (e.g. ``"left"``)
    logged as metadata; ``kind`` is the image content (``"rgb"`` / ``"grayscale"``).
    """

    index: int
    name: str
    kind: SensorKind
    pinhole: PinholeParameters | Fisheye62Parameters


@dataclass
class RigCalibration:
    """A rig's cameras plus which one is the reference (identity) sensor.

    ``reference_index`` is the ``index`` of the camera whose ``rig_T_cam`` is
    identity (the rig origin); in a multi-camera rig it gets a distinct frustum
    colour in the viewer and is recorded in the rig's schema metadata.
    """

    cameras: list[CameraSensor] = field(default_factory=list)
    reference_index: int = 0


@dataclass
class RigPoseStream:
    """Per-frame ``world_T_rig`` for a MOVING rig, with a validity hint.

    ``world_t_rig`` / ``world_R_rig`` are the rig's pose in world (the
    ``parent(world)->child(rig)`` step, logged on the rig node *without*
    ``from_parent``), mirroring how the v1 ego writer logged ``world_T_cam`` on
    the camera node — here it rides on the rig node so all child cameras move
    rigidly with it. The per-frame timestamps are supplied at log time by
    :func:`simplecv.rerun_rig_logger.log_rig_pose_stream` (they come from the
    demuxed video, not the builder).

    Invalid frames (e.g. a HoloLens tracking dropout) carry **NaN** pose values,
    which render no frustum, so the whole rig disappears during a gap. The derived
    :attr:`valid` property exposes that per-frame mask; it is computed from the pose
    (the single source of truth), not stored, so it cannot drift out of sync.
    """

    world_t_rig: Float[ndarray, "n_frames 3"]
    world_R_rig: Float[ndarray, "n_frames 3 3"]

    @property
    def valid(self) -> Bool[ndarray, "n_frames"]:
        """Per-frame validity, derived from the pose: a finite pose is valid, a NaN
        pose is a dropout (rig + all child frusta hidden for that frame)."""
        return np.isfinite(self.world_t_rig).all(axis=1) & np.isfinite(self.world_R_rig).all(axis=(1, 2))


@dataclass
class Rig:
    """One rig instance for logging: a :class:`RigCalibration` plus optional motion.

    - **static world-anchored rig** (exo): ``pose_stream`` is ``None``; the rig
      frame coincides with world (``world_T_rig`` implicit identity, no transform
      on the rig node) and each camera's ``rig_T_cam`` equals its ``world_T_cam``.
    - **moving rig** (a worn ego device): ``pose_stream`` carries ``world_T_rig(t)``
      (the reference camera's world trajectory); the reference camera has identity
      ``rig_T_cam`` and the others are fixed offsets from it.

    ``index`` selects the ``rig_<NN>`` entity path. ``image_plane_distance`` sizes
    the logged frusta (exo and ego historically differ).
    """

    index: int
    calibration: RigCalibration
    pose_stream: RigPoseStream | None = None
    image_plane_distance: float = 0.5


# ── Rig-construction helpers (pure; used by the exoego rig builders) ──

#: Substrings that mark a stream as monochrome/SLAM content (best-effort hint for
#: the ``kind`` metadata only — it is never load-bearing for geometry).
_GRAYSCALE_NAME_HINTS: tuple[str, ...] = ("slam", "mono", "gray", "grey")


def kind_for_name(name: str) -> SensorKind:
    """Best-effort image-content kind from a stream name (``rgb`` unless it looks
    like a SLAM/mono sensor). Metadata only — does not affect projection."""
    lowered: str = name.lower()
    if any(hint in lowered for hint in _GRAYSCALE_NAME_HINTS):
        return "grayscale"
    return "rgb"


def reference_index_for_names(names: list[str]) -> int:
    """Pick the rig's reference camera: the first RGB-named stream, else index 0.

    Honours the convention "reference = the RGB camera where present, else the
    first sensor". Any choice is geometrically valid (the rig math holds for any
    reference); this only sets which camera is the identity origin / green tint.
    """
    for index, name in enumerate(names):
        if "rgb" in name.lower():
            return index
    return 0


def identity_extrinsics() -> Extrinsics:
    """``Extrinsics`` whose ``cam_T_world`` (and ``world_T_cam``) are identity —
    i.e. ``rig_T_cam = I`` for a rig's reference camera."""
    return Extrinsics(cam_R_world=np.eye(3), cam_t_world=np.zeros(3))


def rebuild_camera_with_extrinsics(
    camera: PinholeParameters | Fisheye62Parameters,
    extrinsics: Extrinsics,
) -> PinholeParameters | Fisheye62Parameters:
    """Copy ``camera`` (intrinsics + distortion) but swap in ``extrinsics``.

    Used to express a camera relative to its rig frame (``rig_T_cam``): the
    intrinsics/distortion are unchanged, only the pose is replaced. ``replace``
    preserves the concrete type (a fisheye stays a :class:`Fisheye62Parameters`)
    and re-runs ``__post_init__`` so the derived ``projection_matrix`` is rebuilt.
    """
    return replace(camera, extrinsics=extrinsics)


def stereo_rig_calibration(K_33: Float[ndarray, "3 3"], baseline_m: float, width: int, height: int) -> RigCalibration:
    """Calibration for a rectified stereo pair: ``cam_00`` = left (reference, identity), ``cam_01`` = right at ``+baseline_m`` along x.

    Args:
        K_33: Shared pinhole intrinsics of the rectified pair, ``Float[ndarray, "3 3"]``.
        baseline_m: Distance between the optical centres in metres.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        A two-camera :class:`RigCalibration` with the left camera as reference.
    """
    cameras: list[CameraSensor] = []
    for index, name, t_x in ((0, "left", 0.0), (1, "right", baseline_m)):
        intrinsics: Intrinsics = Intrinsics(camera_conventions="RDF", height=height, width=width, k_matrix=K_33)
        extrinsics: Extrinsics = Extrinsics(world_R_cam=np.eye(3), world_t_cam=np.array([t_x, 0.0, 0.0]))
        cameras.append(CameraSensor(index=index, name=name, kind="rgb", pinhole=PinholeParameters(name=name, extrinsics=extrinsics, intrinsics=intrinsics)))
    return RigCalibration(cameras=cameras, reference_index=0)
