"""MAMMA release readers: per-camera calibration NPZs, sequence labels, and the Vicon lens.

Every sequence directory holds one calibration NPZ per camera (``gt/`` in the eval collections, ``meta/`` in the
markerless ones) beside a ``global.npz``. Calibration is in native pixels and world metres: ``cam_int`` is K,
``cam_ext`` is world-to-camera (OpenCV axes), and IOI cameras add ``vicon_radial_2``. See ``docs/mamma.md``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from jaxtyping import Bool, Float64
from numpy import ndarray
from serde import SerdeError, from_dict, serde

Gender: TypeAlias = Literal["neutral", "male", "female"]
Method: TypeAlias = Literal["mosh_v_template", "mosh_betas", "mamma_pred"]
Family: TypeAlias = Literal["eval", "markerless-dance", "markerless-multiple-people", "markerless-iphones"]

FAMILY_COLLECTIONS: dict[Family, tuple[str, ...]] = {
    "eval": ("mamma_eval_singles", "mamma_eval_extra", "mamma_eval_dance"),
    "markerless-dance": ("mamma_markerless_dance",),
    "markerless-multiple-people": ("mamma_markerless_multiple_people",),
    "markerless-iphones": ("mamma_markerless_iphones",),
}
"""Release collections of each catalog family: the three MoSh++-labelled eval sets, and the markerless captures by rig."""
VIDEO_DIRS: tuple[str, ...] = ("videos_crf16", "videos")
"""Where a sequence keeps its source videos: the IOI rigs' CRF 16 clips, the iPhones' clips."""
EVAL_PREFIX: str = "mamma_eval_"

MILLIMETRE_THRESHOLD: float = 200.0
"""An extrinsic translation larger than this many units is in millimetres: no MAMMA camera stands 200 m away."""
LENS_ITERATIONS: int = 50
"""Fixed-point steps of the lens inversion; converged points stop moving after a few."""
LENS_TOLERANCE_PX: float = 1e-9
"""Largest step, in pixels, of a converged lens inversion."""


@serde
@dataclass(frozen=True, slots=True)
class ViconLens:
    """The release's ``vicon_radial_2``: a radial polynomial about ``centre`` that maps distorted to undistorted pixels.

    ``u = c + d (1 + k1 r^2 + k2 r^4 + k3 r^6)`` with ``d`` the distorted pixel's offset from ``c`` and ``r = |d|``, all in native
    pixels. Projection is therefore a pinhole followed by the inverse of this map.
    """

    centre: Float64[ndarray, "2"]
    """Distortion centre, native pixels."""
    radial: Float64[ndarray, "3"]
    """k1, k2, k3 per pixel^2, pixel^4, pixel^6."""

    def scale(self, offset: Float64[ndarray, "n 2"]) -> Float64[ndarray, "n 1"]:
        r2: Float64[ndarray, "n 1"] = (offset**2).sum(axis=-1, keepdims=True)
        return 1.0 + self.radial[0] * r2 + self.radial[1] * r2**2 + self.radial[2] * r2**3

    def undistort(self, distorted: Float64[ndarray, "n 2"]) -> Float64[ndarray, "n 2"]:
        """Distorted (image) pixels to undistorted (pinhole) pixels: the release's map."""
        offset: Float64[ndarray, "n 2"] = distorted - self.centre
        return self.centre + offset * self.scale(offset)

    def distort(self, undistorted: Float64[ndarray, "n 2"]) -> Float64[ndarray, "n 2"]:
        """Undistorted pixels to image pixels by fixed-point inversion; a point that does not converge is NaN."""
        target: Float64[ndarray, "n 2"] = undistorted - self.centre
        offset: Float64[ndarray, "n 2"] = target.copy()
        for _ in range(LENS_ITERATIONS):
            step: Float64[ndarray, "n 2"] = target / self.scale(offset) - offset
            offset = offset + step
            if not np.abs(step).max(initial=0.0) > LENS_TOLERANCE_PX:
                break
        converged: Bool[ndarray, "n"] = np.abs(self.centre + offset * self.scale(offset) - undistorted).max(axis=-1) <= 1e-6
        return np.where(converged[:, None], self.centre + offset, np.nan)


@serde
@dataclass(frozen=True, slots=True)
class MammaCamera:
    """One camera of one sequence, in the release's native pixel grid."""

    name: str
    """``IOI_XX`` or an iPhone's ``A001``."""
    K: Float64[ndarray, "3 3"]
    """Intrinsics, native pixels."""
    cam_T_world: Float64[ndarray, "4 4"]
    """World to camera (OpenCV axes), metres."""
    width: int
    """Native image width, pixels."""
    height: int
    """Native image height, pixels."""
    portrait: bool
    """The release's ``cam_portrait`` flag; the size already describes the upright image."""
    lens: ViconLens | None
    """The Vicon radial lens; None for the iPhones, which ship none."""

    def project(self, points: Float64[ndarray, "n 3"]) -> Float64[ndarray, "n 2"]:
        """World points to native image pixels through the pinhole and the lens; points behind the camera are NaN."""
        camera: Float64[ndarray, "n 3"] = points @ self.cam_T_world[:3, :3].T + self.cam_T_world[:3, 3]
        depth: Float64[ndarray, "n 1"] = camera[:, 2:]
        pinhole: Float64[ndarray, "n 2"] = (camera @ self.K.T)[:, :2] / np.where(depth > 0, depth, np.nan)
        return pinhole if self.lens is None else self.lens.distort(pinhole)


@serde
@dataclass(frozen=True, slots=True)
class PersonFit:
    """One person's SMPL-X parameters over the whole sequence, as the release ships them (float64; predictions upcast exactly)."""

    subject: str
    """The eval subject id, or the prediction file's stem (``params_00``)."""
    gender: Gender
    """Model the parameters belong to."""
    flat_hand_mean: bool
    """True when the hand poses are absolute; False when SMPL-X's mean hand is added to them."""
    pose: Float64[ndarray, "t 165"]
    """Full SMPL-X axis-angle pose: root, 21 body joints, jaw, left eye, right eye, 15 left-hand and 15 right-hand joints."""
    trans: Float64[ndarray, "t 3"]
    """Root translation in the world frame (smplx ``transl``), metres."""
    betas: Float64[ndarray, "b"]
    """Shape coefficients: 300 for the eval GT, 16 for the predictions."""
    v_template: Float64[ndarray, "10475 3"] | None
    """The subject's registered template mesh, which replaces the model's (``mosh_v_template``); None for beta-only fits."""


@serde
@dataclass(frozen=True, slots=True)
class SequenceLabels:
    """A sequence's body labels and the clock they share with its videos."""

    name: str
    """The release's ``seq_name``."""
    fps: int
    """Label and video rate, frames per second (whole in every MAMMA collection)."""
    frames: int
    """Frames in every camera's video and every person's labels (``frames_len``)."""
    method: Method
    """How the labels were made: MoSh++ on a scanned template or on fitted betas (eval GT), or the MAMMA model's prediction."""
    people: list[PersonFit]
    """One fit per person, in the release's order."""

    def __post_init__(self) -> None:
        for person in self.people:
            if len(person.pose) != self.frames or len(person.trans) != self.frames:
                raise ValueError(f"{person.subject}: {len(person.pose)} poses and {len(person.trans)} translations for {self.frames} frames")
            if (self.method == "mosh_v_template") != (person.v_template is not None):
                raise ValueError(f"{person.subject}: a {self.method} fit {'needs' if person.v_template is None else 'has no use for'} a template")


def decode_labels(path: Path, values: dict[str, object]) -> SequenceLabels:
    try:
        return from_dict(SequenceLabels, values)
    except (SerdeError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from error


def read_eval_labels(path: Path) -> SequenceLabels:
    """Decode an eval ``gt/global.npz``: MoSh++ fits of every person, in ``subject_name`` order (not the sequence name's)."""
    with np.load(path, allow_pickle=False) as npz:
        templates: Float64[ndarray, "..."] | None = npz["v_template"] if "v_template" in npz.files else None
        people: list[dict[str, object]] = [
            {
                "subject": str(subject),
                "gender": str(npz["gender"][index]),
                "flat_hand_mean": bool(npz["flat_hand_mean"]),
                "pose": npz["pose_world"][:, index].astype(np.float64),
                "trans": npz["pose_trans_world"][:, index].astype(np.float64),
                "betas": npz["shape"][index].astype(np.float64),
                "v_template": None if templates is None else templates[index].astype(np.float64),
            }
            for index, subject in enumerate(npz["subject_name"])
        ]
        values: dict[str, object] = {
            "name": str(npz["seq_name"]),
            "fps": int(npz["fps"]),
            "frames": int(npz["frames_len"]),
            "method": str(npz["method_used"]),
            "people": people,
        }
    return decode_labels(path, values)


def read_pred_labels(global_path: Path, pred_paths: list[Path]) -> SequenceLabels:
    """Decode a markerless ``meta/global.npz`` and its ``pred/params_XX.npz`` files, one person each, in the given order."""
    with np.load(global_path, allow_pickle=False) as npz:
        fps: int = int(npz["fps"])
        values: dict[str, object] = {"name": str(npz["seq_name"]), "fps": fps, "frames": int(npz["frames_len"]), "method": str(npz["method_used"])}
    people: list[dict[str, object]] = []
    for path in pred_paths:
        with np.load(path, allow_pickle=False) as npz:
            if float(npz["mocap_frame_rate"]) != fps:
                raise ValueError(f"{path}: labels at {float(npz['mocap_frame_rate'])} fps in a {fps} fps sequence")
            people.append(
                {
                    "subject": path.stem,
                    "gender": str(npz["gender"]),
                    "flat_hand_mean": bool(npz["flat_hand_mean"]),
                    "pose": npz["poses"].astype(np.float64),
                    "trans": npz["trans"].astype(np.float64),
                    "betas": npz["betas"].astype(np.float64),
                    "v_template": None,
                }
            )
    return decode_labels(global_path, {**values, "people": people})


def read_camera(path: Path) -> MammaCamera:
    """Decode one calibration NPZ; per-frame arrays (``is_body_in_img``, ``visibility``, image paths) are not read."""
    with np.load(path, allow_pickle=False) as npz:
        cam_T_world: Float64[ndarray, "..."] = npz["cam_ext"].astype(np.float64)
        if cam_T_world.shape == (4, 4) and np.abs(cam_T_world[:3, 3]).max() > MILLIMETRE_THRESHOLD:
            cam_T_world[:3, 3] /= 1000.0
        lens: Float64[ndarray, "..."] | None = npz["vicon_radial_2"] if "vicon_radial_2" in npz.files else None
        values: dict[str, object] = {
            "name": str(npz["cam_name"]),
            "K": npz["cam_int"].astype(np.float64),
            "cam_T_world": cam_T_world,
            "width": int(npz["cam_img_w"]),
            "height": int(npz["cam_img_h"]),
            "portrait": bool(npz["cam_portrait"]),
            "lens": None if lens is None else {"centre": lens[:2], "radial": lens[2:]},
        }
    try:
        return from_dict(MammaCamera, values)
    except (SerdeError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from error


@dataclass(frozen=True, slots=True)
class MammaSequence:
    """One release sequence directory and the files of it that dataforge reads."""

    collection: str
    """Release collection (``mamma_eval_singles``, ``mamma_markerless_iphones`` ...)."""
    directory: Path
    """The sequence directory."""
    labels_dir: Path
    """``gt/`` (eval) or ``meta/`` (markerless): ``global.npz`` and one calibration NPZ per camera."""
    video_dir: Path
    cameras: tuple[str, ...]
    """Cameras with both a calibration and a video, in name order."""
    predictions: tuple[Path, ...]
    """Markerless ``pred/params_XX.npz``, one per person in name order; empty for eval."""

    @property
    def parts(self) -> tuple[str, ...]:
        """Identity parts: the path under the collection, after the eval set's short name (``singles``) for eval."""
        relative: tuple[str, ...] = self.directory.relative_to(self.directory.parents[self.depth]).parts
        return (self.collection.removeprefix(EVAL_PREFIX), *relative) if self.collection.startswith(EVAL_PREFIX) else relative

    @property
    def depth(self) -> int:
        """Directories between the collection and the sequence: 1, or 2 for the iPhones' ``indoors/<scene>``."""
        return next(index for index, parent in enumerate(self.directory.parents) if parent.name == self.collection)

    def labels(self) -> SequenceLabels:
        if self.predictions:
            return read_pred_labels(self.labels_dir / "global.npz", list(self.predictions))
        return read_eval_labels(self.labels_dir / "global.npz")

    def calibration(self, camera: str) -> Path:
        return self.labels_dir / f"{camera}.npz"

    def video(self, camera: str) -> Path:
        return self.video_dir / f"{camera}.mp4"

    def mask_tar(self, camera: str) -> Path:
        """Eval only: the camera's per-person mask PNGs."""
        return self.directory / "masks" / f"{camera}_masks.tar"

    @property
    def markers_dir(self) -> Path:
        """Eval Extra only: held-out Vicon markers and the MoSh++ baseline's."""
        return self.directory / "markers"


def discover_sequences(root: Path, family: Family) -> tuple[list[MammaSequence], list[str]]:
    """Every convertible sequence of ``family`` under the release ``root``, and why each other sequence directory was skipped.

    A sequence needs a ``global.npz``, a video directory, at least one camera with both a calibration and a video, and,
    when markerless, prediction files (the bachata_improv_2p_32cam variant tree has CRF 24 clips and no predictions).
    """
    found: list[MammaSequence] = []
    skipped: list[str] = []
    for collection in FAMILY_COLLECTIONS[family]:
        base: Path = root / collection
        for global_path in sorted([*base.glob("*/gt/global.npz"), *base.glob("*/meta/global.npz"), *base.glob("*/*/meta/global.npz")]):
            labels_dir: Path = global_path.parent
            directory: Path = labels_dir.parent
            video_dir: Path | None = next((directory / name for name in VIDEO_DIRS if (directory / name).is_dir()), None)
            predictions: tuple[Path, ...] = tuple(sorted((directory / "pred").glob("params_*.npz")))
            if video_dir is None:
                skipped.append(f"{directory}: no {' or '.join(VIDEO_DIRS)} directory")
                continue
            if labels_dir.name == "meta" and not predictions:
                skipped.append(f"{directory}: no pred/params_*.npz")
                continue
            calibrated: set[str] = {path.stem for path in labels_dir.glob("*.npz")} - {"global"}
            cameras: tuple[str, ...] = tuple(sorted(calibrated & {path.stem for path in video_dir.glob("*.mp4")}))
            if not cameras:
                skipped.append(f"{directory}: no camera has both a calibration and a video")
                continue
            if set(cameras) != calibrated:
                skipped.append(f"{directory}: cameras without a video left out: {sorted(calibrated - set(cameras))}")
            found.append(MammaSequence(collection, directory, labels_dir, video_dir, cameras, predictions))
    return found, skipped
