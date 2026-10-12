"""MAMMA release readers: per-camera calibration NPZs, sequence labels, and the Vicon lens.

Every sequence directory holds one calibration NPZ per camera (``gt/`` in the eval collections, ``meta/`` in the
markerless ones) beside a ``global.npz``. Calibration is in native pixels and world metres: ``cam_int`` is K,
``cam_ext`` is world-to-camera (OpenCV axes), and IOI cameras add ``vicon_radial_2``. See ``docs/mamma.md``.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, TypeVar

import numpy as np
from jaxtyping import Bool, Float64
from numpy import ndarray
from serde import SerdeError, from_dict, serde

Gender: TypeAlias = Literal["neutral", "male", "female"]
SmplxModel: TypeAlias = Literal["locked_head_neutral", "v1_1_male", "v1_1_female", "v1_1_neutral"]
Method: TypeAlias = Literal["mosh_v_template", "mosh_betas", "mamma_pred", "mamma_render", "bedlam_render"]
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
MARKERS_COLLECTION: str = "mamma_eval_extra"
"""The one collection that ships Vicon markers."""
MARKER_FILES: tuple[str, str, str] = ("labels_m37.npy", "vicon_m37.npy", "baseline_m37.npy")
"""An Extra sequence's ``markers/``: the 37 held-out marker names, their Vicon positions, the MoSh++ baseline's."""
KNOWN_INCOMPLETE: frozenset[str] = frozenset({"bachata_improv_2p_32cam"})
"""Sequence directories the release ships incomplete on purpose: a markerless-dance variant with CRF 24 clips and no predictions."""
IPHONE_TWO_PEOPLE: frozenset[str] = frozenset(
    {
        "indoors/grappling",
        "indoors/holding_person_lifted",
        "indoors/hugging",
        "indoors/lean_on_shoulder_tap",
        "indoors/person_lift_from_ground",
        "outdoors/MANUAL_START_bumping_to_eachother",
        "outdoors/MANUAL_START_two_friends",
        "outdoors/greeting_interaction",
        "outdoors/passing_ball",
        "outdoors/pushing_and_lifting_from_ground",
        "outdoors/spinning_together",
        "outdoors/tapping_on_shoulder",
        "outdoors/wheelbarrow_walk",
    }
)
"""The iPhone scenes with two predicted people (the release manifest, ``packages/mamma``'s ``IPHONE_*_SEQUENCES``); the rest have one."""
GENDER_MODELS: dict[str, SmplxModel] = {"neutral": "locked_head_neutral", "male": "v1_1_male", "female": "v1_1_female"}
"""The model each gender's fits in the MPI collections use: every fit with betas is neutral (locked head); the gendered fits are
subject templates with zero betas, exact with v1.1 (``docs/mamma.md``)."""

MILLIMETRE_THRESHOLD: float = 200.0
"""An extrinsic translation larger than this many units is in millimetres: no MAMMA camera stands 200 m away."""
LENS_ITERATIONS: int = 50
"""Fixed-point steps of the lens inversion; converged points stop moving after a few."""
LENS_TOLERANCE_PX: float = 1e-9
"""Largest step, in pixels, of a converged lens inversion."""
LENS_RESIDUAL_PX: float = 1e-6
"""Largest error, in pixels, of an inverted point mapped back; a point that misses it did not converge."""

RecordT = TypeVar("RecordT")


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
            if not np.abs(step[np.isfinite(step)]).max(initial=0.0) > LENS_TOLERANCE_PX:  # NaN points (behind the camera) stop no one
                break
        distorted: Float64[ndarray, "n 2"] = self.centre + offset
        converged: Bool[ndarray, "n"] = np.abs(self.undistort(distorted) - undistorted).max(axis=-1) <= LENS_RESIDUAL_PX
        return np.where(converged[:, None], distorted, np.nan)


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
    """Native image height, pixels (portrait cameras ship upright)."""
    lens: ViconLens | None
    """The Vicon radial lens; None for the iPhones, which ship none."""

    @property
    def world_T_cam(self) -> Float64[ndarray, "4 4"]:
        return np.linalg.inv(self.cam_T_world)

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
    """The eval subject id, the prediction file's stem (``params_00``), or a MammaSyn person (``person_00``)."""
    index: int
    """The person's entity, ``person_NN``: the release's order (MPI) or its person index (MammaSyn: instance mask value - 1)."""
    gender: Gender
    """The person's gender as the labels state it."""
    model: SmplxModel
    """The SMPL-X model the parameters belong to (``mamma_body.MODEL_FILES``)."""
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
    present: Bool[ndarray, "t"]
    """Rows at which the person is there (MPI: every row; MammaSyn: rows a view labels them). Absent rows hold zeros."""


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
            if len(person.pose) != self.frames or len(person.trans) != self.frames or len(person.present) != self.frames:
                raise ValueError(f"{person.subject}: {len(person.pose)} poses and {len(person.trans)} translations for {self.frames} frames")
            if (self.method == "mosh_v_template") != (person.v_template is not None):
                raise ValueError(f"{person.subject}: a {self.method} fit {'needs' if person.v_template is None else 'has no use for'} a template")


def decode(cls: type[RecordT], source: Path | str, values: dict[str, object]) -> RecordT:  # noqa: UP047 — beartype requires legacy generics
    """One record from a file's values, naming the file (or shard member) on schema and ``__post_init__`` errors."""
    try:
        return from_dict(cls, values)
    except (SerdeError, ValueError) as error:
        raise ValueError(f"{source}: {error}") from error


def read_eval_labels(path: Path) -> SequenceLabels:
    """Decode an eval ``gt/global.npz``: MoSh++ fits of every person, in ``subject_name`` order (not the sequence name's)."""
    with np.load(path, allow_pickle=False) as npz:
        templates: Float64[ndarray, "..."] | None = npz["v_template"] if "v_template" in npz.files else None
        people: list[dict[str, object]] = [
            {
                "subject": str(subject),
                "index": index,
                "gender": str(npz["gender"][index]),
                "model": GENDER_MODELS.get(str(npz["gender"][index]), str(npz["gender"][index])),
                "flat_hand_mean": bool(npz["flat_hand_mean"]),
                "pose": npz["pose_world"][:, index].astype(np.float64),
                "trans": npz["pose_trans_world"][:, index].astype(np.float64),
                "betas": npz["shape"][index].astype(np.float64),
                "v_template": None if templates is None else templates[index].astype(np.float64),
                "present": np.ones(int(npz["frames_len"]), dtype=bool),
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
    return decode(SequenceLabels, path, values)


def read_pred_labels(global_path: Path, pred_paths: list[Path]) -> SequenceLabels:
    """Decode a markerless ``meta/global.npz`` and its ``pred/params_XX.npz`` files, one person each, in the given order."""
    with np.load(global_path, allow_pickle=False) as npz:
        fps: int = int(npz["fps"])
        frames: int = int(npz["frames_len"])
        values: dict[str, object] = {"name": str(npz["seq_name"]), "fps": fps, "frames": frames, "method": str(npz["method_used"])}
    people: list[dict[str, object]] = []
    for index, path in enumerate(pred_paths):
        with np.load(path, allow_pickle=False) as npz:
            if float(npz["mocap_frame_rate"]) != fps:
                raise ValueError(f"{path}: labels at {float(npz['mocap_frame_rate'])} fps in a {fps} fps sequence")
            people.append(
                {
                    "subject": path.stem,
                    "index": index,
                    "gender": str(npz["gender"]),
                    "model": GENDER_MODELS.get(str(npz["gender"]), str(npz["gender"])),
                    "flat_hand_mean": bool(npz["flat_hand_mean"]),
                    "pose": npz["poses"].astype(np.float64),
                    "trans": npz["trans"].astype(np.float64),
                    "betas": npz["betas"].astype(np.float64),
                    "v_template": None,
                    "present": np.ones(frames, dtype=bool),
                }
            )
    return decode(SequenceLabels, global_path, {**values, "people": people})


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
            "lens": None if lens is None else {"centre": lens[:2], "radial": lens[2:]},
        }
    return decode(MammaCamera, path, values)


def expected_people(collection: str, relative: str) -> int:
    """How many prediction files a markerless sequence ships, from the release's naming (checked against every sequence).

    A couple dance names both dancers' 5-digit ids, a multiple-people capture its head count (``MultiMama_3``); the iPhone
    scenes are listed.
    """
    name: str = Path(relative).name
    if collection == "mamma_markerless_dance":
        return 2 if re.search(r"_[0-9]{5}_[0-9]{5}(?:_[0-9]+){1,3}$", name) else 1
    if collection == "mamma_markerless_multiple_people":
        count: re.Match[str] | None = re.search(r"MultiMama_([0-9]+)_", name)
        if count is None:
            raise ValueError(f"{collection}/{relative}: no MultiMama_<people> in the name")
        return int(count.group(1))
    return 2 if relative in IPHONE_TWO_PEOPLE else 1


@dataclass(frozen=True, slots=True)
class MammaSequence:
    """One release sequence directory and the files of it that dataforge reads."""

    collection: str
    """Release collection (``mamma_eval_singles``, ``mamma_markerless_iphones`` ...)."""
    directory: Path
    """The sequence directory."""
    parts: tuple[str, ...]
    """Identity parts: the path under the collection, after the eval set's short name (``singles``) for eval."""
    labels_dir: Path
    """``gt/`` (eval) or ``meta/`` (markerless): ``global.npz`` and one calibration NPZ per camera."""
    video_dir: Path
    cameras: tuple[str, ...]
    """Cameras with both a calibration and a video, in name order."""
    predictions: tuple[Path, ...]
    """Markerless ``pred/params_XX.npz``, one per person in order; empty for eval."""

    @property
    def is_eval(self) -> bool:
        """MoSh++ ground truth with per-camera masks, rather than the MAMMA model's predictions."""
        return self.collection.startswith(EVAL_PREFIX)

    @property
    def global_npz(self) -> Path:
        return self.labels_dir / "global.npz"

    def labels(self) -> SequenceLabels:
        return read_eval_labels(self.global_npz) if self.is_eval else read_pred_labels(self.global_npz, list(self.predictions))

    def calibration(self, camera: str) -> Path:
        return self.labels_dir / f"{camera}.npz"

    def video(self, camera: str) -> Path:
        return self.video_dir / f"{camera}.mp4"

    def mask_tar(self, camera: str) -> Path:
        """Eval: the camera's per-person mask PNGs."""
        return self.directory / "masks" / f"{camera}_masks.tar"

    @property
    def markers_dir(self) -> Path:
        """Eval Extra: held-out Vicon markers and the MoSh++ baseline's (``MARKER_FILES``)."""
        return self.directory / "markers"

    @property
    def has_markers(self) -> bool:
        return self.collection == MARKERS_COLLECTION


def discover_sequences(root: Path, family: Family) -> tuple[list[MammaSequence], list[tuple[Path, str]]]:
    """Every convertible sequence of ``family`` under the release ``root``, and each other sequence directory with why it was skipped.

    A sequence needs a ``global.npz``, a video directory, at least one camera with both a calibration and a video, and, when
    markerless, exactly the prediction files its name implies (a missing one would renumber the people after it).
    """
    found: list[MammaSequence] = []
    skipped: list[tuple[Path, str]] = []
    for collection in FAMILY_COLLECTIONS[family]:
        base: Path = root / collection
        for global_path in sorted([*base.glob("*/gt/global.npz"), *base.glob("*/meta/global.npz"), *base.glob("*/*/meta/global.npz")]):
            labels_dir: Path = global_path.parent
            directory: Path = labels_dir.parent
            relative: tuple[str, ...] = directory.relative_to(base).parts
            video_dir: Path | None = next((directory / name for name in VIDEO_DIRS if (directory / name).is_dir()), None)
            if video_dir is None:
                skipped.append((directory, f"no {' or '.join(VIDEO_DIRS)} directory"))
                continue
            predictions: tuple[Path, ...] = ()
            if not collection.startswith(EVAL_PREFIX):
                predictions = tuple(sorted((directory / "pred").glob("params_*.npz")))
                expected: list[str] = [f"params_{index:02d}" for index in range(expected_people(collection, "/".join(relative)))]
                if [path.stem for path in predictions] != expected:
                    skipped.append((directory, f"predictions {[path.stem for path in predictions]}, the release ships {expected}"))
                    continue
            calibrated: set[str] = {path.stem for path in labels_dir.glob("*.npz")} - {"global"}
            cameras: tuple[str, ...] = tuple(sorted(calibrated & {path.stem for path in video_dir.glob("*.mp4")}))
            if not cameras:
                skipped.append((directory, "no camera has both a calibration and a video"))
                continue
            if set(cameras) != calibrated:
                skipped.append((directory, f"cameras without a video left out: {sorted(calibrated - set(cameras))}"))
            parts: tuple[str, ...] = (collection.removeprefix(EVAL_PREFIX), *relative) if collection.startswith(EVAL_PREFIX) else relative
            found.append(MammaSequence(collection, directory, parts, labels_dir, video_dir, cameras, predictions))
    return found, skipped
