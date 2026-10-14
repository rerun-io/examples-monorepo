"""The EgoHumans capture layout, which Harmony4D reuses: one directory per sequence, read from a zip or from disk.

Both releases come from the same CMU pipeline (Khirodkar et al., EgoHumans ICCV 2023 and Harmony4D NeurIPS 2024) and lay a
sequence out the same way::

    exo/camNN/images/NNNNN.jpg                  static GoPros, 3840x2160 JPEG, one file per release frame (from 00001), 20 fps
    ego/ariaNN/calib/NNNNN.txt                  per frame: FISHEYE624 intrinsics and world->camera of rgb, left, right
    ego/ariaNN/images/{rgb,left,right}/NNNNN.jpg   Aria frames turned a quarter turn clockwise (upright)
    colmap/workplace/{cameras,images}.txt       exo lenses (OPENCV_FISHEYE) and poses in the COLMAP frame
    colmap/workplace/colmap_from_aria_transforms.pkl   each Aria's world -> COLMAP (with glasses), or scale.npy (exo only)
    processed_data/poses3d/NNNNN.npy            {person: COCO-17 x (x, y, z, confidence)}, world metres
    processed_data/smpl/NNNNN.npy               {person: SMPL global_orient, transl, body_pose, betas, vertices, joints}
    processed_data/bbox[es]/<camera>/<mode>/NNNNN.npy  {person: [x1, y1, x2, y2]} in the stored image

People are keyed ``aria01``, ``aria02``... after the glasses they wear, also in sequences captured without glasses. The world is
the anchor Aria's (``aria01``) gravity-aligned, metric SLAM world, z up; every pose here is mapped into it the way the upstream
loaders do (``lib/datasets/{ego_exo_scene,exo_camera,aria_human}.py`` in rawalkhirodkar/egohumans and jyuntins/harmony4d).
"""

from __future__ import annotations

import io
import pickle
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal, TypeAlias

import cv2
import numpy as np
from jaxtyping import Bool, Float32, Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from serde import serde
from simplecv.camera_parameters import Fisheye624Parameters
from simplecv.se3 import SE3
from simplecv.sensors.camera import fisheye624

FPS: int = 20
"""Release frame rate: both papers cut their takes at 20 fps; the files carry frame numbers, no timestamps."""
ANCHOR: str = "aria01"
"""The Aria whose SLAM world every pose is mapped into (upstream ``CALIBRATION.ANCHOR_EGO_CAMERA``)."""
AriaMode: TypeAlias = Literal["rgb", "left", "right"]
ARIA_MODES: tuple[AriaMode, ...] = ("rgb", "left", "right")
"""Order of an Aria's cameras in its calib file and under its rig (``cam_00`` rgb, ``cam_01`` left, ``cam_02`` right)."""
ARIA_STREAM_IDS: dict[AriaMode, str] = {"rgb": "214-1", "left": "1201-1", "right": "1201-2"}
"""VRS stream id of each Aria camera (the release ships extracted JPEGs, the ids name the sensor)."""
ARIA_MAX_SOLID_ANGLE: dict[AriaMode, float] = {"rgb": np.deg2rad(55.0), "left": np.deg2rad(75.0), "right": np.deg2rad(75.0)}
"""Half-angle a point must lie within to project, as projectaria-tools' Gen1 camera defaults (``dataforge.aria``)."""
PICKLE_GLOBALS: frozenset[tuple[str, str]] = frozenset(
    {(module, name) for module in ("numpy.core.multiarray", "numpy._core.multiarray") for name in ("_reconstruct", "scalar")}
    | {(module, "_frombuffer") for module in ("numpy.core.numeric", "numpy._core.numeric")}
    | {("numpy", "dtype"), ("numpy", "ndarray")}
)
"""Everything a release pickle may build besides plain containers: numpy arrays and scalars (numpy 1 wrote ``numpy.core``)."""
COCO17: int = 17
SMPL_JOINTS: int = 45
SMPL_VERTICES: int = 6890


class NumpyOnlyUnpickler(pickle.Unpickler):
    """Unpickles plain containers and numpy arrays only: a release pickle naming any other global is refused, not run."""

    def find_class(self, module: str, name: str) -> object:
        if (module, name) not in PICKLE_GLOBALS:
            raise pickle.UnpicklingError(f"the pickle refers to {module}.{name}, which is not a numpy type")
        return super().find_class(module, name)


def load_pickle(payload: bytes, origin: str) -> object:
    """A release pickle, or an object ``.npy`` (a 0-d object array around one pickled dict), through ``NumpyOnlyUnpickler``."""
    if payload.startswith(b"\x93NUMPY"):
        header_length: int = int.from_bytes(payload[8:10], "little") if payload[6] == 1 else int.from_bytes(payload[8:12], "little")
        start: int = (10 if payload[6] == 1 else 12) + header_length
        if b"'|O'" not in payload[:start]:
            return np.load(io.BytesIO(payload), allow_pickle=False)
        payload = payload[start:]
    try:
        value: object = NumpyOnlyUnpickler(io.BytesIO(payload)).load()
    except (pickle.UnpicklingError, EOFError, ValueError) as error:
        raise ValueError(f"{origin}: {error}") from error
    # np.save of a dict pickles the 0-d object array that wraps it.
    return value.item() if isinstance(value, np.ndarray) and value.dtype == object and value.shape == () else value


def person_dict(value: object, origin: str) -> dict[str, object]:
    """A per-frame label file: a dict keyed by person name (``aria01``...)."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{origin}: expected a dict keyed by person name")
    return value


# ── the per-sequence capture config ────────────────────────────────────────


@serde
@dataclass(frozen=True, slots=True)
class CaptureCalibration:
    """``CALIBRATION`` of an upstream sequence config: the cameras whose COLMAP pose was replaced by hand."""

    MANUAL_EXO_CAMERAS: list[str] = field(default_factory=list)
    """Cameras posed by ``colmap/workplace/<camera>.npy`` (a 3x4 or 4x4 camera-from-COLMAP), not by ``images.txt``."""


@serde
@dataclass(frozen=True, slots=True)
class CaptureConfig:
    """What an upstream sequence config (``egohumans/configs/<activity>/<sequence>.yaml``) says about the release's cameras.

    Third-party YAML read partially: the many processing keys (detector checkpoints, thresholds) are not ours to police.
    Harmony4D ships no per-sequence configs; its sequences use the default (every camera valid, none posed by hand).
    """

    INVALID_ARIAS: list[str] = field(default_factory=list)
    """Glasses the authors left out (no usable calibration or video); their wearer may still be labelled."""
    INVALID_EXOS: list[str] = field(default_factory=list)
    """Static cameras the authors left out."""
    CALIBRATION: CaptureCalibration = field(default_factory=CaptureCalibration)
    """Hand-posed cameras."""


# ── the sequence's files ───────────────────────────────────────────────────


class SequenceFiles(ABC):
    """Read access to one sequence directory, wherever it lives (a release zip, an extracted tar)."""

    @abstractmethod
    def read(self, relative: str) -> bytes:
        """Whole contents of ``relative`` (a posix path under the sequence)."""

    @abstractmethod
    def files(self, relative: str) -> list[str]:
        """Sorted names of the files directly in directory ``relative``; empty when it does not exist."""

    @abstractmethod
    def exists(self, relative: str) -> bool:
        """Whether file ``relative`` exists (asked once per frame and label directory: must not list a directory)."""

    def directories(self, relative: str) -> list[str]:
        """Sorted names of the subdirectories of ``relative`` that hold files somewhere below."""
        return sorted({name.split("/", 1)[0] for name in self.tree(relative) if "/" in name})

    @abstractmethod
    def tree(self, relative: str) -> list[str]:
        """Every file below ``relative``, as paths relative to it."""


class ZipSequenceFiles(SequenceFiles):
    """A sequence inside a release zip (Harmony4D ships one zip per scene, several sequences each), read in place."""

    def __init__(self, archive: zipfile.ZipFile, prefix: str) -> None:
        self.archive: zipfile.ZipFile = archive
        self.prefix: str = prefix.rstrip("/")
        self.listing: dict[str, list[str]] = {}
        """Directory (relative to the sequence) -> file names, built once from the zip's central directory."""
        for name in archive.namelist():
            if name.endswith("/") or not name.startswith(self.prefix + "/"):
                continue
            path: PurePosixPath = PurePosixPath(name[len(self.prefix) + 1 :])
            self.listing.setdefault(str(path.parent), []).append(path.name)
        for names in self.listing.values():
            names.sort()
        self.members: frozenset[str] = frozenset(f"{directory}/{name}" for directory, names in self.listing.items() for name in names)

    def read(self, relative: str) -> bytes:
        return self.archive.read(f"{self.prefix}/{relative}")

    def exists(self, relative: str) -> bool:
        return relative in self.members

    def files(self, relative: str) -> list[str]:
        return self.listing.get(relative.rstrip("/") or ".", [])

    def tree(self, relative: str) -> list[str]:
        base: str = relative.rstrip("/")
        return sorted(
            str(PurePosixPath(directory, name).relative_to(base))
            for directory, names in self.listing.items()
            if directory == base or directory.startswith(base + "/")
            for name in names
        )


class DirectorySequenceFiles(SequenceFiles):
    """A sequence directory on disk (EgoHumans ships one tar.gz per sequence, extracted before conversion)."""

    def __init__(self, root: Path) -> None:
        self.root: Path = root

    def read(self, relative: str) -> bytes:
        return (self.root / relative).read_bytes()

    def exists(self, relative: str) -> bool:
        return (self.root / relative).is_file()

    def files(self, relative: str) -> list[str]:
        directory: Path = self.root / relative
        return sorted(path.name for path in directory.iterdir() if path.is_file()) if directory.is_dir() else []

    def tree(self, relative: str) -> list[str]:
        base: Path = self.root / relative
        return sorted(str(path.relative_to(base)) for path in base.rglob("*") if path.is_file()) if base.is_dir() else []


# ── cameras ────────────────────────────────────────────────────────────────


def frame_number(name: str) -> int:
    """``00042.jpg`` / ``00042.npy`` / ``00042.txt`` -> 42."""
    return int(PurePosixPath(name).stem)


def rigid(cam_T_world: Float64[ndarray, "4 4"]) -> Float64[ndarray, "4 4"]:
    """``world_T_cam`` of a world->camera similarity: the COLMAP maps carry a scale (1.84 on 002_hugging), which divides out of a
    projection but not out of a pose; dividing it from rotation and translation leaves the camera's metric pose."""
    scale: float = float(np.cbrt(np.linalg.det(cam_T_world[:3, :3])))
    return np.linalg.inv(matrix(cam_T_world[:3, :3] / scale, cam_T_world[:3, 3] / scale))


def matrix(rotation: Float64[ndarray, "3 3"], translation: Float64[ndarray, "3"]) -> Float64[ndarray, "4 4"]:
    """The 4x4 rigid transform with this rotation and translation."""
    result: Float64[ndarray, "4 4"] = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


@dataclass(frozen=True, slots=True)
class ExoCamera:
    """One static GoPro: COLMAP's OPENCV_FISHEYE lens and its pose in the anchor world."""

    name: str
    """``cam01``..., as the release names its directory."""
    width: int
    """Source image width in pixels (3840)."""
    height: int
    """Source image height in pixels (2160)."""
    lens: Float64[ndarray, "8"]
    """``fx, fy, cx, cy, k1, k2, k3, k4``: COLMAP OPENCV_FISHEYE (Kannala-Brandt, four radial terms) at the source size."""
    world_T_cam: Float64[ndarray, "4 4"]
    """Camera pose in the anchor world, metres."""
    registration: str
    """The COLMAP image the pose comes from (each camera is registered at a few frames; they agree within ~2 mm)."""

    @property
    def number(self) -> int:
        """``cam07`` -> 7."""
        return int(self.name.removeprefix("cam"))

    def project(self, points_world: Float64[ndarray, "n 3"]) -> Float64[ndarray, "n 2"]:
        """World points through the full lens, source pixels; NaN behind the camera or outside the image."""
        cam_T_world: Float64[ndarray, "4 4"] = np.linalg.inv(self.world_T_cam)
        points_cam: Float64[ndarray, "n 3"] = points_world @ cam_T_world[:3, :3].T + cam_T_world[:3, 3]
        pixels: Float64[ndarray, "n 2"] = np.full((len(points_world), 2), np.nan)
        front: Bool[ndarray, "n"] = np.isfinite(points_cam).all(axis=1) & (points_cam[:, 2] > 0.0)
        if front.any():
            K: Float64[ndarray, "3 3"] = np.array([[self.lens[0], 0.0, self.lens[2]], [0.0, self.lens[1], self.lens[3]], [0.0, 0.0, 1.0]])
            projected: Float64[ndarray, "m 1 2"] = cv2.fisheye.projectPoints(points_cam[front].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, self.lens[4:])[0]
            pixels[front] = projected.reshape(-1, 2)
        inside: Bool[ndarray, "n"] = (pixels[:, 0] >= 0) & (pixels[:, 0] < self.width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < self.height)
        pixels[~inside] = np.nan
        return pixels


@dataclass(frozen=True, slots=True)
class AriaCamera:
    """One Aria camera over the sequence, in its device (VRS) orientation; the stored JPEGs are turned a quarter turn clockwise."""

    mode: AriaMode
    """rgb, left or right."""
    width: int
    """Device-orientation width (rgb 1408, slam 640)."""
    height: int
    """Device-orientation height (rgb 1408, slam 480)."""
    intrinsics: Float64[ndarray, "t 15"]
    """Per-frame FISHEYE624 ``[f, cx, cy, k1..k6, p1, p2, s1..s4]`` (Aria's online calibration moves it by < 0.1 px)."""
    world_T_cam: Float64[ndarray, "t 4 4"]
    """Per-frame camera pose in the anchor world, metres."""

    def calibration(self, frame: int) -> Fisheye624Parameters:
        """Frame ``frame``'s lens in the stored (upright) orientation, ``rig_T_cam`` left as identity."""
        device: Fisheye624Parameters = Fisheye624Parameters(
            name=self.mode,
            width=self.width,
            height=self.height,
            params=self.intrinsics[frame],
            rig_T_cam=SE3(np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3)),
            max_solid_angle=float(ARIA_MAX_SOLID_ANGLE[self.mode]),
        )
        return fisheye624.rotate_cw90(device)

    def project(self, frame: int, points_world: Float64[ndarray, "n 3"]) -> Float64[ndarray, "n 2"]:
        """World points through frame ``frame``'s full lens, stored (upright) pixels; NaN where the lens does not see them."""
        upright: Fisheye624Parameters = self.calibration(frame)
        world_T_upright: Float64[ndarray, "4 4"] = self.world_T_cam[frame] @ upright.rig_T_cam.matrix()
        cam_T_world: Float64[ndarray, "4 4"] = np.linalg.inv(world_T_upright)
        points_cam: Float64[ndarray, "n 3"] = points_world @ cam_T_world[:3, :3].T + cam_T_world[:3, 3]
        # project() keeps a point only inside the lens cone and the image (NaN points fail the cone test).
        return fisheye624.project(upright, points_cam)


@dataclass(frozen=True, slots=True)
class AriaGlasses:
    """One pair of Aria glasses and the person wearing them."""

    name: str
    """``aria01``..., also the wearer's person key in the labels."""
    cameras: tuple[AriaCamera, AriaCamera, AriaCamera]
    """rgb, left, right (``ARIA_MODES``)."""

    @property
    def number(self) -> int:
        """``aria02`` -> 2."""
        return int(self.name.removeprefix("aria"))

    @property
    def world_T_rig(self) -> Float64[ndarray, "t 4 4"]:
        """The rig frame is the rgb camera's device-orientation frame."""
        return self.cameras[0].world_T_cam

    def rig_T_cam(self, mode: AriaMode) -> Float64[ndarray, "4 4"]:
        """A camera's pose on the rig at the first frame (the slam cameras stay within 0.3 mm and 0.03 deg of it)."""
        camera: AriaCamera = self.cameras[ARIA_MODES.index(mode)]
        return np.linalg.inv(self.world_T_rig[0]) @ camera.world_T_cam[0]


# ── labels ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PersonTrack:
    """One person over the sequence, NaN on frames the release has no label for them."""

    name: str
    """``aria01``...: the glasses this person wears (or would wear: the key stays in exo-only takes)."""
    coco17: Float32[ndarray, "t 17 3"]
    """COCO-17 body keypoints, world metres (``processed_data/poses3d``)."""
    coco17_confidence: Float32[ndarray, "t 17"]
    """Shipped confidence of each keypoint; 0.0 on missing rows."""
    global_orient: Float32[ndarray, "t 3"]
    """SMPL root rotation, axis-angle."""
    body_pose: Float32[ndarray, "t 69"]
    """SMPL body pose, 23 joints axis-angle."""
    betas: Float32[ndarray, "t 10"]
    """SMPL shape per frame, as shipped."""
    transl: Float32[ndarray, "t 3"]
    """SMPL translation (applied after the root-joint-pivoted pose, smplx convention)."""
    vertices: Float32[ndarray, "t 6890 3"]
    """Shipped posed SMPL vertices, world metres."""
    joints: Float32[ndarray, "t 45 3"]
    """Shipped SMPL-45 joints (24 SMPL + 21 extra), world metres."""
    fit_loss: Float32[ndarray, "t"]
    """The fit's final ``epoch_loss``; NaN where absent."""

    @property
    def number(self) -> int:
        """``aria02`` -> 2."""
        return int(self.name.removeprefix("aria"))

    @property
    def present(self) -> Bool[ndarray, "t"]:
        """Frames with an SMPL fit."""
        return np.isfinite(self.vertices[:, 0, 0])


@dataclass(frozen=True, slots=True)
class CameraBoxes:
    """One camera's shipped person boxes, ``[x1, y1, x2, y2]`` stored pixels, NaN where a person has none."""

    camera: str
    """``cam03`` or ``aria01/rgb``."""
    boxes: dict[str, Float32[ndarray, "t 4"]]
    """Person name -> per-frame box."""


@dataclass(frozen=True, slots=True)
class Sequence:
    """One sequence: its frames, cameras and labels, all in the anchor world."""

    frames: Int64[ndarray, "t"]
    """Release frame numbers (file names), ascending; from 1."""
    exo: list[ExoCamera]
    """Static cameras, by name."""
    glasses: list[AriaGlasses]
    """Aria glasses, by name; empty in an exo-only take."""
    people: list[PersonTrack]
    """People, by name."""
    boxes: list[CameraBoxes]
    """Shipped boxes of every camera that has a box directory."""
    world_source: str
    """How the world was reached: ``colmap_from_aria_transforms.pkl`` or ``scale.npy``."""
    keypoints_source: str
    """The ``processed_data`` directory the COCO-17 keypoints came from."""

    @property
    def video_times_ns(self) -> Int64[ndarray, "t"]:
        """``video_time``: (frame - first frame) / ``FPS``."""
        return (self.frames - self.frames[0]) * (1_000_000_000 // FPS)


def read_colmap_cameras(text: str) -> dict[int, tuple[str, int, int, Float64[ndarray, "p"]]]:
    """COLMAP ``cameras.txt``: id -> (model, width, height, params)."""
    cameras: dict[int, tuple[str, int, int, Float64[ndarray, "p"]]] = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts: list[str] = line.split()
        cameras[int(parts[0])] = (parts[1], int(parts[2]), int(parts[3]), np.array([float(value) for value in parts[4:]]))
    return cameras


def read_colmap_images(text: str) -> list[tuple[str, int, Float64[ndarray, "4 4"]]]:
    """COLMAP ``images.txt``: (image name, camera id, cam_T_colmap) for every image; the point lines are skipped."""
    lines: list[str] = [line for line in text.splitlines() if not line.startswith("#")]
    images: list[tuple[str, int, Float64[ndarray, "4 4"]]] = []
    for header in lines[::2]:
        parts: list[str] = header.split()
        if len(parts) < 10:
            continue
        qw, qx, qy, qz, tx, ty, tz = (float(value) for value in parts[1:8])
        rotation: Float64[ndarray, "3 3"] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        images.append((parts[9], int(parts[8]), matrix(rotation, np.array([tx, ty, tz]))))
    return images


def read_world(files: SequenceFiles) -> tuple[dict[str, Float64[ndarray, "4 4"]], Float64[ndarray, "4 4"], str]:
    """(each Aria's world -> COLMAP, anchor world -> COLMAP, source file): the two ways a take reaches its world.

    With glasses, ``colmap_from_aria_transforms.pkl`` maps every Aria's world into COLMAP and the anchor's is the world.
    Exo-only takes ship ``scale.npy`` (COLMAP -> metric world) instead, so the world is ``inv(scale)`` of COLMAP.
    """
    workplace: str = "colmap/workplace"
    if files.exists(f"{workplace}/scale.npy"):
        scale: Float64[ndarray, "4 4"] = np.asarray(load_pickle(files.read(f"{workplace}/scale.npy"), "scale.npy"), dtype=np.float64)
        return {}, np.linalg.inv(scale), "scale.npy"
    name: str = "colmap_from_aria_transforms.pkl"
    raw: object = load_pickle(files.read(f"{workplace}/{name}"), name)
    transforms: dict[str, Float64[ndarray, "4 4"]] = {key: np.asarray(value, dtype=np.float64) for key, value in person_dict(raw, name).items()}
    if ANCHOR not in transforms:
        raise ValueError(f"{name} has no {ANCHOR}: {sorted(transforms)}")
    return transforms, transforms[ANCHOR], name


def read_exo_cameras(files: SequenceFiles, colmap_from_world: Float64[ndarray, "4 4"], config: CaptureConfig) -> list[ExoCamera]:
    """Every valid ``exo/camNN``: lens at the source size and pose in the anchor world.

    The pose is the camera's earliest registered image: upstream picks the registration nearest each frame, and the few
    registrations of one static camera agree within ~2 mm. A hand-posed camera (``MANUAL_EXO_CAMERAS``) takes its pose from
    ``colmap/workplace/<camera>.npy`` instead and keeps the lens of its COLMAP camera, as upstream's ``ExoCamera`` does.
    """
    lenses: dict[int, tuple[str, int, int, Float64[ndarray, "p"]]] = read_colmap_cameras(files.read("colmap/workplace/cameras.txt").decode())
    images: list[tuple[str, int, Float64[ndarray, "4 4"]]] = read_colmap_images(files.read("colmap/workplace/images.txt").decode())
    cameras: list[ExoCamera] = []
    for name in files.directories("exo"):
        if not name.startswith("cam") or name in config.INVALID_EXOS:
            continue
        registered: list[tuple[str, int, Float64[ndarray, "4 4"]]] = sorted(
            (image for image in images if image[0].split("/")[0] == name), key=lambda image: frame_number(image[0])
        )
        if not registered:
            raise ValueError(f"exo/{name}: no COLMAP registration in images.txt")
        image_name, camera_id, cam_T_colmap = registered[0]
        if name in config.CALIBRATION.MANUAL_EXO_CAMERAS:
            manual: str = f"colmap/workplace/{name}.npy"
            extrinsic: Float64[ndarray, "r 4"] = np.asarray(load_pickle(files.read(manual), manual), dtype=np.float64)
            cam_T_colmap = np.eye(4)
            cam_T_colmap[:3] = extrinsic[:3]
            image_name = manual
        model, width, height, params = lenses[camera_id]
        if model != "OPENCV_FISHEYE" or len(params) != 8:
            raise ValueError(f"exo/{name}: COLMAP camera {camera_id} is {model} with {len(params)} parameters, expected OPENCV_FISHEYE with 8")
        world_T_cam: Float64[ndarray, "4 4"] = rigid(cam_T_colmap @ colmap_from_world)
        cameras.append(ExoCamera(name, width, height, params, world_T_cam, image_name))
    return cameras


def read_aria_calibration(text: str, origin: str) -> list[tuple[Float64[ndarray, "15"], Float64[ndarray, "4 4"]]]:
    """One calib file: (FISHEYE624 intrinsics, cam_T_aria_world) of rgb, left and right.

    Line 1 is a header, line 2 the device; then intrinsics and a 3x4 extrinsic per camera, the extrinsic written column by column
    (upstream reads it with ``reshape(4, 3).T``).
    """
    lines: list[str] = [line.strip() for line in text.splitlines()[1:] if line.strip()]
    if len(lines) != 7:
        raise ValueError(f"{origin}: {len(lines)} lines after the header, expected 7 (one person)")
    result: list[tuple[Float64[ndarray, "15"], Float64[ndarray, "4 4"]]] = []
    for index in range(3):
        intrinsics: Float64[ndarray, "15"] = np.array([float(value) for value in lines[1 + 2 * index].split()])
        columns: Float64[ndarray, "12"] = np.array([float(value) for value in lines[2 + 2 * index].split()])
        if intrinsics.shape != (15,) or columns.shape != (12,):
            raise ValueError(f"{origin}: camera {index} has {intrinsics.size} intrinsics and {columns.size} extrinsic values, expected 15 and 12")
        extrinsic: Float64[ndarray, "4 4"] = np.eye(4)
        extrinsic[:3] = columns.reshape(4, 3).T
        result.append((intrinsics, extrinsic))
    return result


def jpeg_size(payload: bytes) -> tuple[int, int]:
    """(width, height) from a JPEG's start-of-frame header."""
    index: int = 2
    while index < len(payload) - 9:
        marker: int = payload[index + 1]
        length: int = int.from_bytes(payload[index + 2 : index + 4], "big")
        if marker in (0xC0, 0xC1, 0xC2):
            return int.from_bytes(payload[index + 7 : index + 9], "big"), int.from_bytes(payload[index + 5 : index + 7], "big")
        index += 2 + length
    raise ValueError("JPEG without a frame header")


def read_glasses(
    files: SequenceFiles,
    frames: Int64[ndarray, "t"],
    aria_to_colmap: dict[str, Float64[ndarray, "4 4"]],
    colmap_from_world: Float64[ndarray, "4 4"],
    invalid: list[str],
) -> list[AriaGlasses]:
    """Every valid ``ego/ariaNN``, its cameras' per-frame lens and pose in the anchor world."""
    glasses: list[AriaGlasses] = []
    for name in files.directories("ego"):
        if not name.startswith("aria") or name in invalid:
            continue
        if name not in aria_to_colmap:
            raise ValueError(f"ego/{name}: not in colmap_from_aria_transforms.pkl")
        # A world point in the anchor world, seen from this Aria's world: upstream's ``coordinate_transform``.
        aria_world_T_world: Float64[ndarray, "4 4"] = np.linalg.inv(aria_to_colmap[name]) @ colmap_from_world
        calibrations: list[list[tuple[Float64[ndarray, "15"], Float64[ndarray, "4 4"]]]] = [
            read_aria_calibration(files.read(f"ego/{name}/calib/{frame:05d}.txt").decode(), f"ego/{name}/calib/{frame:05d}.txt") for frame in frames
        ]
        cameras: list[AriaCamera] = []
        for index, mode in enumerate(ARIA_MODES):
            upright: tuple[int, int] = jpeg_size(files.read(f"ego/{name}/images/{mode}/{frames[0]:05d}.jpg"))
            intrinsics: Float64[ndarray, "t 15"] = np.stack([calibration[index][0] for calibration in calibrations])
            world_T_cam: Float64[ndarray, "t 4 4"] = np.stack([rigid(calibration[index][1] @ aria_world_T_world) for calibration in calibrations])
            # The stored image is upright: its height is the device-orientation width.
            cameras.append(AriaCamera(mode, upright[1], upright[0], intrinsics, world_T_cam))
        glasses.append(AriaGlasses(name, (cameras[0], cameras[1], cameras[2])))
    return glasses


def read_frames(files: SequenceFiles, exo: list[ExoCamera]) -> Int64[ndarray, "t"]:
    """The release frames: the first exo camera's JPEG numbers, which every other exo camera must match."""
    numbered: list[list[int]] = [[frame_number(name) for name in files.files(f"exo/{camera.name}/images") if name.endswith(".jpg")] for camera in exo]
    if not numbered or not numbered[0]:
        raise ValueError("no exo camera with JPEG frames")
    for camera, numbers in zip(exo, numbered, strict=True):
        if numbers != numbered[0]:
            raise ValueError(f"exo/{camera.name}: {len(numbers)} JPEG frames differ from {exo[0].name}'s {len(numbered[0])}")
    return np.array(numbered[0], dtype=np.int64)


def label_dir(files: SequenceFiles, names: tuple[str, ...]) -> str | None:
    """The first of ``names`` (under ``processed_data``) that the release ships, or None."""
    for name in names:
        if files.tree(f"processed_data/{name}"):
            return f"processed_data/{name}"
    return None


def read_people(files: SequenceFiles, frames: Int64[ndarray, "t"], pose_dir: str) -> list[PersonTrack]:
    """Every person named in the SMPL or keypoint files, at every frame (NaN rows where a frame has no label for them)."""
    smpl_dir: str | None = label_dir(files, ("smpl",))
    if smpl_dir is None:
        raise ValueError("processed_data lacks smpl")
    smpl_frames: list[dict[str, object]] = []
    pose_frames: list[dict[str, object]] = []
    for frame in frames:
        smpl_name: str = f"{smpl_dir}/{frame:05d}.npy"
        pose_name: str = f"{pose_dir}/{frame:05d}.npy"
        smpl_frames.append(person_dict(load_pickle(files.read(smpl_name), smpl_name), smpl_name) if files.exists(smpl_name) else {})
        pose_frames.append(person_dict(load_pickle(files.read(pose_name), pose_name), pose_name) if files.exists(pose_name) else {})
    names: list[str] = sorted({name for record in smpl_frames for name in record} | {name for record in pose_frames for name in record})
    t: int = len(frames)
    people: list[PersonTrack] = []
    for name in names:
        arrays: dict[str, Float32[ndarray, "..."]] = {
            "coco17": np.full((t, COCO17, 3), np.nan, dtype=np.float32),
            "coco17_confidence": np.zeros((t, COCO17), dtype=np.float32),
            "global_orient": np.full((t, 3), np.nan, dtype=np.float32),
            "body_pose": np.full((t, 69), np.nan, dtype=np.float32),
            "betas": np.full((t, 10), np.nan, dtype=np.float32),
            "transl": np.full((t, 3), np.nan, dtype=np.float32),
            "vertices": np.full((t, SMPL_VERTICES, 3), np.nan, dtype=np.float32),
            "joints": np.full((t, SMPL_JOINTS, 3), np.nan, dtype=np.float32),
            "fit_loss": np.full(t, np.nan, dtype=np.float32),
        }
        for row in range(t):
            pose: object = pose_frames[row].get(name)
            if pose is not None:
                keypoints: Float64[ndarray, "17 4"] = np.asarray(pose, dtype=np.float64).reshape(COCO17, 4)
                arrays["coco17"][row] = keypoints[:, :3]
                arrays["coco17_confidence"][row] = keypoints[:, 3]
            fit: object = smpl_frames[row].get(name)
            if fit is None:
                continue
            if not isinstance(fit, dict):
                raise ValueError(f"{smpl_dir}/{frames[row]:05d}.npy: {name} is not a dict")
            for key in ("global_orient", "body_pose", "betas", "transl", "vertices", "joints"):
                arrays[key][row] = np.asarray(fit[key], dtype=np.float32).reshape(arrays[key].shape[1:])
            loss: object = fit.get("epoch_loss")
            arrays["fit_loss"][row] = np.nan if loss is None else float(np.asarray(loss))
        people.append(PersonTrack(name, **arrays))
    return people


def read_boxes(files: SequenceFiles, frames: Int64[ndarray, "t"], cameras: list[str], people: list[PersonTrack]) -> list[CameraBoxes]:
    """Shipped boxes of each camera (``cam03``, ``aria01/rgb``...); a camera without box files is left out.

    A GoPro's boxes are ``<box dir>/camNN/rgb/NNNNN.npy`` in some sequences and ``<box dir>/camNN/NNNNN.npy`` in most (90 of
    95 Harmony4D sequences sampled; where both exist they hold the same values): the ``rgb`` level is used when it has files.
    """
    box_dir: str | None = label_dir(files, ("bbox", "bboxes"))
    if box_dir is None:
        return []
    result: list[CameraBoxes] = []
    for camera in cameras:
        directory: str = f"{box_dir}/{camera}"
        if "/" not in camera and files.files(f"{directory}/rgb"):
            directory += "/rgb"
        if not files.files(directory):
            continue
        boxes: dict[str, Float32[ndarray, "t 4"]] = {person.name: np.full((len(frames), 4), np.nan, dtype=np.float32) for person in people}
        for row, frame in enumerate(frames):
            name: str = f"{directory}/{frame:05d}.npy"
            if not files.exists(name):
                continue
            for person, box in person_dict(load_pickle(files.read(name), name), name).items():
                if person in boxes and box is not None:
                    boxes[person][row] = np.asarray(box, dtype=np.float32).reshape(-1)[:4]
        result.append(CameraBoxes(camera, boxes))
    return result


def read_sequence(files: SequenceFiles, *, pose_dirs: tuple[str, ...] = ("poses3d",), config: CaptureConfig | None = None) -> Sequence:
    """Everything but the images: cameras, glasses, people and boxes, mapped into the anchor world.

    Args:
        files: The sequence directory.
        pose_dirs: ``processed_data`` directories to take COCO-17 keypoints from, first found wins.
        config: The sequence's upstream config; None is the default (every camera valid, none posed by hand).
    """
    capture: CaptureConfig = config or CaptureConfig()
    aria_to_colmap, colmap_from_world, world_source = read_world(files)
    exo: list[ExoCamera] = read_exo_cameras(files, colmap_from_world, capture)
    frames: Int64[ndarray, "t"] = read_frames(files, exo)
    glasses: list[AriaGlasses] = read_glasses(files, frames, aria_to_colmap, colmap_from_world, capture.INVALID_ARIAS) if aria_to_colmap else []
    pose_dir: str | None = label_dir(files, pose_dirs)
    if pose_dir is None:
        raise ValueError(f"processed_data lacks {' / '.join(pose_dirs)}")
    people: list[PersonTrack] = read_people(files, frames, pose_dir)
    box_cameras: list[str] = [camera.name for camera in exo] + [f"{pair.name}/{mode}" for pair in glasses for mode in ARIA_MODES]
    return Sequence(frames, exo, glasses, people, read_boxes(files, frames, box_cameras, people), world_source, pose_dir)
