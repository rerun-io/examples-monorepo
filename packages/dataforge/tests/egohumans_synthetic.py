"""A synthetic sequence in the EgoHumans capture layout, with geometry whose projections are known by construction.

The world is aria01's. COLMAP's frame is a similarity of it (scale 2, a quarter turn about z, an offset), so a reader that forgets
to divide the scale out of a pose puts the cameras at the wrong distance. Two GoPros look along world +y and -x from 3 m; aria01's
rgb camera sits at the origin looking along world +y (device orientation: x = world +x, y = world -z). Two people: aria01 wears
the glasses and is labelled on every frame, aria02 is missing from the last frame's labels.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import turbojpeg
from jaxtyping import Float64

from dataforge.datasets.egohumans_source import AriaMode

FRAMES: tuple[int, ...] = (1, 2, 3)
EXO_SIZE: tuple[int, int] = (3840, 2160)
"""The GoPro size the COLMAP model states; the JPEGs are this size too, so the base layer halves them to 1920x1080."""
EXO_LENS: tuple[float, ...] = (1750.0, 1752.0, 1919.5, 1079.5, 0.04, 0.06, -0.07, 0.03)
ARIA_RGB_UPRIGHT: tuple[int, int] = (256, 256)
"""Upright stored sizes; AV1 NVENC refuses frames much smaller than these."""
ARIA_SLAM_UPRIGHT: tuple[int, int] = (192, 256)
ARIA_FOCAL: float = 120.0
COLMAP_SCALE: float = 2.0


def rotation_z(quarter_turns: int) -> Float64[np.ndarray, "3 3"]:
    angle: float = np.pi / 2 * quarter_turns
    return np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])


COLMAP_FROM_WORLD: Float64[np.ndarray, "4 4"] = np.eye(4)
COLMAP_FROM_WORLD[:3, :3] = COLMAP_SCALE * rotation_z(1)
COLMAP_FROM_WORLD[:3, 3] = (1.0, 2.0, 3.0)

# Camera axes in the world (columns x, y, z): looking along +y with x = +x, y = -z; and along -x with x = +y, y = -z.
LOOK_PLUS_Y: Float64[np.ndarray, "3 3"] = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
LOOK_MINUS_X: Float64[np.ndarray, "3 3"] = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
EXO_POSES: dict[str, Float64[np.ndarray, "4 4"]] = {}
for _name, _axes, _centre in (("cam01", LOOK_PLUS_Y, (0.0, -3.0, 1.0)), ("cam02", LOOK_MINUS_X, (3.0, 0.0, 1.0))):
    _pose = np.eye(4)
    _pose[:3, :3] = _axes
    _pose[:3, 3] = _centre
    EXO_POSES[_name] = _pose
ARIA_RGB_POSE: Float64[np.ndarray, "4 4"] = np.eye(4)
ARIA_RGB_POSE[:3, :3] = LOOK_PLUS_Y
ARIA_SLAM_OFFSETS: dict[AriaMode, Float64[np.ndarray, "3"]] = {"left": np.array([0.05, 0.0, 0.0]), "right": np.array([-0.05, 0.0, 0.0])}


def gradient_jpeg(width: int, height: int, value: int) -> bytes:
    """A JPEG whose frames differ (``value``) and compress well."""
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[..., 0] = (np.arange(width, dtype=np.int64)[None, :] * 255 // max(width - 1, 1)).astype(np.uint8)
    pixels[..., 1] = (np.arange(height, dtype=np.int64)[:, None] * 255 // max(height - 1, 1)).astype(np.uint8)
    pixels[..., 2] = value
    encoded = turbojpeg.TurboJPEG().encode(pixels, quality=60)
    assert isinstance(encoded, bytes)
    return encoded


def npy_pickle(path: Path, value: object) -> None:
    """``np.save`` of a dict, as the release writes its labels (a 0-d object array)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.array(value, dtype=object), allow_pickle=True)


def colmap_image_line(image_id: int, name: str, camera_id: int, world_T_cam: Float64[np.ndarray, "4 4"]) -> str:
    """One images.txt header line: the camera's rigid pose in COLMAP's frame, as COLMAP writes it (w, x, y, z then t)."""
    from scipy.spatial.transform import Rotation

    colmap_T_cam: Float64[np.ndarray, "4 4"] = COLMAP_FROM_WORLD @ world_T_cam
    rotation: Float64[np.ndarray, "3 3"] = colmap_T_cam[:3, :3] / COLMAP_SCALE  # COLMAP poses are rigid in its own units
    centre: Float64[np.ndarray, "3"] = colmap_T_cam[:3, 3]
    cam_R_colmap: Float64[np.ndarray, "3 3"] = rotation.T
    translation: Float64[np.ndarray, "3"] = -cam_R_colmap @ centre
    x, y, z, w = Rotation.from_matrix(cam_R_colmap).as_quat()
    return f"{image_id} {w} {x} {y} {z} {translation[0]} {translation[1]} {translation[2]} {camera_id} {name}"


def aria_calibration(frame: int) -> str:
    """aria01's calib file: rgb at ARIA_RGB_POSE, the slam cameras beside it; extrinsics written column by column."""
    lines: list[str] = ["Serial, intrinsics (radtanthinprsim), extrinsic (3x4)", "aria"]
    for mode, (upright_width, upright_height) in (("rgb", ARIA_RGB_UPRIGHT), ("left", ARIA_SLAM_UPRIGHT), ("right", ARIA_SLAM_UPRIGHT)):
        # Device orientation is the upright image turned back: device width = upright height.
        cx: float = (upright_height - 1) / 2
        cy: float = (upright_width - 1) / 2
        intrinsics = [ARIA_FOCAL + 0.001 * frame, cx, cy, *([0.0] * 12)]
        pose: Float64[np.ndarray, "4 4"] = ARIA_RGB_POSE.copy()
        if mode != "rgb":
            pose[:3, 3] += ARIA_SLAM_OFFSETS[mode]
        cam_T_world: Float64[np.ndarray, "4 4"] = np.linalg.inv(pose)
        lines.append(" ".join(f"{value:.12f}" for value in intrinsics))
        lines.append(" ".join(f"{value:.12f}" for value in cam_T_world[:3].T.reshape(-1)))
    return "\n".join(lines) + "\n"


def person_pose(name: str, frame: int) -> tuple[Float64[np.ndarray, "17 4"], dict[str, object]]:
    """A person standing 2 m (aria01 at 0, 2, aria02 at 1, 2) in front of the glasses; joints spread vertically."""
    base: Float64[np.ndarray, "3"] = np.array([0.0 if name == "aria01" else 1.0, 2.0, 0.0]) + 0.01 * frame
    keypoints: Float64[np.ndarray, "17 4"] = np.zeros((17, 4))
    keypoints[:, :3] = base + np.stack([np.zeros(17), np.zeros(17), np.linspace(-0.8, 0.8, 17)], axis=1)
    keypoints[:, 3] = 0.9
    vertices = np.repeat(base[None, :], 6890, axis=0).astype(np.float32)
    smpl: dict[str, object] = {
        "global_orient": np.zeros(3, dtype=np.float32),
        "transl": base.astype(np.float32),
        "body_pose": np.zeros(69, dtype=np.float32),
        "betas": np.full(10, 0.1, dtype=np.float32),
        "epoch_loss": 0.5,
        "vertices": vertices,
        "joints": np.repeat(base[None, :], 45, axis=0).astype(np.float32),
    }
    return keypoints, smpl


def write_sequence(directory: Path, *, glasses: bool = True, pose_dir: str = "poses3d", box_dir: str = "bbox", flat_boxes: bool = False) -> Path:
    """Write the sequence under ``directory`` (``<scene>/<sequence>`` is up to the caller) and return it.

    Harmony4D ships ``poses3d`` and ``bbox`` (a dict per frame keyed by person); EgoHumans ``fit_poses3d`` (among earlier
    stages) and ``bboxes`` (a list of upstream box records per frame). Most Harmony4D sequences put a GoPro's boxes straight
    under ``bbox/camNN`` (``flat_boxes``), a few under ``bbox/camNN/rgb``.
    """
    for name in EXO_POSES:
        images: Path = directory / "exo" / name / "images"
        images.mkdir(parents=True)
        for frame in FRAMES:
            (images / f"{frame:05d}.jpg").write_bytes(gradient_jpeg(*EXO_SIZE, 40 * frame))
        if name == "cam01":  # leftovers some release image directories hold (train/12_mma/003_mma)
            (images / "modified_image.jpg").write_bytes(gradient_jpeg(*EXO_SIZE, 0))
            (images / "draw_bbox.py").write_text("print('authoring script')\n")
    workplace: Path = directory / "colmap" / "workplace"
    workplace.mkdir(parents=True)
    width, height = EXO_SIZE
    (workplace / "cameras.txt").write_text(
        "# Camera list\n# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n# Number of cameras: 2\n"
        + "".join(f"{index} OPENCV_FISHEYE {width} {height} {' '.join(map(str, EXO_LENS))}\n" for index in (1, 2))
    )
    lines: list[str] = ["# Image list", "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME", "#   POINTS2D[]", "# Number of images: 3"]
    # cam01 is registered twice; the reader takes the earliest frame (00001), whose pose is the true one.
    shifted: Float64[np.ndarray, "4 4"] = EXO_POSES["cam01"].copy()
    shifted[0, 3] += 0.002
    for image_id, (name, camera_id, pose) in enumerate(
        (("cam01/00301.jpg", 1, shifted), ("cam01/00001.jpg", 1, EXO_POSES["cam01"]), ("cam02/00001.jpg", 2, EXO_POSES["cam02"])), start=1
    ):
        lines += [colmap_image_line(image_id, name, camera_id, pose), ""]
    (workplace / "images.txt").write_text("\n".join(lines) + "\n")
    if glasses:
        with (workplace / "colmap_from_aria_transforms.pkl").open("wb") as stream:
            pickle.dump({"aria01": COLMAP_FROM_WORLD.copy()}, stream)
        aria: Path = directory / "ego" / "aria01"
        for mode, size in (("rgb", ARIA_RGB_UPRIGHT), ("left", ARIA_SLAM_UPRIGHT), ("right", ARIA_SLAM_UPRIGHT)):
            (aria / "images" / mode).mkdir(parents=True)
            for frame in FRAMES:
                (aria / "images" / mode / f"{frame:05d}.jpg").write_bytes(gradient_jpeg(*size, 60 * frame))
        (aria / "calib").mkdir()
        for frame in FRAMES:
            (aria / "calib" / f"{frame:05d}.txt").write_text(aria_calibration(frame))
    else:
        np.save(workplace / "scale.npy", np.linalg.inv(COLMAP_FROM_WORLD))
    for frame in FRAMES:
        names: list[str] = ["aria01", "aria02"] if frame != FRAMES[-1] else ["aria01"]
        poses = {name: person_pose(name, frame) for name in names}
        npy_pickle(directory / "processed_data" / pose_dir / f"{frame:05d}.npy", {name: pose[0] for name, pose in poses.items()})
        npy_pickle(directory / "processed_data" / "smpl" / f"{frame:05d}.npy", {name: pose[1] for name, pose in poses.items()})
        boxes = directory / "processed_data" / box_dir / "cam01" / ("" if flat_boxes else "rgb")
        if box_dir == "bboxes":
            records = [{"bbox": np.array([10, 20, 110, 220, 1.0]), "human_name": name, "human_id": index, "color": [255, 0, 0]} for index, name in enumerate(names)]
            npy_pickle(boxes / f"{frame:05d}.npy", records)
        else:
            npy_pickle(boxes / f"{frame:05d}.npy", {name: np.array([10, 20, 110, 220]) for name in names})
    return directory
