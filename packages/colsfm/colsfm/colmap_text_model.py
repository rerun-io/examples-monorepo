# SPDX-License-Identifier: Apache-2.0
"""A COLMAP text model read without pycolmap.

`colsfm` reads models through `pycolmap.Reconstruction`, which is the right
answer wherever pycolmap is already present. The Rerun demo's environment is a
CUDA-13 solve with a TensorRT dependency override, and pulling pycolmap's
ceres/CUDA stack into it for the sake of two text formats — a dozen lines each —
risks perturbing it. So this module reads those two formats with numpy and scipy
and nothing else, which is what lets the demo and the pycolmap side share one
parser instead of keeping two.

It handles the one thing a naive reader gets wrong. `images.txt` writes **two**
lines per image: a pose line, then that image's 2D observations. An image that
registered no points still gets its second line, and that line is empty.
Dropping empty lines before pairing therefore shifts every later image onto an
observation line, and the images after the first unobserved one are silently
lost. The lines are consumed in strict pairs instead.

Only what a viewer needs is parsed: image names, camera poses, point positions
and point colours. Tracks, observations, cameras, rigs and frames are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float64, UInt8
from numpy import ndarray
from scipy.spatial.transform import Rotation

POSE_LINE_FIELD_COUNT: int = 10
"""`IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME`; fewer fields is not a pose line."""

POINT_LINE_FIELD_COUNT: int = 7
"""`POINT3D_ID X Y Z R G B ...`; fewer fields is not a point line."""


@dataclass
class ColmapModel:
    """The parts of a COLMAP sparse model a viewer needs."""

    world_T_cam: dict[str, Float64[ndarray, "4 4"]]
    """Image name (as written in ``frames_meta.json``) -> camera pose in world."""
    points_xyz: Float64[ndarray, "n_points 3"]
    """Sparse point positions."""
    points_rgb: UInt8[ndarray, "n_points 3"]
    """Per-point colour."""


@dataclass(frozen=True, slots=True)
class ColmapImage:
    """One registered image, as written on a pose line of ``images.txt``."""

    image_id: int
    """COLMAP's ``IMAGE_ID``, the first field of the pose line."""
    name: str
    """``NAME``: the path as it appears in ``frames_meta.json``. May contain spaces."""
    world_T_cam: Float64[ndarray, "4 4"]
    """Camera pose in world, i.e. the inverse of COLMAP's stored ``cam_T_world``."""


def parse_colmap_images_text(text: str) -> list[ColmapImage]:
    """Parse the contents of a COLMAP ``images.txt`` into pose records.

    The format writes **two** lines per image: a pose line, then that image's
    2D observations. An image that registered no points still gets its second
    line, and that line is empty. Dropping empty lines before pairing therefore
    shifts every later image onto an observation line, which silently discards
    it; the lines are consumed in strict pairs instead, and only the header
    comments are skipped.

    Args:
        text: The whole file, header comments included.

    Returns:
        One `ColmapImage` per pose line, in file order.
    """
    lines: list[str] = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    # Padding between the header and the first record would offset the pairing;
    # blank lines only carry meaning once the records have started.
    while lines and not lines[0].strip():
        lines.pop(0)

    images: list[ColmapImage] = []
    for index in range(0, len(lines), 2):  # lines[index + 1] holds the observations
        fields: list[str] = lines[index].split()
        if len(fields) < POSE_LINE_FIELD_COUNT:  # trailing blank line, or a truncated record
            continue
        quaternion_wxyz: Float64[ndarray, "4"] = np.asarray(fields[1:5], dtype=np.float64)
        cam_R_world: Float64[ndarray, "3 3"] = Rotation.from_quat(
            np.roll(quaternion_wxyz, -1)  # COLMAP stores wxyz; scipy wants xyzw
        ).as_matrix()
        cam_T_world: Float64[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
        cam_T_world[:3, :3] = cam_R_world
        cam_T_world[:3, 3] = np.asarray(fields[5:8], dtype=np.float64)
        images.append(
            ColmapImage(
                image_id=int(fields[0]),
                name=" ".join(fields[9:]),
                world_T_cam=np.linalg.inv(cam_T_world),
            )
        )
    return images


def parse_colmap_points_text(text: str) -> tuple[Float64[ndarray, "n_points 3"], UInt8[ndarray, "n_points 3"]]:
    """Parse the contents of a COLMAP ``points3D.txt`` into positions and colours.

    One line per point, so no pairing to get wrong; comments and blank lines are
    skipped and a truncated line is ignored.

    Args:
        text: The whole file, header comments included.

    Returns:
        Float64 positions with shape `[n_points, 3]` and uint8 colours with the
        same shape, in file order.
    """
    positions: list[list[float]] = []
    colours: list[list[int]] = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields: list[str] = line.split()
        if len(fields) < POINT_LINE_FIELD_COUNT:
            continue
        positions.append([float(value) for value in fields[1:4]])
        colours.append([int(value) for value in fields[4:7]])
    return (
        np.asarray(positions, dtype=np.float64).reshape(-1, 3),
        np.asarray(colours, dtype=np.uint8).reshape(-1, 3),
    )


def read_colmap_model(sparse_dir: Path) -> ColmapModel:
    """Load a COLMAP sparse model from ``images.txt`` + ``points3D.txt``.

    cuSFM writes text by default (``--export_binary_colmap_files`` is opt-in), so
    a binary model here means the run asked for one; that case is named rather
    than parsed, since reading it would need pycolmap.

    Args:
        sparse_dir: Directory holding ``images.txt`` and, optionally, ``points3D.txt``.

    Returns:
        The poses keyed by image name, and the coloured point cloud.

    Raises:
        NotImplementedError: When the directory holds a binary model instead.
        FileNotFoundError: When it holds no model at all.
    """
    images_path: Path = sparse_dir / "images.txt"
    points_path: Path = sparse_dir / "points3D.txt"
    if not images_path.is_file():
        if (sparse_dir / "images.bin").is_file():
            raise NotImplementedError(
                f"{sparse_dir} holds a binary COLMAP model; re-run cuSFM without "
                "--export_binary_colmap_files, or add pycolmap."
            )
        raise FileNotFoundError(f"no COLMAP model at {sparse_dir}")

    images: list[ColmapImage] = parse_colmap_images_text(images_path.read_text())
    world_T_cam: dict[str, Float64[ndarray, "4 4"]] = {image.name: image.world_T_cam for image in images}
    points_xyz, points_rgb = parse_colmap_points_text(points_path.read_text() if points_path.is_file() else "")

    print(f"[colmap] {len(world_T_cam)} registered images, {len(points_xyz)} points")
    return ColmapModel(world_T_cam=world_T_cam, points_xyz=points_xyz, points_rgb=points_rgb)
