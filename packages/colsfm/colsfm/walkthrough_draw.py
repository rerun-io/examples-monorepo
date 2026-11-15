"""The drawing primitives every walkthrough panel shares.

Geometry (frusta, camera centres, scene extent, polylines, loop arcs), imagery
(decode and downscale, keypoint conversion, where an image lives) and the
Markdown notes document each panel closes with. Nothing here knows what a stage
*means* — it takes numbers and produces the shapes and the text a panel logs, so
a change to how the walkthrough looks does not touch what it reports.
"""


from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pycolmap
import rerun as rr
from jaxtyping import Float, Float64, Int64, UInt8
from numpy import ndarray
from PIL import Image as PillowImage

from colsfm.cameras import colmap_camera
from colsfm.frames_meta import FramesMeta, KeyframeMeta
from colsfm.walkthrough_stages import (
    ARC_LIFT_FRACTION,
    ARC_POINTS,
    MarkdownRows,
    StageCard,
    WalkthroughStage,
)

# ══════════════════════════════════════════════════════════════════════════════════════
# geometry and imagery helpers
# ══════════════════════════════════════════════════════════════════════════════════════


def transform_points(pose: pycolmap.Rigid3d, points_xyz: Float64[ndarray, "n 3"]) -> Float64[ndarray, "n 3"]:
    """Apply a rigid transform to a batch of points.

    Args:
        pose: The transform, read as `dst_T_src`.
        points_xyz: Float64 points with shape `[n, 3]`, in the source frame.

    Returns:
        Float64 points with shape `[n, 3]`, in the destination frame.
    """
    matrix: Float64[ndarray, "3 4"] = np.asarray(pose.matrix(), dtype=np.float64)
    return points_xyz @ matrix[:3, :3].T + matrix[:3, 3]


def frustum_strips(world_T_cam: pycolmap.Rigid3d, camera: pycolmap.Camera, depth: float) -> list[Float64[ndarray, "n 3"]]:
    """The two polylines that draw one camera as a pyramid in the world frame.

    The image plane is placed at `depth` metres and its corners are back-projected
    through the camera's own calibration matrix, so a wide camera draws a wide
    frustum.

    Args:
        world_T_cam: The camera's pose in the world frame.
        camera: The COLMAP camera, for its pixel size and calibration matrix.
        depth: Distance to the drawn image plane, in metres.

    Returns:
        Two float64 polylines: the image rectangle, and the rays from the centre
        of projection to its four corners.
    """
    calibration: Float64[ndarray, "3 3"] = np.asarray(camera.calibration_matrix(), dtype=np.float64)
    focal_x: float = float(calibration[0, 0])
    focal_y: float = float(calibration[1, 1])
    principal_x: float = float(calibration[0, 2])
    principal_y: float = float(calibration[1, 2])
    corners_uv: Float64[ndarray, "4 2"] = np.array(
        [[0.0, 0.0], [float(camera.width), 0.0], [float(camera.width), float(camera.height)], [0.0, float(camera.height)]],
        dtype=np.float64,
    )
    corners_cam: Float64[ndarray, "4 3"] = np.stack(
        [
            depth * (corners_uv[:, 0] - principal_x) / focal_x,
            depth * (corners_uv[:, 1] - principal_y) / focal_y,
            np.full(4, depth, dtype=np.float64),
        ],
        axis=-1,
    )
    apex_and_corners: Float64[ndarray, "5 3"] = transform_points(
        world_T_cam, np.vstack([np.zeros((1, 3), dtype=np.float64), corners_cam])
    )
    apex: Float64[ndarray, "3"] = apex_and_corners[0]
    corners: Float64[ndarray, "4 3"] = apex_and_corners[1:]
    rectangle: Float64[ndarray, "5 3"] = np.vstack([corners, corners[:1]])
    rays: Float64[ndarray, "8 3"] = np.stack([np.tile(apex, (4, 1)), corners], axis=1).reshape(-1, 3)
    return [rectangle, rays]


def camera_frusta(
    frames_meta: FramesMeta, keyframes: Sequence[KeyframeMeta], depth: float
) -> list[Float64[ndarray, "n 3"]]:
    """Draw every keyframe of a collection as a frustum.

    Args:
        frames_meta: The collection the keyframes belong to, for its calibration.
        keyframes: The keyframes to draw.
        depth: Frustum depth in metres.

    Returns:
        Two polylines per keyframe, in keyframe order.
    """
    cameras: dict[int, pycolmap.Camera] = {
        camera_params_id: colmap_camera(camera) for camera_params_id, camera in frames_meta.cameras.items()
    }
    strips: list[Float64[ndarray, "n 3"]] = []
    for keyframe in keyframes:
        strips.extend(frustum_strips(keyframe.world_T_cam, cameras[keyframe.camera_params_id], depth))
    return strips


def camera_centres(keyframes: Sequence[KeyframeMeta]) -> Float64[ndarray, "n 3"]:
    """Every keyframe's centre of projection in the world frame.

    Args:
        keyframes: The keyframes to read.

    Returns:
        Float64 positions with shape `[n, 3]`, in metres.
    """
    return np.asarray([keyframe.world_T_cam.translation for keyframe in keyframes], dtype=np.float64).reshape(-1, 3)


def scene_extent(positions: Float64[ndarray, "n 3"]) -> float:
    """The diagonal of a point set's bounding box, as a scale for radii and frusta.

    Args:
        positions: Float64 positions with shape `[n, 3]`, in metres.

    Returns:
        The diagonal in metres, or 1.0 for fewer than two distinct positions, so a
        degenerate scene still draws something.
    """
    if len(positions) == 0:
        return 1.0
    diagonal: float = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
    return diagonal if diagonal > 0.0 else 1.0


def polyline_segments(positions: Float64[ndarray, "n 3"]) -> Float64[ndarray, "m 2 3"]:
    """Turn a path into the two-point segments a `LineStrips3D` draws it with.

    Args:
        positions: Float64 path positions with shape `[n, 3]`.

    Returns:
        Float64 segments with shape `[n - 1, 2, 3]`; empty for fewer than two poses.
    """
    if len(positions) < 2:
        return np.zeros((0, 2, 3), dtype=np.float64)
    return np.stack([positions[:-1], positions[1:]], axis=1)


def arc_strip(start_xyz: Float64[ndarray, "3"], end_xyz: Float64[ndarray, "3"]) -> Float64[ndarray, "n 3"]:
    """A polyline from `start` to `end` bulging upwards, so overlapping links stay readable.

    Args:
        start_xyz: Float64 start position with shape `[3]`.
        end_xyz: Float64 end position with shape `[3]`.

    Returns:
        Float64 polyline with shape `[ARC_POINTS, 3]`.
    """
    fraction: Float64[ndarray, "n"] = np.linspace(0.0, 1.0, ARC_POINTS)
    chord: Float64[ndarray, "n 3"] = start_xyz + (end_xyz - start_xyz) * fraction[:, None]
    lift: float = ARC_LIFT_FRACTION * float(np.linalg.norm(end_xyz - start_xyz))
    chord[:, 2] += lift * np.sin(np.pi * fraction)
    return chord


def load_rgb(path: Path, max_width: int) -> tuple[UInt8[ndarray, "h w 3"], float]:
    """Read one JPEG as RGB, scaled down to fit a width budget.

    Args:
        path: The image file.
        max_width: Largest width to return, in pixels.

    Returns:
        The uint8 RGB image and the scale factor that was applied to it.
    """
    with PillowImage.open(path) as opened:
        image = opened.convert("RGB")
        scale: float = min(1.0, max_width / float(image.width))
        if scale < 1.0:
            image = image.resize((int(image.width * scale), int(image.height * scale)), PillowImage.BILINEAR)
        return np.asarray(image, dtype=np.uint8), scale


def rerun_keypoints(keypoints_xy: Float[ndarray, "n 2"]) -> Float64[ndarray, "n 2"]:
    """Convert COLMAP keypoints to Rerun's pixel convention.

    COLMAP puts the origin at the top-left *corner* of the top-left pixel and
    reports the pixel *centre* at (0.5, 0.5); Rerun puts the centre of that pixel
    at (0, 0). The half-pixel is the whole difference.

    Args:
        keypoints_xy: Keypoint pixel coordinates with shape `[n, 2]`.

    Returns:
        Float64 coordinates with shape `[n, 2]`, in Rerun's convention.
    """
    return np.asarray(keypoints_xy, dtype=np.float64).reshape(-1, 2) - 0.5


def image_path_of(input_dir: Path, keyframe: KeyframeMeta) -> Path:
    """Where one keyframe's JPEG lives.

    Args:
        input_dir: The dataset's raw image root.
        keyframe: The keyframe naming it.

    Returns:
        `<input_dir>/<image_name>`.
    """
    return input_dir / keyframe.image_name


def spread_indices(count: int, wanted: int) -> list[int]:
    """Pick evenly spaced indices out of a sequence.

    Args:
        count: Length of the sequence to sample.
        wanted: How many indices to return.

    Returns:
        Ascending, deduplicated indices; empty when `count` is 0.
    """
    if count <= 0 or wanted <= 0:
        return []
    picks: Int64[ndarray, "k"] = np.unique(np.linspace(0, count - 1, min(wanted, count)).round().astype(np.int64))
    return [int(index) for index in picks]


# ══════════════════════════════════════════════════════════════════════════════════════
# the notes document
# ══════════════════════════════════════════════════════════════════════════════════════


def render_notes(stage: WalkthroughStage, rows: MarkdownRows, console: str, extra: str = "") -> str:
    """Render one stage's notes document.

    Args:
        stage: The stage being documented.
        rows: Name/value pairs measured on this run.
        console: Whatever the stage's own functions printed while they ran.
        extra: Further markdown appended after the console block; a stage with a
            table of its own puts it here rather than inside the code fence.

    Returns:
        Markdown holding the prose, the run's numbers, the stage's own log and
        whatever extra section it asked for.
    """
    card: StageCard = stage.card
    lines: list[str] = [
        f"# {card.title}",
        "",
        f"**Computes** — {card.computes}",
        "",
        f"**Replaces the cuSFM binary** — {card.replaces}",
        "",
        f"**Implemented by** — {card.module}",
        "",
        f"**Paper Figure 2** — {card.figure}",
        "",
        "## This run",
        "",
        "| quantity | value |",
        "| --- | --- |",
    ]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    stripped: str = console.strip()
    if stripped:
        lines.extend(["", "## What the stage printed", "", "```", stripped, "```"])
    if extra:
        lines.extend(["", extra])
    return "\n".join(lines) + "\n"


def log_notes(stage: WalkthroughStage, rows: MarkdownRows, console: str, extra: str = "") -> None:
    """Log one stage's notes document at its own stage index.

    Args:
        stage: The stage being documented.
        rows: Name/value pairs measured on this run.
        console: Whatever the stage printed.
        extra: Further markdown appended after the console block.
    """
    rr.log(
        f"{stage.entity_root}/notes",
        rr.TextDocument(render_notes(stage, rows, console, extra), media_type="text/markdown"),
    )


