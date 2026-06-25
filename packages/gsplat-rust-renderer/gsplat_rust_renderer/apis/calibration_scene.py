"""Camera-calibration scene for cross-renderer validation.

Generates a tiny 3DGS PLY with splats at exactly known world positions plus a
synthesized NeRF-style ``transforms_test.json``, and checks rendered images
against analytically predicted pixel coordinates (pinhole model).

Each marker's color encodes what it verifies:

* origin (white-ish gray), +X (red), +Y (green), +Z (blue) — handedness and the
  raster y-flip
* four corner markers in the z=0 plane — focal length / FOV via inter-blob
  pixel distances
* an overlapping front/back pair on one camera ray (orange in front of purple)
  — depth ordering / alpha compositing

Usage:
    python tools/calibration_scene.py generate --out-dir data/calibration
    python tools/calibration_scene.py check --image render.png \
        --scene-dir data/calibration --renderer gsplat-render
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from jaxtyping import Bool, Float32, Float64, Shaped
from PIL import Image
from plyfile import PlyData, PlyElement

from gsplat_rust_renderer.gaussians3d import SH_C0 as _SH_C0

SH_C0: float = float(_SH_C0)
"""Zeroth spherical-harmonic coefficient (1 / (2 * sqrt(pi))) — shared with the loader."""

CAMERA_ANGLE_X: float = 0.6911112070083618
"""Horizontal FOV in radians — same value as the NeRF synthetic dataset."""

MARKER_SCALE: float = 0.02
"""Isotropic world-space radius of each marker splat (before activation)."""

MARKER_OPACITY: float = 0.99
"""Target opacity of each marker after the sigmoid activation."""


@dataclass(frozen=True)
class Marker:
    """One calibration splat at a known world position."""

    name: str
    """Identifier used in reports (e.g. ``origin``, ``+x``)."""
    position: tuple[float, float, float]
    """World-space center of the splat."""
    rgb: tuple[float, float, float]
    """Activated linear RGB in [0, 1] the renderer should reproduce."""
    occluded: bool = False
    """True for the back splat of the depth probe — must NOT be visible."""


def _depth_probe_markers() -> tuple[Marker, Marker]:
    """Build the front/back depth-probe pair on a single camera ray.

    The back marker (purple) sits at a fixed point; the front marker (orange)
    is placed on the segment from the back marker toward the camera, so both
    project to the same pixel and only the front one may be visible.

    Returns:
        ``(front_marker, back_marker)``.
    """
    cam_pos: Float64[np.ndarray, "3"] = np.array(CAMERA_POSITION, dtype=np.float64)
    back: Float64[np.ndarray, "3"] = np.array([-0.3, 0.3, 0.5], dtype=np.float64)
    front: Float64[np.ndarray, "3"] = back + 0.4 * (cam_pos - back)
    return (
        Marker("depth-front", tuple(front.tolist()), (1.0, 0.55, 0.0)),
        Marker("depth-back", tuple(back.tolist()), (0.55, 0.0, 0.85), occluded=True),
    )


CAMERA_POSITION: tuple[float, float, float] = (2.0, -2.0, 1.2)
"""Calibration camera position (world, Z-up)."""

CAMERA_TARGET: tuple[float, float, float] = (0.0, 0.0, 0.25)
"""Point the calibration camera looks at."""


def calibration_markers() -> list[Marker]:
    """All calibration markers: axes, corners, and the depth probe.

    Returns:
        Markers in drawing-independent order; the depth probe is last.
    """
    front, back = _depth_probe_markers()
    return [
        Marker("origin", (0.0, 0.0, 0.0), (0.85, 0.85, 0.85)),
        Marker("+x", (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        Marker("+y", (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        Marker("+z", (0.0, 0.0, 1.0), (0.0, 0.3, 1.0)),
        Marker("corner-pp", (0.5, 0.5, 0.0), (1.0, 1.0, 0.0)),
        Marker("corner-pm", (0.5, -0.5, 0.0), (0.0, 1.0, 1.0)),
        Marker("corner-mp", (-0.5, 0.5, 0.0), (1.0, 0.0, 1.0)),
        Marker("corner-mm", (-0.5, -0.5, 0.0), (0.3, 0.3, 0.3)),
        front,
        back,
    ]


def look_at_c2w(
    position: Float64[np.ndarray, "3"],
    target: Float64[np.ndarray, "3"],
    world_up: Float64[np.ndarray, "3"],
) -> Float64[np.ndarray, "4 4"]:
    """Build an OpenGL-convention camera-to-world matrix (camera looks down −Z).

    Args:
        position: Camera center in world space.
        target: World point the camera looks at.
        world_up: Approximate up direction used to orthogonalize the frame.

    Returns:
        4×4 c2w with columns ``[right, up, -forward, position]``.
    """
    forward: Float64[np.ndarray, "3"] = target - position
    forward = forward / np.linalg.norm(forward)
    right: Float64[np.ndarray, "3"] = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    true_up: Float64[np.ndarray, "3"] = np.cross(right, forward)

    c2w: Float64[np.ndarray, "4 4"] = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = true_up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = position
    return c2w


def project_to_raster(
    point_world: Float64[np.ndarray, "3"],
    c2w: Float64[np.ndarray, "4 4"],
    camera_angle_x: float,
    width: int,
    height: int,
) -> tuple[float, float, float]:
    """Project a world point to raster pixel coordinates (origin top-left).

    Pinhole model with centered principal point, matching ``gsplat-render``:
    ``u = fx·X/Z + cx`` in a y-up image plane, then one y-flip to raster rows.

    Args:
        point_world: World-space point.
        c2w: Camera-to-world matrix (OpenGL convention).
        camera_angle_x: Horizontal FOV in radians.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        ``(u, v, depth)`` — raster pixel coordinates and positive view depth.
    """
    w2c: Float64[np.ndarray, "4 4"] = np.linalg.inv(c2w)
    view: Float64[np.ndarray, "3"] = (w2c @ np.append(point_world, 1.0))[:3]
    depth: float = float(-view[2])

    fx: float = (width / 2.0) / math.tan(camera_angle_x / 2.0)
    aspect: float = width / height
    fov_y: float = 2.0 * math.atan(math.tan(camera_angle_x / 2.0) / aspect)
    fy: float = (height / 2.0) / math.tan(fov_y / 2.0)

    u: float = fx * float(view[0]) / depth + width / 2.0
    v_up: float = fy * float(view[1]) / depth + height / 2.0
    v_raster: float = height - v_up
    return (u, v_raster, depth)


def _inverse_sigmoid(value: float) -> float:
    """Logit — the inverse of the opacity activation used by 3DGS loaders."""
    return math.log(value / (1.0 - value))


def write_calibration_ply(path: Path, markers: list[Marker]) -> None:
    """Write the markers as a standard 3DGS PLY (inverse activations applied).

    Loaders apply ``exp`` to scales, ``sigmoid`` to opacity, and
    ``SH_C0·dc + 0.5`` to colors — so we store ``log(scale)``,
    ``logit(opacity)``, and ``(rgb − 0.5)/SH_C0``.

    Args:
        path: Output ``.ply`` path.
        markers: Calibration markers to encode.
    """
    n: int = len(markers)
    fields: list[tuple[str, str]] = (
        [(axis, "f4") for axis in ("x", "y", "z")]
        + [(f"n{axis}", "f4") for axis in ("x", "y", "z")]
        + [(f"f_dc_{i}", "f4") for i in range(3)]
        + [("opacity", "f4")]
        + [(f"scale_{i}", "f4") for i in range(3)]
        + [(f"rot_{i}", "f4") for i in range(4)]
    )
    vertex: Shaped[np.ndarray, "n"] = np.zeros(n, dtype=fields)

    log_scale: float = math.log(MARKER_SCALE)
    raw_opacity: float = _inverse_sigmoid(MARKER_OPACITY)
    for i, marker in enumerate(markers):
        vertex["x"][i], vertex["y"][i], vertex["z"][i] = marker.position
        for c in range(3):
            vertex[f"f_dc_{c}"][i] = (marker.rgb[c] - 0.5) / SH_C0
        vertex["opacity"][i] = raw_opacity
        for c in range(3):
            vertex[f"scale_{c}"][i] = log_scale
        vertex["rot_0"][i] = 1.0  # identity quaternion, PLY order wxyz

    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")]).write(str(path))


@dataclass
class GenerateConfig:
    """Generate the calibration PLY, camera JSON, and expected-pixel table."""

    out_dir: Path = Path("data/calibration")
    """Directory receiving calibration.ply / transforms_test.json / expected_pixels.json."""
    width: int = 800
    """Image width used for the precomputed pixel predictions."""
    height: int = 800
    """Image height used for the precomputed pixel predictions."""


def generate(config: GenerateConfig) -> None:
    """Write the calibration scene artifacts.

    Args:
        config: Output directory and prediction resolution.
    """
    markers: list[Marker] = calibration_markers()
    c2w: Float64[np.ndarray, "4 4"] = look_at_c2w(
        np.array(CAMERA_POSITION, dtype=np.float64),
        np.array(CAMERA_TARGET, dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )

    write_calibration_ply(config.out_dir / "calibration.ply", markers)

    transforms: dict = {
        "camera_angle_x": CAMERA_ANGLE_X,
        "frames": [{"file_path": "./test/r_0", "transform_matrix": c2w.tolist()}],
    }
    (config.out_dir / "transforms_test.json").write_text(json.dumps(transforms, indent=2))

    expected: list[dict] = []
    for marker in markers:
        u, v, depth = project_to_raster(
            np.array(marker.position, dtype=np.float64), c2w, CAMERA_ANGLE_X, config.width, config.height
        )
        expected.append(
            {
                "name": marker.name,
                "world": list(marker.position),
                "rgb": list(marker.rgb),
                "occluded": marker.occluded,
                "u": u,
                "v": v,
                "depth": depth,
            }
        )
    payload: dict = {
        "width": config.width,
        "height": config.height,
        "camera_angle_x": CAMERA_ANGLE_X,
        "camera_position": list(CAMERA_POSITION),
        "camera_target": list(CAMERA_TARGET),
        "c2w": c2w.tolist(),
        "markers": expected,
    }
    (config.out_dir / "expected_pixels.json").write_text(json.dumps(payload, indent=2))
    print(f"Wrote calibration scene to {config.out_dir} ({len(markers)} markers)")
    for row in expected:
        flag: str = " (occluded)" if row["occluded"] else ""
        print(f"  {row['name']:>12}: ({row['u']:7.1f}, {row['v']:7.1f}) depth {row['depth']:.2f}{flag}")


@dataclass
class CheckConfig:
    """Check a rendered calibration image against the analytic predictions."""

    image: Path
    """Rendered PNG of the calibration scene."""
    scene_dir: Path = Path("data/calibration")
    """Directory holding expected_pixels.json from `generate`."""
    renderer: str = "unknown"
    """Label for the report (e.g. gsplat-render, brush, rerun-viewer)."""
    tolerance_px: float = 4.0
    """Max allowed centroid error in pixels (after any resolution rescale)."""
    color_tolerance: float = 0.25
    """Per-channel color-mask tolerance in [0, 1]."""
    min_blob_pixels: int = 4
    """Minimum mask size for a marker to count as detected."""
    report_json: Path | None = None
    """Optional path to write a machine-readable result."""


def _detect_blob(
    rgb: Float32[np.ndarray, "h w 3"],
    target_rgb: tuple[float, float, float],
    color_tolerance: float,
) -> tuple[float, float, int] | None:
    """Find the centroid of pixels matching a marker color.

    Args:
        rgb: Image in [0, 1].
        target_rgb: Expected activated marker color.
        color_tolerance: Per-channel tolerance.

    Returns:
        ``(u, v, count)`` centroid and mask size, or ``None`` if not found.
    """
    target: Float32[np.ndarray, "3"] = np.array(target_rgb, dtype=np.float32)
    mask: Bool[np.ndarray, "h w"] = np.all(np.abs(rgb - target) < color_tolerance, axis=-1)
    count: int = int(mask.sum())
    if count == 0:
        return None
    ys, xs = np.nonzero(mask)
    return (float(xs.mean()) + 0.5, float(ys.mean()) + 0.5, count)


def check(config: CheckConfig) -> None:
    """Compare blob centroids in a rendered image against predictions.

    Exits nonzero if any visible marker is missing/misplaced or the occluded
    marker is visible.

    Args:
        config: Image path, scene dir, and tolerances.
    """
    payload: dict = json.loads((config.scene_dir / "expected_pixels.json").read_text())
    image: Image.Image = Image.open(config.image).convert("RGB")
    rgb: Float32[np.ndarray, "h w 3"] = np.asarray(image, dtype=np.float32) / 255.0
    h, w = rgb.shape[:2]
    scale_u: float = w / payload["width"]
    scale_v: float = h / payload["height"]

    results: list[dict] = []
    failures: int = 0
    for marker in payload["markers"]:
        expected_u: float = marker["u"] * scale_u
        expected_v: float = marker["v"] * scale_v
        blob: tuple[float, float, int] | None = _detect_blob(rgb, tuple(marker["rgb"]), config.color_tolerance)

        if marker["occluded"]:
            ok: bool = blob is None or blob[2] < config.min_blob_pixels
            err: float | None = None
        elif blob is None or blob[2] < config.min_blob_pixels:
            ok = False
            err = None
        else:
            err = math.hypot(blob[0] - expected_u, blob[1] - expected_v)
            ok = err <= config.tolerance_px
        failures += 0 if ok else 1
        results.append(
            {
                "name": marker["name"],
                "expected": [expected_u, expected_v],
                "detected": list(blob[:2]) if blob else None,
                "blob_pixels": blob[2] if blob else 0,
                "error_px": err,
                "occluded": marker["occluded"],
                "ok": ok,
            }
        )
        status: str = "PASS" if ok else "FAIL"
        if marker["occluded"]:
            print(f"  [{status}] {marker['name']:>12}: occlusion (visible pixels: {blob[2] if blob else 0})")
        else:
            err_text: str = f"{err:.2f}px" if err is not None else "not detected"
            print(f"  [{status}] {marker['name']:>12}: expected ({expected_u:7.1f},{expected_v:7.1f}) error {err_text}")

    verdict: str = "PASS" if failures == 0 else f"FAIL ({failures} markers)"
    print(f"calibration check [{config.renderer}] on {config.image.name}: {verdict}")

    if config.report_json is not None:
        config.report_json.parent.mkdir(parents=True, exist_ok=True)
        config.report_json.write_text(
            json.dumps({"renderer": config.renderer, "image": str(config.image), "pass": failures == 0, "markers": results}, indent=2)
        )
    raise SystemExit(0 if failures == 0 else 1)


def main() -> None:
    """CLI entry point with `generate` and `check` subcommands."""
    command = tyro.extras.subcommand_cli_from_dict({"generate": GenerateConfig, "check": CheckConfig})
    if isinstance(command, GenerateConfig):
        generate(command)
    else:
        check(command)
