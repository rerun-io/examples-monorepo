"""Dependency-light, depth-aware Goliath40 rasterization."""

from __future__ import annotations

import math

import cv2
import numpy as np

from fdanyone.config import INFERENCE, SKELETON
from fdanyone.skeleton.keypoints import EXTRA_KEYPOINT_IDS, LINKS, VISIBLE_KEYPOINT_IDS, keypoint_color

_BODY_SCALE_SEGMENTS = (
    (("left-shoulder",), ("right-shoulder",), 0.26),
    (("left-hip",), ("right-hip",), 0.18),
    (("left-shoulder", "right-shoulder"), ("left-hip", "right-hip"), 0.32),
    (("left-shoulder",), ("left-hip",), 0.32),
    (("right-shoulder",), ("right-hip",), 0.32),
    (("left-shoulder",), ("left-elbow",), 0.19),
    (("right-shoulder",), ("right-elbow",), 0.19),
    (("left-elbow",), ("left-wrist",), 0.16),
    (("right-elbow",), ("right-wrist",), 0.16),
    (("left-hip",), ("left-knee",), 0.245),
    (("right-hip",), ("right-knee",), 0.245),
    (("left-knee",), ("left-ankle",), 0.245),
    (("right-knee",), ("right-ankle",), 0.245),
)
_BODY_CENTER_NAMES = ("left-shoulder", "right-shoulder", "left-hip", "right-hip")


def _name_index(names) -> dict[str, int]:
    normalized = [str(name).strip().lower().replace("_", "-") for name in names]
    mapping = dict(zip(normalized, range(len(normalized)), strict=True))
    if len(mapping) != len(normalized):
        raise ValueError("Keypoint names must be unique after normalization.")
    return mapping


def estimate_body_height(keypoints_3d: np.ndarray, names) -> float:
    """Robustly infer physical body height from stable anatomical segments."""

    points = np.asarray(keypoints_3d, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (len(names), 3):
        raise ValueError(f"Expected keypoints [frames,{len(names)},3], got {points.shape}.")
    mapping = _name_index(names)
    required = {name for first, second, _ in _BODY_SCALE_SEGMENTS for group in (first, second) for name in group}
    missing = sorted(required - mapping.keys())
    if missing:
        raise ValueError(f"Missing body-scale keypoints: {missing}.")

    frame_heights = []
    for frame in points:
        candidates = []
        for first, second, height_ratio in _BODY_SCALE_SEGMENTS:
            point_a = frame[[mapping[name] for name in first]].mean(axis=0)
            point_b = frame[[mapping[name] for name in second]].mean(axis=0)
            length = float(np.linalg.norm(point_a - point_b))
            if np.isfinite(length) and length > 0:
                candidates.append(length / height_ratio)
        if candidates:
            frame_heights.append(float(np.median(candidates)))
    heights = np.asarray(frame_heights, dtype=np.float64)
    heights = heights[np.isfinite(heights) & (heights > 0)]
    if not heights.size:
        raise ValueError("Cannot estimate body height from the supplied keypoints.")
    if heights.size >= 5:
        lower, upper = np.percentile(heights, [10.0, 90.0])
        trimmed = heights[(heights >= lower) & (heights <= upper)]
        if trimmed.size:
            heights = trimmed
    return float(np.median(heights))


def projected_body_scales(
    keypoint_depths: np.ndarray,
    names,
    body_height_3d: float,
    focal_px: float,
) -> np.ndarray:
    """Convert physical body height and camera depth to per-frame pixel scale."""

    depths = np.asarray(keypoint_depths, dtype=np.float64)
    if depths.ndim != 2 or depths.shape[1] != len(names):
        raise ValueError(f"Expected keypoint depths [frames,{len(names)}], got {depths.shape}.")
    if not np.isfinite(body_height_3d) or body_height_3d <= 0:
        raise ValueError("body_height_3d must be positive and finite.")
    if not np.isfinite(focal_px) or focal_px <= 0:
        raise ValueError("focal_px must be positive and finite.")
    mapping = _name_index(names)
    missing = [name for name in _BODY_CENTER_NAMES if name not in mapping]
    if missing:
        raise ValueError(f"Missing body-center keypoints: {missing}.")
    center_depths = depths[:, [mapping[name] for name in _BODY_CENTER_NAMES]]
    valid = np.isfinite(center_depths) & (center_depths > 1e-6)
    counts = valid.sum(axis=1)
    mean_depths = np.full(depths.shape[0], np.nan, dtype=np.float64)
    enough = counts >= 2
    mean_depths[enough] = np.where(valid, center_depths, 0.0).sum(axis=1)[enough] / counts[enough]
    scales = np.full(depths.shape[0], np.nan, dtype=np.float32)
    scales[enough] = (body_height_3d * focal_px / mean_depths[enough]).astype(np.float32)
    return scales


def _draw_point(z_buffer, canvas, point, depth, radius, color) -> None:
    if not np.isfinite(depth) or depth <= 0:
        return
    height, width = z_buffer.shape
    x, y = point
    radius = max(1, int(radius))
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
    view = z_buffer[y0:y1, x0:x1]
    update = mask & (depth <= view)
    view[update] = depth
    canvas[y0:y1, x0:x1][update] = color


def _draw_line(z_buffer, canvas, p1, p2, d1, d2, thickness, color) -> None:
    if not np.isfinite(d1) or not np.isfinite(d2) or max(d1, d2) <= 0:
        return
    height, width = z_buffer.shape
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = float(x2 - x1), float(y2 - y1)
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-6:
        _draw_point(z_buffer, canvas, p1, (d1 + d2) / 2.0, max(1, thickness // 2), color)
        return
    radius = max(0.5, float(thickness) / 2.0)
    pad = int(math.ceil(radius)) + 1
    x0, x3 = max(0, min(x1, x2) - pad), min(width, max(x1, x2) + pad + 1)
    y0, y3 = max(0, min(y1, y2) - pad), min(height, max(y1, y2) + pad + 1)
    if x0 >= x3 or y0 >= y3:
        return
    yy, xx = np.ogrid[y0:y3, x0:x3]
    t = np.clip(((xx - x1) * dx + (yy - y1) * dy) / length_sq, 0.0, 1.0)
    mask = (xx - (x1 + t * dx)) ** 2 + (yy - (y1 + t * dy)) ** 2 <= radius**2
    depth = d1 + t * (d2 - d1)
    view = z_buffer[y0:y3, x0:x3]
    update = mask & (depth > 0) & (depth <= view)
    view[update] = depth[update]
    canvas[y0:y3, x0:x3][update] = color


def render_goliath40(
    keypoints: np.ndarray,
    depths: np.ndarray,
    scores: np.ndarray,
    *,
    canvas_height: int,
    canvas_width: int,
    output_height: int,
    output_width: int,
    score_threshold: float = 0.3,
    body_scale_px: float | None = None,
) -> np.ndarray:
    """Render one RGB frame using the reference Sapiens2 sizing rules."""

    canvas_scale = max(1.0, INFERENCE.skeleton_max_dimension / max(output_height, output_width))
    render_height = int(round(output_height * canvas_scale))
    render_width = int(round(output_width * canvas_scale))
    points = np.asarray(keypoints, dtype=np.float32).copy()
    points[:, 0] *= render_width / canvas_width
    points[:, 1] *= render_height / canvas_height
    depths = np.asarray(depths, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    # Keep the reference renderer's BGR draw -> resize -> RGB conversion order.
    # Resizing a channel-permuted uint8 image can differ by one LSB in OpenCV's
    # optimized interpolation kernels, so drawing directly in RGB is not quite
    # byte-exact even though the colors are semantically identical.
    canvas = np.zeros((render_height, render_width, 3), dtype=np.uint8)
    z_buffer = np.full((render_height, render_width), np.inf, dtype=np.float32)
    line_scale = render_height / 1024.0
    if body_scale_px is not None and np.isfinite(body_scale_px) and body_scale_px > 0:
        keypoint_scale = render_height / canvas_height
        line_scale = max(
            0.25,
            float(body_scale_px) * keypoint_scale / SKELETON.draw_body_reference_px,
        )
    base_radius = max(1, int(round(2 * line_scale)))
    base_thickness = max(1, int(round(2 * line_scale)))
    point_items: dict[int, tuple[tuple[int, int], float, int, tuple[int, int, int]]] = {}

    def remember(index: int, radius: int) -> None:
        if scores[index] < score_threshold or not np.isfinite(points[index]).all():
            return
        item = (
            (int(round(points[index, 0])), int(round(points[index, 1]))),
            float(depths[index]),
            radius,
            keypoint_color(index)[::-1],
        )
        if index not in point_items or radius > point_items[index][2]:
            point_items[index] = item

    for _, first, second, color, major in LINKS:
        if min(float(scores[first]), float(scores[second])) < score_threshold:
            continue
        if not np.isfinite(points[[first, second]]).all():
            continue
        p1 = (int(round(points[first, 0])), int(round(points[first, 1])))
        p2 = (int(round(points[second, 0])), int(round(points[second, 1])))
        thickness = base_thickness * (2 if major else 1)
        _draw_line(
            z_buffer,
            canvas,
            p1,
            p2,
            float(depths[first]),
            float(depths[second]),
            thickness,
            color[::-1],
        )
        radius = max(1, int(round(base_radius * (1.75 if major else 1.0))))
        remember(first, radius)
        remember(second, radius)

    for index in EXTRA_KEYPOINT_IDS:
        remember(index, max(2, int(round(base_radius * 1.5))))
    for index, (point, depth, radius, color) in point_items.items():
        if index in VISIBLE_KEYPOINT_IDS:
            _draw_point(z_buffer, canvas, point, depth, radius, color)
    if (render_height, render_width) != (output_height, output_width):
        canvas = cv2.resize(canvas, (output_width, output_height), interpolation=cv2.INTER_AREA)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(canvas)
