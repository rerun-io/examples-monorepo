"""Unproject predicted depth to a fused world-frame point cloud.

Convention (consistent with training / eval):
    depth  = z-depth in meters
    d_cam  = OpenCV camera-frame unit ray (X right, Y down, Z forward)
    c2w    = view0-canonical camera-to-world

    pt_cam   = (depth / |d_cam.z|) * d_cam_unit      # radial ray * depth-along-z
    pt_world = c2w @ pt_cam                           # works for fisheye AND pinhole
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Bool, Float, Float32, UInt8


def unproject_to_world(
    depth: Float32[np.ndarray, "height width"],
    d_cam: Float32[np.ndarray, "height width 3"],
    c2w: Float[np.ndarray, "4 4"],
) -> Float32[np.ndarray, "height width 3"]:
    """One view: depth (H, W), d_cam (H, W, 3) unit, c2w (4, 4) -> world points (H, W, 3)."""
    z: Float32[np.ndarray, "height width"] = np.clip(np.abs(d_cam[..., 2]), 1e-6, None)
    pt_cam: Float32[np.ndarray, "height width 3"] = (depth / z)[..., None] * d_cam
    R, t = c2w[:3, :3], c2w[:3, 3]
    return pt_cam @ R.T + t


def fuse_point_cloud(
    depth: Float32[np.ndarray, "views height width"],
    d_cam: Float32[np.ndarray, "views height width 3"],
    c2w: Float[np.ndarray, "views 4 4"],
    rgb: UInt8[np.ndarray, "views height width 3"] | None = None,
    conf: Float32[np.ndarray, "views height width"] | None = None,
    conf_thresh: float = 0.0,
    conf_drop_pct: float = 0.0,
    masks: Bool[np.ndarray, "views height width"] | None = None,
    max_depth: float | None = None,
    fov_max_deg: float | None = None,
) -> tuple[Float32[np.ndarray, "points 3"], UInt8[np.ndarray, "points 3"] | None]:
    """Fuse all S views into a single (N, 3) world point cloud (+ optional colors).

    Args:
        depth: (S, H, W)         predicted metric depth
        d_cam: (S, H, W, 3)      per-pixel unit rays (fisheye LUT edge = (0,0,0) -> masked out)
        c2w:   (S, 4, 4)         view0-canonical poses
        rgb:   (S, H, W, 3)      uint8 colors (optional)
        conf:  (S, H, W)         confidence (optional), pixels below conf_thresh dropped
        conf_drop_pct:           drop the lowest N% of points by confidence (0-100), after other
                                 filters. More intuitive than an absolute threshold; needs `conf`.
        masks: (S, H, W)         bool valid mask per view (optional), e.g. static camera masks
        max_depth:               drop points beyond this **euclidean** range (m) from their camera.
                                 Uses euclidean (not z-) distance so wide-FoV fisheye edge rays,
                                 whose z-depth is tiny but true range is large, are clipped correctly.
        fov_max_deg:             drop pixels whose viewing ray is more than this many degrees off the
                                 optical axis (|d_cam.z| < cos(fov_max_deg)). Trims the extreme
                                 peripheral rays of fisheye lenses, where large distortion makes depth
                                 unreliable and points shoot outside the scene. Recommended: 85 for fisheye.
    Returns:
        points (N, 3) float32, colors (N, 3) uint8 or None
    """
    pts_all: list[Float32[np.ndarray, "points 3"]] = []
    col_all: list[UInt8[np.ndarray, "points 3"]] = []
    conf_all: list[Float32[np.ndarray, "points"]] = []
    for s in range(depth.shape[0]):
        pw: Float32[np.ndarray, "pixels 3"] = unproject_to_world(depth[s], d_cam[s], c2w[s]).reshape(-1, 3)
        valid: Bool[np.ndarray, "pixels"] = np.isfinite(pw).all(-1) & (depth[s].reshape(-1) > 0)
        valid &= np.linalg.norm(d_cam[s].reshape(-1, 3), axis=-1) > 1e-3  # drop LUT edge / null rays
        if fov_max_deg is not None:  # trim high-distortion fisheye periphery
            valid &= np.abs(d_cam[s].reshape(-1, 3)[:, 2]) >= np.cos(np.deg2rad(fov_max_deg))
        if masks is not None:
            valid &= masks[s].reshape(-1).astype(bool)
        if conf is not None and conf_thresh > 0:
            valid &= conf[s].reshape(-1) >= conf_thresh
        if max_depth is not None:
            cam_center = c2w[s][:3, 3]
            valid &= np.linalg.norm(pw - cam_center, axis=-1) <= max_depth
        pts_all.append(pw[valid])
        if rgb is not None:
            col_all.append(rgb[s].reshape(-1, 3)[valid])
        if conf is not None:
            conf_all.append(conf[s].reshape(-1)[valid])
    points: Float32[np.ndarray, "points 3"] = np.concatenate(pts_all, 0).astype(np.float32)
    colors: UInt8[np.ndarray, "points 3"] | None = np.concatenate(col_all, 0).astype(np.uint8) if rgb is not None else None

    # global confidence-percentile drop (keep the most confident points)
    if conf is not None and conf_drop_pct > 0 and len(points):
        c: Float32[np.ndarray, "points"] = np.concatenate(conf_all, 0)
        keep: Bool[np.ndarray, "points"] = c >= np.percentile(c, conf_drop_pct)
        points = points[keep]
        colors = colors[keep] if colors is not None else None
    return points, colors
