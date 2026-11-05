# pyrefly: ignore-errors
"""Warp-checked kernels for SAM's affine crops and OpenCV interpolation."""

import warp as wp

wp.set_module_options({"enable_backward": False, "fuse_fp": False})


@wp.kernel
def crop_geometry(
    boxes: wp.array2d(dtype=wp.float32), padding: float,
    centers: wp.array2d(dtype=wp.float32), scales: wp.array2d(dtype=wp.float32),
    affine: wp.array3d(dtype=wp.float32), inverse: wp.array2d(dtype=wp.float64),
):
    n = wp.tid()
    cx = (boxes[n, 0] + boxes[n, 2]) * 0.5
    cy = (boxes[n, 1] + boxes[n, 3]) * 0.5
    width = (boxes[n, 2] - boxes[n, 0]) * padding
    height = (boxes[n, 3] - boxes[n, 1]) * padding
    side = wp.max(width / 0.75, height)
    centers[n, 0] = cx
    centers[n, 1] = cy
    scales[n, 0] = side
    scales[n, 1] = side
    # getAffineTransform uses three float32 source points. Retain their
    # rounding before solving the axis-aligned transform in float64.
    top = float(wp.float64(cy) - wp.float64(side) * wp.float64(0.5))
    half_y = cy - top
    left = cx - half_y
    half_x = cx - left
    sx = wp.float64(256.0) / wp.float64(half_x)
    sy = wp.float64(256.0) / wp.float64(half_y)
    tx = wp.float64(256.0) - wp.float64(cx) * sx
    ty = wp.float64(256.0) - wp.float64(cy) * sy
    affine[n, 0, 0] = float(sx)
    affine[n, 0, 1] = 0.0
    affine[n, 0, 2] = float(tx)
    affine[n, 1, 0] = 0.0
    affine[n, 1, 1] = float(sy)
    affine[n, 1, 2] = float(ty)
    inverse[n, 0] = wp.float64(1.0) / sx
    inverse[n, 1] = wp.float64(1.0) / sy
    inverse[n, 2] = -tx / sx
    inverse[n, 3] = -ty / sy


@wp.kernel
def affine_sample(
    frames: wp.array4d(dtype=wp.uint8), frame_indices: wp.array(dtype=wp.int64),
    inverse: wp.array2d(dtype=wp.float64), mirrored: wp.array(dtype=wp.bool),
    masks: wp.array3d(dtype=wp.float32), has_masks: bool,
    images: wp.array4d(dtype=wp.float32), output_masks: wp.array4d(dtype=wp.float32),
):
    n, y, x = wp.tid()
    height = frames.shape[1]
    width = frames.shape[2]
    frame = int(frame_indices[n])
    # This environment uses OpenCV 5's continuous bilinear interpolation.
    sx = inverse[n, 0] * wp.float64(x) + inverse[n, 2]
    sy = inverse[n, 1] * wp.float64(y) + inverse[n, 3]
    x0 = int(wp.floor(sx))
    y0 = int(wp.floor(sy))
    dx = float(sx - wp.float64(x0))
    dy = float(sy - wp.float64(y0))
    color = wp.vec3(0.0, 0.0, 0.0)
    mask = 0.0
    for iy in range(2):
        for ix in range(2):
            xx = x0 + ix
            yy = y0 + iy
            wx = 1.0 - dx
            wy = 1.0 - dy
            if ix == 1:
                wx = dx
            if iy == 1:
                wy = dy
            weight = wx * wy
            if xx >= 0 and xx < width and yy >= 0 and yy < height:
                source_x = xx
                if mirrored[n]:
                    source_x = width - xx - 1
                for channel in range(3):
                    color[channel] += float(frames[frame, yy, source_x, channel]) * weight
                if has_masks:
                    mask += masks[n, yy, source_x] * weight
    for channel in range(3):
        images[n, channel, y, x] = wp.floor(color[channel] + 0.5) / 255.0
    output_masks[n, 0, y, x] = mask
