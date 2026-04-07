"""Image resizing and normalization utilities for MASt3R input.

Pure image utilities with no dependency on SLAM types (Frame, config, etc.),
extracted here to avoid circular imports between ``frame`` and ``mast3r_utils``.
"""

from typing import Any, Literal

import numpy as np
import PIL
import PIL.Image
from dust3r.utils.image import ImgNorm
from jaxtyping import Float
from numpy import ndarray


def _resize_pil_image(img: PIL.Image.Image, long_edge_size: int) -> PIL.Image.Image:
    """Resize a PIL image so its longest edge matches the given size.

    Uses Lanczos interpolation when downscaling and bicubic when upscaling.

    Args:
        img: The input PIL image.
        long_edge_size: Desired length of the longest edge in pixels.

    Returns:
        The resized PIL image.
    """
    S = max(img.size)
    if long_edge_size < S:
        interp = PIL.Image.Resampling.LANCZOS
    else:
        interp = PIL.Image.Resampling.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def resize_img(
    img: Float[ndarray, "h w 3"],
    size: Literal[224, 512],
    square_ok: bool = False,
    return_transformation: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], tuple[float, float, float, float]]:
    """Resize and normalize an image for MASt3R input.

    Converts a float ``[0, 1]`` HWC numpy image to a PIL image, resizes it to
    the target resolution (224 with center-crop or 512 with padding-friendly
    crop), applies ``ImgNorm``, and returns the processed tensors.

    Args:
        img: Input RGB image in ``[0, 1]`` float range, shape ``(H, W, 3)``.
        size: Target resolution -- ``224`` (center-crop to square) or ``512``
            (resize long edge, crop to 16-pixel-aligned dimensions).
        square_ok: When ``True`` and ``size=512``, allow square output even if
            the resized image is square.
        return_transformation: When ``True``, also return the crop/scale
            parameters needed to map back to the original image coordinates.

    Returns:
        A dict with keys ``"img"`` (normalized ``(1, 3, h, w)`` tensor),
        ``"true_shape"`` (``(1, 2)`` int32 array), and ``"unnormalized_img"``
        (uint8 HWC array). When ``return_transformation`` is ``True``, returns
        a tuple of ``(dict, (scale_w, scale_h, half_crop_w, half_crop_h))``.
    """
    assert size == 224 or size == 512
    # numpy to PIL format
    img = PIL.Image.fromarray(np.uint8(img * 255))
    W1, H1 = img.size
    if size == 224:
        # resize short side to 224 (then crop)
        img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
    else:
        # resize long side to 512
        img = _resize_pil_image(img, size)
    W, H = img.size
    cx, cy = W // 2, H // 2
    if size == 224:
        half = min(cx, cy)
        img = img.crop((cx - half, cy - half, cx + half, cy + half))
    else:
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if not (square_ok) and W == H:
            halfh = 3 * halfw / 4
        img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

    res = dict(
        img=ImgNorm(img)[None],
        true_shape=np.array([img.size[::-1]], dtype=np.int32),
        unnormalized_img=np.asarray(img),
    )
    if return_transformation:
        scale_w = W1 / W
        scale_h = H1 / H
        half_crop_w = (W - img.size[0]) / 2
        half_crop_h = (H - img.size[1]) / 2
        return res, (scale_w, scale_h, half_crop_w, half_crop_h)

    return res
