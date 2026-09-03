"""Build X-Lens model inputs from raw images + camera calibration.

The model's ``forward`` consumes four tensors (all view 0 is the world frame):

    images     (B, S, 3, H, W)   ImageNet-normalized RGB
    d_cam      (B, S, 3, H, W)    per-pixel camera-frame **unit** rays (OpenCV: X right, Y down, Z fwd)
    ray_map    (B, S, 6, H, W)    [ d_world (=R_c2w @ d_cam) ; t_normalized (broadcast) ]
    cam_types  (B, S)             0 = fisheye, 1 = pinhole

The only difference between the three inference modes is *how* ``d_cam`` and
``cam_types`` are produced:

    pinhole        d_cam from the pinhole intrinsics K            cam_type = 1
    fisheye        d_cam from a per-camera calibration LUT        cam_type = 0
    heterogeneous  a mix of the two, one entry per view           cam_type per view

Everything here is pure numpy/torch; no model or training-code dependency.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import numpy as np
import torch
from jaxtyping import Bool, Float, Float32, Int64, UInt8
from torch import Tensor

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class AssembledBatch(TypedDict):
    """One model-ready calibrated multi-view scene."""

    images: Float32[Tensor, "1 views 3 height width"]
    d_cam: Float32[Tensor, "1 views 3 height width"]
    ray_map: Float32[Tensor, "1 views 6 height width"] | None
    cam_types: Int64[Tensor, "1 views"]


# ---------------------------------------------------------------------------
# image normalization
# ---------------------------------------------------------------------------
def normalize_image(img_hwc_uint8: UInt8[np.ndarray, "height width 3"]) -> Float32[np.ndarray, "3 height width"]:
    """(H, W, 3) uint8 RGB -> (3, H, W) float32 ImageNet-normalized."""
    img: Float32[np.ndarray, "height width 3"] = img_hwc_uint8.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# ray_map = [ d_world ; t_normalized ]
# ---------------------------------------------------------------------------
def build_ray_map(
    d_cam: Float32[Tensor, "batch views 3 height width"],
    c2w: Float32[Tensor, "batch views 4 4"],
    view_mask: Bool[Tensor, "batch views"] | None = None,
) -> Float32[Tensor, "batch views 6 height width"]:
    """Assemble the (B, S, 6, H, W) ray map from d_cam + canonicalized c2w poses.

    channels 0..2 = d_world = R_c2w @ d_cam
    channels 3..5 = t_normalized (broadcast over pixels), where t is normalized
                    by the mean camera baseline (mean ||t_i|| over valid views i>0),
                    so pinhole and fisheye views share a consistent t magnitude.

    Args:
        d_cam:     (B, S, 3, H, W) unit rays
        c2w:       (B, S, 4, 4) camera-to-world, **already canonicalized** so view0 = I
        view_mask: (B, S) bool valid-view mask (None -> all valid)
    """
    B, S, _, H, W = d_cam.shape
    device, dtype = d_cam.device, d_cam.dtype
    R: Float32[Tensor, "batch views 3 3"] = c2w[..., :3, :3].to(device, dtype)
    t: Float32[Tensor, "batch views 3"] = c2w[..., :3, 3].to(device, dtype)

    d_world: Float32[Tensor, "batch views 3 height width"] = torch.matmul(R, d_cam.reshape(B, S, 3, H * W)).reshape(B, S, 3, H, W)

    valid_views: Bool[Tensor, "batch views"] = torch.ones(B, S, dtype=torch.bool, device=device) if view_mask is None else view_mask.to(device)
    nonzero: Bool[Tensor, "batch views"] = valid_views & (torch.arange(S, device=device)[None] > 0)
    t_norm: Float32[Tensor, "batch views"] = torch.linalg.norm(t, dim=-1)
    count: Float32[Tensor, "batch"] = nonzero.sum(-1).clamp(min=1).to(dtype)
    pose_scale: Float32[Tensor, "batch"] = ((t_norm * nonzero).sum(-1) / count).clamp(min=1e-6)
    t_n: Float32[Tensor, "batch views 3 height width"] = (t / pose_scale[:, None, None]).reshape(B, S, 3, 1, 1).expand(B, S, 3, H, W)

    return torch.cat([d_world, t_n.contiguous()], dim=2)  # (B, S, 6, H, W)


def canonicalize_c2w(c2w: Float32[Tensor, "batch views 4 4"]) -> Float32[Tensor, "batch views 4 4"]:
    """Canonicalize a (B, S, 4, 4) c2w stack so view 0 is the world frame (view0 = I)."""
    return torch.linalg.inv(c2w[:, :1]) @ c2w


def build_cam_types(types: Sequence[int], device: torch.device | None = None) -> Int64[Tensor, "1 views"]:
    """(S,) list of {0 fisheye, 1 pinhole} -> (1, S) long tensor."""
    return torch.tensor(list(types), dtype=torch.long, device=device)[None]


# ---------------------------------------------------------------------------
# convenience: assemble a full batch for one scene (S views)
# ---------------------------------------------------------------------------
def assemble_batch(
    images_hwc: Sequence[UInt8[np.ndarray, "height width 3"]],
    d_cam_hwc: Sequence[Float32[np.ndarray, "height width 3"]],
    cam_types: Sequence[int],
    c2w: Float[np.ndarray, "views 4 4"] | None = None,
    device: torch.device | None = None,
) -> AssembledBatch:
    """Pack one multi-view scene into model-ready tensors (B=1).

    Args:
        images_hwc: list of S (H, W, 3) uint8 RGB, all the same H,W.
        d_cam_hwc:  list of S (H, W, 3) unit-ray fields (pinhole from K or fisheye LUT).
        cam_types:  list of S ints (0 fisheye / 1 pinhole).
        c2w:        optional (S, 4, 4) camera-to-world. If given, a ray_map is built
                    (geometry prior). If None, ray_map is omitted (images-only + d_cam).
    Returns dict with keys: images, d_cam, ray_map (or None), cam_types.
    """
    S = len(images_hwc)
    assert len(d_cam_hwc) == S == len(cam_types)
    images: Float32[Tensor, "1 views 3 height width"] = torch.from_numpy(np.stack([normalize_image(image) for image in images_hwc]))[None]
    d_cam: Float32[Tensor, "1 views 3 height width"] = torch.from_numpy(np.stack([ray_field.transpose(2, 0, 1) for ray_field in d_cam_hwc]))[None]
    images, d_cam = images.to(device), d_cam.to(device)

    ray_map: Float32[Tensor, "1 views 6 height width"] | None = None
    if c2w is not None:
        c2w_t: Float32[Tensor, "1 views 4 4"] = canonicalize_c2w(torch.from_numpy(np.asarray(c2w, np.float32))[None].to(device))
        ray_map = build_ray_map(d_cam, c2w_t)
    return {
        "images": images,
        "d_cam": d_cam,
        "ray_map": ray_map,
        "cam_types": build_cam_types(cam_types, device=device),
    }
