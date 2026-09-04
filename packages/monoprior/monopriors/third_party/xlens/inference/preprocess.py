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

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# camera-type ids (must match training)
CAM_TYPE_FISHEYE = 0
CAM_TYPE_PINHOLE = 1


# ---------------------------------------------------------------------------
# image normalization
# ---------------------------------------------------------------------------
def normalize_image(img_hwc_uint8: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 RGB -> (3, H, W) float32 ImageNet-normalized."""
    img = img_hwc_uint8.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1).astype(np.float32)


def denormalize_image(img_chw: np.ndarray) -> np.ndarray:
    """(3, H, W) ImageNet-normalized -> (H, W, 3) uint8 RGB."""
    img = img_chw.transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# per-pixel camera-frame unit rays (d_cam)
# ---------------------------------------------------------------------------
def pinhole_d_cam(K: np.ndarray, h: int, w: int) -> np.ndarray:
    """Per-pixel unit viewing directions (H, W, 3) for a pinhole camera.

    OpenCV convention (X right, Y down, Z forward). Pixel centers (+0.5).
        d = normalize( K^{-1} @ [u+0.5, v+0.5, 1] )
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    us = (np.arange(w, dtype=np.float32) + 0.5 - cx) / fx
    vs = (np.arange(h, dtype=np.float32) + 0.5 - cy) / fy
    uu, vv = np.meshgrid(us, vs)
    dirs = np.stack([uu, vv, np.ones_like(uu)], axis=-1)
    dirs /= np.maximum(np.linalg.norm(dirs, axis=-1, keepdims=True), 1e-6)
    return dirs.astype(np.float32)


def load_fisheye_lut(lut_path: str, h: int, w: int) -> np.ndarray:
    """Load a fisheye calibration LUT of per-pixel unit rays -> (H, W, 3) float32.

    A LUT is the standard way to feed an arbitrary (fisheye / omnidirectional)
    camera model to X-Lens: for every pixel it stores the OpenCV
    camera-frame **unit** viewing direction. Precompute one per physical camera
    from your calibration (OpenCV ``cv2.fisheye``, Kannala-Brandt, Mei, etc.).

    Supported files:
        * ``.npy``  float32 array of shape (H, W, 3)
        * ``.exr``  3-channel image (requires ``imageio`` + OpenEXR)

    The LUT resolution must match the image resolution you feed the model; use
    :func:`resize_d_cam` if you resize the images.
    """
    p = Path(lut_path)
    if p.suffix == ".npy":
        lut = np.load(p).astype(np.float32)
    elif p.suffix in (".exr", ".hdr"):
        import imageio.v2 as imageio  # optional dependency
        lut = np.asarray(imageio.imread(p, format="EXR-FI"))[..., :3].astype(np.float32)
    else:
        raise ValueError(f"unsupported LUT format: {p.suffix} (use .npy or .exr)")
    assert lut.shape[:2] == (h, w), f"LUT {lut.shape} != image ({h},{w}); resize the LUT first"
    lut /= np.maximum(np.linalg.norm(lut, axis=-1, keepdims=True), 1e-6)
    return lut


def resize_d_cam(d_cam_hwc: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Nearest-resize a (H, W, 3) unit-ray field and re-normalize."""
    t = torch.from_numpy(d_cam_hwc).permute(2, 0, 1)[None]
    t = F.interpolate(t, size=(out_h, out_w), mode="nearest")[0].permute(1, 2, 0).numpy()
    t = t / np.maximum(np.linalg.norm(t, axis=-1, keepdims=True), 1e-6)
    return t.astype(np.float32)


# ---------------------------------------------------------------------------
# ray_map = [ d_world ; t_normalized ]
# ---------------------------------------------------------------------------
def build_ray_map(d_cam: torch.Tensor, c2w: torch.Tensor,
                  view_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
    R = c2w[..., :3, :3].to(device, dtype)
    t = c2w[..., :3, 3].to(device, dtype)                                     # (B, S, 3)

    d_world = torch.matmul(R, d_cam.reshape(B, S, 3, H * W)).reshape(B, S, 3, H, W)

    if view_mask is None:
        view_mask = torch.ones(B, S, dtype=torch.bool, device=device)
    nonzero = view_mask.to(device) & (torch.arange(S, device=device)[None] > 0)
    t_norm = torch.linalg.norm(t, dim=-1)                                     # (B, S)
    count = nonzero.sum(-1).clamp(min=1).to(dtype)
    pose_scale = ((t_norm * nonzero).sum(-1) / count).clamp(min=1e-6)          # (B,)
    t_n = (t / pose_scale[:, None, None]).reshape(B, S, 3, 1, 1).expand(B, S, 3, H, W)

    return torch.cat([d_world, t_n.contiguous()], dim=2)                       # (B, S, 6, H, W)


def canonicalize_c2w(c2w: torch.Tensor) -> torch.Tensor:
    """Canonicalize a (B, S, 4, 4) c2w stack so view 0 is the world frame (view0 = I)."""
    return torch.linalg.inv(c2w[:, :1]) @ c2w


def build_cam_types(types: Sequence[int], device=None) -> torch.Tensor:
    """(S,) list of {0 fisheye, 1 pinhole} -> (1, S) long tensor."""
    return torch.tensor(list(types), dtype=torch.long, device=device)[None]


# ---------------------------------------------------------------------------
# convenience: assemble a full batch for one scene (S views)
# ---------------------------------------------------------------------------
def assemble_batch(
    images_hwc: List[np.ndarray],
    d_cam_hwc: List[np.ndarray],
    cam_types: Sequence[int],
    c2w: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
) -> dict:
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
    images = torch.from_numpy(np.stack([normalize_image(im) for im in images_hwc]))[None]   # (1,S,3,H,W)
    d_cam = torch.from_numpy(np.stack([d.transpose(2, 0, 1) for d in d_cam_hwc]))[None]      # (1,S,3,H,W)
    images, d_cam = images.to(device), d_cam.to(device)

    ray_map = None
    if c2w is not None:
        c2w_t = canonicalize_c2w(torch.from_numpy(np.asarray(c2w, np.float32))[None].to(device))
        ray_map = build_ray_map(d_cam, c2w_t)
    return {
        "images": images,
        "d_cam": d_cam,
        "ray_map": ray_map,
        "cam_types": build_cam_types(cam_types, device=device),
    }
