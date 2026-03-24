import os
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import gsplat as gs
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float
from numpy import ndarray
from plyfile import PlyData, PlyElement
from torch import Tensor

from vistadream.ops.utils import (
    alpha_inpaint_mask,
    dpt2xyz,
    numpy_normalize,
    numpy_quaternion_from_matrix,
    transform_points,
)


@dataclass
class Frame:
    """
    rgb: in shape of H*W*3, in range of 0-1
    dpt: in shape of H*W, real depth
    inpaint: bool mask in shape of H*W for inpainting
    intrinsic: 3*3
    world_T_cam: array in shape of 4*4

    As a class for:
    initialize camera
    accept rendering result
    accept inpainting result
    All at 2D-domain
    """

    H: int | None = None
    W: int | None = None
    rgb: np.ndarray | None = None
    dpt: np.ndarray | None = None
    inpaint: Bool[ndarray, "H W"] | None = None
    inpaint_wo_edge: Bool[ndarray, "H W"] | None = None
    dpt_conf_mask: Bool[ndarray, "H W"] | None = None
    intrinsic: Float[ndarray, "3 3"] | None = None
    cam_T_world: Float[ndarray, "4 4"] | None = None
    ideal_dpt: Float[ndarray, "H W"] | None = None
    ideal_nml: Float[ndarray, "H W 3"] | None = None
    keep: bool = False  # for keep supervision

    def __post_init__(self):
        self._rgb_rect()
        self._extr_rect()

    def _rgb_rect(self):
        if self.rgb is not None:
            if isinstance(self.rgb, PIL.PngImagePlugin.PngImageFile):
                self.rgb = np.array(self.rgb)
            if isinstance(self.rgb, PIL.JpegImagePlugin.JpegImageFile):
                self.rgb = np.array(self.rgb)
            if np.amax(self.rgb) > 1.1:
                self.rgb = self.rgb / 255

    def _extr_rect(self):
        if self.cam_T_world is None:
            self.cam_T_world = np.eye(4)
        self.world_T_cam: Float[ndarray, "4 4"] = np.linalg.inv(self.cam_T_world)


@dataclass
class Gaussian_Frame:
    """
    In-frame-frustrum
    Gaussians from a single RGBD frame
    As a class for:
    accept information from initialized/inpainting+geo-estimated frame
    saving pixelsplat properties including rgb, xyz, scale, rotation, opacity; note here, we made a modification to xyz;
    we first project depth to xyz
    then we tune a scale map(initialized to ones) and a shift map(initialized to zeros), they are optimized and add to the original xyz when rendering
    """

    # as pixelsplat guassian
    rgb: Float[Tensor, "n_splats 3"] | None = None
    scale: Float[Tensor, "n_splats 3"] | None = None
    opacity: Float[Tensor, "n_splats 1"] | None = None
    rotation: Float[Tensor, "n_splats 4"] | None = None
    # gaussian center
    dpt: Float[ndarray, "H W"] | None = None
    xyz: Float[Tensor, "n_splats 3"] | None = None
    # as a frame
    H: int = 480
    W: int = 640

    def __init__(self, frame: Frame, device: Literal["cuda", "cpu"] = "cuda") -> None:
        """after inpainting"""
        # de-active functions
        self.rgbs_deact = torch.logit
        self.scales_deact = torch.log
        self.opacity_deact = torch.logit
        self.device = device
        # for gaussian initialization
        self._set_property_from_frame(frame)

    def _to_3d(self) -> Float[np.ndarray, "H W 3"]:
        # inv intrinsic
        xyz = dpt2xyz(self.dpt, self.intrinsic)
        inv_extrinsic = np.linalg.inv(self.world_T_cam)
        xyz = transform_points(xyz, inv_extrinsic)
        return xyz

    def _paint_filter(self, paint_mask: Bool[ndarray, "H W"]) -> None:
        """
        Applies a boolean mask to filter the object's attributes.

        If the number of True values in `paint_mask` is less than 3, a default mask is applied to ensure at least one element is selected.
        The method then filters the `rgb`, `xyz`, `scale`, `opacity`, and `rotation` attributes using the provided or default mask.

        Args:
            paint_mask (Bool[ndarray, "H W"]): A boolean mask indicating which elements to keep.

        Returns:
            None
        """
        if np.sum(paint_mask) < 3:
            paint_mask = np.zeros((self.H, self.W))
            paint_mask[0:1] = 1
            paint_mask = paint_mask > 0.5
        self.rgb = self.rgb[paint_mask]
        self.xyz = self.xyz[paint_mask]
        self.scale = self.scale[paint_mask]
        self.opacity = self.opacity[paint_mask]
        self.rotation = self.rotation[paint_mask]

    def _to_cuda(self) -> None:
        self.rgb = torch.from_numpy(self.rgb.astype(np.float32)).to(self.device)
        self.xyz = torch.from_numpy(self.xyz.astype(np.float32)).to(self.device)
        self.scale = torch.from_numpy(self.scale.astype(np.float32)).to(self.device)
        self.opacity = torch.from_numpy(self.opacity.astype(np.float32)).to(self.device)
        self.rotation = torch.from_numpy(self.rotation.astype(np.float32)).to(self.device)

    def _fine_init_scale_rotations(self):
        # from https://arxiv.org/pdf/2406.09394
        """Compute rotation matrices that align z-axis with given normal vectors using matrix operations."""
        up_axis = np.array([0, 1, 0])
        nml = self.nml @ self.world_T_cam[0:3, 0:3]
        qz = numpy_normalize(nml)
        qx = np.cross(up_axis, qz)
        qx = numpy_normalize(qx)
        qy = np.cross(qz, qx)
        qy = numpy_normalize(qy)
        rot = np.concatenate([qx[..., None], qy[..., None], qz[..., None]], axis=-1)
        self.rotation = numpy_quaternion_from_matrix(rot)
        # scale
        safe_nml = deepcopy(self.nml)
        safe_nml[safe_nml[:, :, -1] < 0.2, -1] = 0.2
        normal_xoz = deepcopy(safe_nml)
        normal_yoz = deepcopy(safe_nml)
        normal_xoz[..., 1] = 0.0
        normal_yoz[..., 0] = 0.0
        normal_xoz = numpy_normalize(normal_xoz)
        normal_yoz = numpy_normalize(normal_yoz)
        cos_theta_x = np.abs(normal_xoz[..., 2])
        cos_theta_y = np.abs(normal_yoz[..., 2])
        scale_basic = self.dpt / self.intrinsic[0, 0] / np.sqrt(2)
        scale_x = scale_basic / cos_theta_x
        scale_y = scale_basic / cos_theta_y
        scale_z = (scale_x + scale_y) / 10.0
        self.scale = np.concatenate([scale_x[..., None], scale_y[..., None], scale_z[..., None]], axis=-1)

    def _coarse_init_scale_rotations(self):
        # gaussian property -- HW3 scale
        self.scale = self.dpt / self.intrinsic[0, 0] / np.sqrt(2)
        self.scale = self.scale[:, :, None].repeat(3, -1)
        # gaussian property -- HW4 rotation
        self.rotation = np.zeros((self.H, self.W, 4))
        self.rotation[:, :, 0] = 1.0

    def _set_property_from_frame(self, frame: Frame):
        """frame here is a complete init/inpainted frame"""
        # basic frame-level property
        self.H: int = frame.H
        self.W: int = frame.W
        self.dpt: Float[np.ndarray, "H W"] = frame.dpt
        self.intrinsic: Float[np.ndarray, "3 3"] = frame.intrinsic
        self.world_T_cam: Float[np.ndarray, "4 4"] = frame.cam_T_world
        # gaussian property -- xyz with train-able pixel-aligned scale and shift
        self.xyz: Float[np.ndarray, "H W 3"] = self._to_3d()
        # gaussian property -- HW3 rgb
        self.rgb: Float[np.ndarray, "H W 3"] = frame.rgb
        # gaussian property -- HW4 rotation HW3 scale
        self._coarse_init_scale_rotations()
        # gaussian property -- HW opacity
        self.opacity: Float[np.ndarray, "H W 1"] = np.ones((self.H, self.W, 1)) * 0.8
        # to cuda
        self._paint_filter(frame.inpaint_wo_edge)
        self._to_cuda()
        # de-activate
        self.rgb = self.rgbs_deact(self.rgb)
        self.scale = self.scales_deact(self.scale)
        self.opacity = self.opacity_deact(self.opacity)
        # to torch parameters
        self.rgb = nn.Parameter(self.rgb, requires_grad=False)
        self.xyz = nn.Parameter(self.xyz, requires_grad=False)
        self.scale = nn.Parameter(self.scale, requires_grad=False)
        self.opacity = nn.Parameter(self.opacity, requires_grad=False)
        self.rotation = nn.Parameter(self.rotation, requires_grad=False)

    def _require_grad(self, sign: bool = True) -> None:
        self.rgb = self.rgb.requires_grad_(sign)
        self.xyz = self.xyz.requires_grad_(sign)
        self.scale = self.scale.requires_grad_(sign)
        self.opacity = self.opacity.requires_grad_(sign)
        self.rotation = self.rotation.requires_grad_(sign)


@dataclass
class RenderOutput:
    """
    Output from Gaussian splatting rendering containing RGB, depth, and alpha channels.
    Supports both PyTorch tensors and numpy arrays for flexible usage.
    """

    rgb: Float[Tensor, "H W 3"] | Float[ndarray, "H W 3"]
    depth: Float[Tensor, "H W"] | Float[ndarray, "H W"]
    alpha: Float[Tensor, "1 H W 1"] | Float[ndarray, "1 H W 1"]

    def to_numpy(self) -> "RenderOutput":
        """Convert all tensors to numpy arrays, returning new RenderOutput."""
        if isinstance(self.rgb, torch.Tensor):
            return RenderOutput(
                rgb=self.rgb.detach().cpu().numpy(),
                depth=self.depth.detach().cpu().numpy(),
                alpha=self.alpha.detach().cpu().numpy(),
            )
        return self  # Already numpy

    def to_tensor(self, device: str = "cuda") -> "RenderOutput":
        """Convert all numpy arrays to tensors, returning new RenderOutput."""
        if isinstance(self.rgb, np.ndarray):
            return RenderOutput(
                rgb=torch.from_numpy(self.rgb.astype(np.float32)).to(device),
                depth=torch.from_numpy(self.depth.astype(np.float32)).to(device),
                alpha=torch.from_numpy(self.alpha.astype(np.float32)).to(device),
            )
        return self  # Already tensor


class Gaussian_Scene:
    def __init__(self):
        # frames initialing the frame
        self.frames: list[Frame] = []
        self.gaussian_frames: list[Gaussian_Frame] = []  # gaussian frame require training at this optimization
        # activate fuctions
        self.rgbs_act = torch.sigmoid
        self.scales_act = torch.exp
        self.opacity_act = torch.sigmoid
        self.device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"
        # for traj generation
        self.traj_type = "spiral"
        self.traj_min_percentage = 5
        self.traj_max_percentage = 50
        self.traj_forward_ratio: float = 0.3
        self.traj_backward_ratio: float = 0.4

    # basic operations
    def _render_RGBD(
        self, frame: Frame, background_color: Literal["black", "white"] = "black"
    ) -> tuple[Float[Tensor, "H W 3"], Float[Tensor, "H W"], Float[Tensor, "1 H W 1"]]:
        background = None
        if background_color == "white":
            background = torch.ones(1, 4, device=self.device) * 0.1
            background[:, -1] = 0.0  # for depth
        # aligned untrainable xyz and unaligned trainable xyz
        # others
        xyz: Float[Tensor, "n_splats 3"] = torch.cat([gf.xyz.reshape(-1, 3) for gf in self.gaussian_frames], dim=0)
        rgb: Float[Tensor, "n_splats 3"] = torch.cat([gf.rgb.reshape(-1, 3) for gf in self.gaussian_frames], dim=0)
        scale: Float[Tensor, "n_splats 3"] = torch.cat([gf.scale.reshape(-1, 3) for gf in self.gaussian_frames], dim=0)
        opacity: Float[Tensor, "n_splats"] = torch.cat([gf.opacity.reshape(-1) for gf in self.gaussian_frames], dim=0)  # noqa: UP037
        rotation: Float[Tensor, "n_splats 4"] = torch.cat(
            [gf.rotation.reshape(-1, 4) for gf in self.gaussian_frames], dim=0
        )
        # activate
        rgb = self.rgbs_act(rgb)
        scale = self.scales_act(scale)
        rotation = F.normalize(rotation, dim=1)
        opacity = self.opacity_act(opacity)
        # property
        H = frame.H
        W = frame.W
        intrinsic: Float[Tensor, "3 3"] = torch.from_numpy(frame.intrinsic.astype(np.float32)).to(self.device)
        extrinsic: Float[Tensor, "4 4"] = torch.from_numpy(frame.cam_T_world.astype(np.float32)).to(self.device)
        # render
        raster_output: tuple[Float[Tensor, "1 H W 4"], Float[Tensor, "1 H W 1"], dict] = gs.rendering.rasterization(
            means=xyz,
            scales=scale,
            quats=rotation,
            opacities=opacity,
            colors=rgb,
            Ks=intrinsic[None],
            viewmats=extrinsic[None],
            width=W,
            height=H,
            packed=False,
            near_plane=0.01,
            render_mode="RGB+ED",
            backgrounds=background,
        )  # render: 1*H*W*(3+1)
        render_out: Float[Tensor, "1 H W 4"] = raster_output[0]
        render_alpha: Float[Tensor, "1 H W 1"] = raster_output[1]
        render_out: Float[Tensor, "H W 4"] = render_out.squeeze()
        # separate rgb, dpt, alpha
        render_rgb: Float[Tensor, "H W 3"] = render_out[:, :, 0:3]
        render_dpt: Float[Tensor, "H W"] = render_out[:, :, -1]
        return render_rgb, render_dpt, render_alpha

    @torch.no_grad()
    def _render_for_inpaint(self, frame: Frame) -> Frame:
        # first render
        render_rgb, render_dpt, render_alpha = self._render_RGBD(frame)
        render_msk = alpha_inpaint_mask(render_alpha)
        # to numpy
        render_rgb = render_rgb.detach().cpu().numpy()
        render_dpt = render_dpt.detach().cpu().numpy()
        render_alpha = render_alpha.detach().cpu().numpy()
        # assign back
        frame.rgb = render_rgb
        frame.dpt = render_dpt
        frame.inpaint = render_msk
        return frame

    def _add_trainable_frame(self, frame: Frame, require_grad: bool = True) -> None:
        # for the init frame, we keep all pixels for finetuning
        self.frames.append(frame)
        gf = Gaussian_Frame(frame, self.device)
        gf._require_grad(require_grad)
        self.gaussian_frames.append(gf)


def color2feat(color: Float[Tensor, "n_splats 3"]) -> Float[Tensor, "n_splats 3 16"]:
    """
    Converts input color values to a set of features for spherical harmonics (SH) representation.

    Args:
        color (torch.Tensor): Input color tensor of shape (N, 3), where N is the number of color samples.

    Returns:
        torch.Tensor: SH features tensor of shape (N, 3, 16).
    """
    max_sh_degree = 3
    # https://medium.com/data-science/a-comprehensive-overview-of-gaussian-splatting-e7d570081362#4cd8:~:text=While%20a%20bit,through%20SH.
    fused_color = (color - 0.5) / 0.28209479177387814
    features = np.zeros((fused_color.shape[0], 3, (max_sh_degree + 1) ** 2))
    features: Float[Tensor, "n_splats 3 16"] = torch.from_numpy(features.astype(np.float32))
    # Set the DC coefficients for RGB channels, everything else is zero.
    features[:, :3, 0] = fused_color
    features[:, 3:, 1:] = 0.0
    return features


def construct_list_of_attributes(features_dc, features_rest, scale, rotation) -> list[str]:
    attributes: list[str] = ["x", "y", "z", "nx", "ny", "nz"]
    # All channels except the 3 DC
    for i in range(features_dc.shape[1] * features_dc.shape[2]):
        attributes.append(f"f_dc_{i}")
    for i in range(features_rest.shape[1] * features_rest.shape[2]):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(scale.shape[1]):
        attributes.append(f"scale_{i}")
    for i in range(rotation.shape[1]):
        attributes.append(f"rot_{i}")
    return attributes


def save_ply(scene: Gaussian_Scene, path: Path) -> None:
    # extract data from scene.gaussian_frames
    xyz: Float[ndarray, "n_splats 3"] = (
        torch.cat([gf.xyz.reshape(-1, 3) for gf in scene.gaussian_frames], dim=0).detach().cpu().numpy()
    )
    scale: Float[ndarray, "n_splats 3"] = (
        torch.cat([gf.scale.reshape(-1, 3) for gf in scene.gaussian_frames], dim=0).detach().cpu().numpy()
    )
    opacities: Float[ndarray, "n_splats 1"] = (
        torch.cat([gf.opacity.reshape(-1) for gf in scene.gaussian_frames], dim=0)[:, None].detach().cpu().numpy()
    )
    rotation: Float[ndarray, "n_splats 4"] = (
        torch.cat([gf.rotation.reshape(-1, 4) for gf in scene.gaussian_frames], dim=0).detach().cpu().numpy()
    )
    rgb: Float[torch.Tensor, "n_splats 3"] = torch.sigmoid(
        torch.cat([gf.rgb.reshape(-1, 3) for gf in scene.gaussian_frames], dim=0)
    )
    # rgb
    features: Float[Tensor, "n_splats 3 16"] = color2feat(rgb)
    features_dc: Float[Tensor, "n_splats 3 1"] = features[:, :, 0:1]
    features_rest: Float[Tensor, "n_splats 3 15"] = features[:, :, 1:]

    f_dc: Float[ndarray, "n_splats 3"] = features_dc.flatten(start_dim=1).detach().cpu().numpy()
    f_rest: Float[ndarray, "n_splats 45"] = features_rest.flatten(start_dim=1).detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    # construct dtype for PLY saving
    dtype_full: list[tuple[str, str]] = [
        (attribute, "f4") for attribute in construct_list_of_attributes(features_dc, features_rest, scale, rotation)
    ]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)
    # compress using splat-transform
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    splat_transform_path = Path(conda_prefix) / "bin" / "splat-transform"

    # Convert to compressed PLY
    compressed_path: Path = path.parent / f"{path.stem}.compressed.ply"
    cmd = [str(splat_transform_path), "-w", str(path), str(compressed_path)]
    try:
        process = subprocess.run(cmd, capture_output=True, text=True)
        if process.returncode != 0:
            raise RuntimeError(f"Failed to compress PLY: {process.stderr}")
    except Exception as e:
        print(f"Error during PLY compression: {e}")
