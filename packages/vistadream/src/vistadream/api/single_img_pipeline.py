from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from einops import rearrange
from icecream import ic
from jaxtyping import Bool, Float, UInt8
from monopriors.depth_utils import depth_edges_mask, depth_to_points
from monopriors.models.relative_depth import (
    RelativeDepthPrediction,
    get_relative_predictor,
)
from monopriors.models.relative_depth.base_relative_depth import BaseRelativePredictor
from PIL import Image
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole

from vistadream.ops.connect import Smooth_Connect_Tool
from vistadream.ops.flux import FluxInpainting, FluxInpaintingConfig
from vistadream.ops.gs.basic import Frame, Gaussian_Scene, save_ply
from vistadream.ops.gs.train import GS_Train_Tool
from vistadream.ops.trajs import _generate_trajectory
from vistadream.resize_utils import add_border_and_mask, process_image


def log_frame(parent_log_path: Path, frame: Frame, cam_params: PinholeParameters, log_pcd: bool = False) -> None:
    cam_log_path: Path = parent_log_path / cam_params.name
    pinhole_log_path: Path = cam_log_path / "pinhole"
    # extract values from frame
    rgb_hw3: UInt8[np.ndarray, "H W 3"] = (deepcopy(frame.rgb) * 255).astype(np.uint8)
    depth_hw: Float[np.ndarray, "H W"] = deepcopy(frame.dpt)
    edges_mask: Bool[np.ndarray, "h w"] = depth_edges_mask(depth_hw, threshold=0.01)
    masked_depth_hw: Float[np.ndarray, "h w"] = depth_hw * ~edges_mask
    inpaint_mask: Bool[np.ndarray, "H W"] = deepcopy(frame.inpaint)
    inpaint_wo_edge_mask: Bool[np.ndarray, "H W"] = deepcopy(frame.inpaint_wo_edge)
    depth_conf_mask: Bool[np.ndarray, "H W"] = deepcopy(frame.dpt_conf_mask)
    # convert masked_depth to uint16 for depth image
    masked_depth_hw = (masked_depth_hw * 1000).astype(np.uint16)  # Convert to uint16 for depth image

    rr.log(f"{pinhole_log_path}/rgb", rr.Image(rgb_hw3, color_model=rr.ColorModel.RGB).compress())
    rr.log(f"{pinhole_log_path}/depth", rr.DepthImage(masked_depth_hw, meter=1000.0))
    rr.log(
        f"{pinhole_log_path}/dpt_conf_mask",
        rr.Image(depth_conf_mask.astype(np.uint8) * 255, color_model=rr.ColorModel.L).compress(),
    )
    rr.log(
        f"{pinhole_log_path}/inpaint_mask",
        rr.Image(inpaint_mask.astype(np.uint8) * 255, color_model=rr.ColorModel.L),
    )
    rr.log(
        f"{pinhole_log_path}/inpaint_wo_edges_mask",
        rr.Image(inpaint_wo_edge_mask.astype(np.uint8) * 255, color_model=rr.ColorModel.L),
    )
    log_pinhole(camera=cam_params, cam_log_path=cam_log_path, image_plane_distance=0.05)

    if log_pcd:
        # convert back to f32 for point cloud logging
        depth_1hw: Float[np.ndarray, "1 h w"] = rearrange(masked_depth_hw, "h w -> 1 h w").astype(np.float32) / 1000
        pts_3d: Float[np.ndarray, "h w 3"] = depth_to_points(
            depth_1hw, cam_params.intrinsics.k_matrix.astype(np.float32)
        )
        rr.log(
            f"{parent_log_path}/{cam_params.name}_point_cloud",
            rr.Points3D(
                positions=pts_3d.reshape(-1, 3),
                colors=rgb_hw3.reshape(-1, 3),
            ),
        )


@dataclass
class SingleImageConfig:
    """
    Configuration for Single Image Processing.
    """

    rr_config: RerunTyroConfig
    image_path: Path
    offload: bool = True
    num_steps: int = 25
    guidance: float = 30.0
    expansion_percent: float = 0.3
    n_frames: int = 10
    max_resolution: Literal[512, 1024, 1920] = 512
    stage: Literal["no-outpaint", "outpaint", "coarse", "fine"] = "no-outpaint"


def pose_to_frame(scene: Gaussian_Scene, cam_T_world: Float[np.ndarray, "4 4"], margin: int = 32) -> Frame:
    """
    Convert camera pose to a Frame object with optional margin expansion.

    Args:
        scene: Gaussian scene containing reference frames
        cam_T_world: Camera-to-world transformation matrix
        margin: Additional pixels to add to frame dimensions

    Returns:
        Frame object with rendered content for inpainting
    """
    # Calculate expanded dimensions
    base_height: int = scene.frames[0].H
    base_width: int = scene.frames[0].W
    expanded_height: int = base_height + margin
    expanded_width: int = base_width + margin

    # Create adjusted intrinsics for expanded frame
    base_intrinsic: Float[np.ndarray, "3 3"] = deepcopy(scene.frames[0].intrinsic)
    adjusted_intrinsic: Float[np.ndarray, "3 3"] = base_intrinsic.copy()
    adjusted_intrinsic[0, 2] = expanded_width / 2.0  # cx - principal point x
    adjusted_intrinsic[1, 2] = expanded_height / 2.0  # cy - principal point y

    # Create frame with expanded dimensions and adjusted intrinsics
    frame: Frame = Frame(
        H=expanded_height,
        W=expanded_width,
        intrinsic=adjusted_intrinsic,
        cam_T_world=cam_T_world,
    )

    # Render the frame for inpainting using the scene
    rendered_frame: Frame = scene._render_for_inpaint(frame)

    return rendered_frame


class SingleImagePipeline:
    """
    Pipeline for Flux Outpainting using VistaDream.
    """

    def __init__(self, config: SingleImageConfig):
        self.config: SingleImageConfig = config
        self.scene: Gaussian_Scene = Gaussian_Scene()
        if self.config.stage in ["outpaint", "coarse", "fine"]:
            self.flux_inpainter: FluxInpainting = FluxInpainting(FluxInpaintingConfig())
        self.predictor: BaseRelativePredictor = get_relative_predictor("MogeV1Predictor")(device="cuda")
        self.smooth_connector: Smooth_Connect_Tool = Smooth_Connect_Tool()
        # Initialize rerun with the provided configuration
        self.shared_intrinsics: Intrinsics | None = None
        self.image_plane_distance: float = 0.01
        self.logged_cam_idx_list: list[int] = []

        # Detect image orientation early for blueprint configuration
        input_image: Image.Image = Image.open(self.config.image_path).convert("RGB")
        self.orientation: Literal["landscape", "portrait"] = (
            "landscape" if input_image.width >= input_image.height else "portrait"
        )
        print(f"[INFO] Detected {self.orientation} image ({input_image.width}x{input_image.height})")

        ic("Pipeline initialized with configuration:", self.config)

    def __call__(self):
        # Initialize rerun with the provided configuration
        self.setup_rerun()
        # outpaint -> depth prediction -> scene generation
        self._initialize()
        # generate the coarse scene
        if self.config.stage == "coarse":
            self._coarse()
        # save the scene
        save_dir: Path = Path("data/test_dir/")
        gf_path: Path = save_dir / "gf.ply"
        save_dir.mkdir(exist_ok=True, parents=True)
        self._render_splats()
        save_ply(self.scene, gf_path)

    def _coarse(self):
        """
        Generate coarse scene by iteratively adding frames with good inpainting areas.
        """
        # Generate dense trajectory for frame selection
        nframes: int = self.config.n_frames * 10  # Dense trajectory for selection
        dense_cam_T_world_traj: Float[np.ndarray, "n_frames 4 4"] = _generate_trajectory(self.scene, nframes=nframes)

        # Track selected frame indices to avoid adjacent selections
        select_frames: list[int] = []
        margin: int = 32

        print(f"[INFO] Generating coarse scene with up to {self.config.n_frames} frames...")

        # Iteratively add frames until we reach target count or no good frames found
        for frame_idx in range(self.config.n_frames - 2):  # -2 because we already have input+outpaint frames
            print(f"[INFO] Processing frame {frame_idx + 3}/{self.config.n_frames}...")

            # Find next best frame for inpainting
            next_frame: Frame | None = self._next_frame(dense_cam_T_world_traj, select_frames, margin)
            if next_frame is None:
                print("[INFO] No more suitable frames found for inpainting")
                break

            # Update logged camera list for blueprint
            cam_idx: int = len(self.scene.frames) + 2  # +2 for input and outpaint frames
            self.logged_cam_idx_list.append(cam_idx)

            # Inpaint the selected frame
            inpainted_frame: Frame = self._inpaint_next_frame(next_frame)
            cam_params: PinholeParameters = PinholeParameters(
                name=f"camera_{cam_idx}",
                intrinsics=self.shared_intrinsics,
                extrinsics=Extrinsics(
                    cam_R_world=inpainted_frame.cam_T_world[:3, :3],
                    cam_t_world=inpainted_frame.cam_T_world[:3, 3],
                ),
            )
            log_frame(self.parent_log_path, inpainted_frame, cam_params)

            # Add frame to scene and optimize
            self.scene._add_trainable_frame(inpainted_frame, require_grad=True)
            self.scene: Gaussian_Scene = GS_Train_Tool(self.scene, iters=500)(self.scene.frames, log=False)
            rr.send_blueprint(blueprint=self._create_blueprint(tab_idx=1))

    def _next_frame(
        self, dense_cam_T_world_traj: Float[np.ndarray, "n_frames 4 4"], select_frames: list[int], margin: int = 32
    ) -> Frame | None:
        """
        Select the frame with largest inpaint holes but less than 60% of the image.

        Args:
            dense_cam_T_world_traj: Dense trajectory of camera poses
            select_frames: List of already selected frame indices
            margin: Margin for frame expansion

        Returns:
            Frame object ready for inpainting, or None if no suitable frame found
        """
        # Calculate inpaint area ratio for each pose
        inpaint_area_ratios: list[float] = []

        for _pose_idx, cam_T_world in enumerate(dense_cam_T_world_traj):
            temp_frame: Frame = pose_to_frame(self.scene, cam_T_world, margin)
            inpaint_mask: Bool[np.ndarray, "H W"] = temp_frame.inpaint
            inpaint_ratio: float = float(np.mean(inpaint_mask.astype(np.float32)))
            inpaint_area_ratios.append(inpaint_ratio)

        inpaint_area_ratio_array: Float[np.ndarray, "n_frames"] = np.array(inpaint_area_ratios, dtype=np.float32)

        # Filter out frames with too much inpainting (> 25%)
        inpaint_area_ratio_array[inpaint_area_ratio_array > 0.25] = 0.0

        # Remove adjacent frames to already selected ones
        for selected_idx in select_frames:
            inpaint_area_ratio_array[selected_idx] = 0.0
            if selected_idx - 1 >= 0:
                inpaint_area_ratio_array[selected_idx - 1] = 0.0
            if selected_idx + 1 < len(dense_cam_T_world_traj):
                inpaint_area_ratio_array[selected_idx + 1] = 0.0

        # Select frame with largest inpaint area
        best_frame_idx: int = int(np.argmax(inpaint_area_ratio_array))
        best_inpaint_ratio: float = float(inpaint_area_ratio_array[best_frame_idx])

        # Minimum inpainting area ratio required to consider a frame suitable
        min_inpaint_ratio: float = 0.05
        # Check if we found a suitable frame (at least min_inpaint_ratio inpainting area)
        if best_inpaint_ratio < min_inpaint_ratio:
            return None

        # Add to selected frames and return the frame
        select_frames.append(best_frame_idx)
        selected_pose: Float[np.ndarray, "4 4"] = dense_cam_T_world_traj[best_frame_idx]
        selected_frame: Frame = pose_to_frame(self.scene, selected_pose, margin)

        print(f"[INFO] Selected frame {best_frame_idx} with inpaint ratio: {best_inpaint_ratio:.3f}")

        return selected_frame

    def _inpaint_next_frame(self, frame: Frame) -> Frame:
        """
        Inpaint the given frame using Flux and update depth information.

        Args:
            frame: Frame object containing RGB, depth, and inpaint mask

        Returns:
            Inpainted frame with updated RGB, depth, and masks
        """
        # Convert frame RGB and mask for Flux inpainting
        frame_rgb_hw3: UInt8[np.ndarray, "H W 3"] = (frame.rgb * 255).astype(np.uint8)
        frame_mask: UInt8[np.ndarray, "H W"] = frame.inpaint.astype(np.uint8) * 255

        # Inpaint RGB using Flux
        inpainted_image: Image.Image = self.flux_inpainter(rgb_hw3=frame_rgb_hw3, mask=frame_mask)
        inpainted_rgb_hw3: UInt8[np.ndarray, "H W 3"] = np.array(inpainted_image.convert("RGB"))

        # Update frame with inpainted RGB
        frame.rgb = inpainted_rgb_hw3.astype(np.float32) / 255.0  # Convert to [0,1] range

        # Predict depth for the inpainted frame
        depth_prediction: RelativeDepthPrediction = self.predictor.__call__(rgb=inpainted_rgb_hw3, K_33=frame.intrinsic)
        predicted_depth_hw: Float[np.ndarray, "H W"] = depth_prediction.depth

        # Remove any nans or infs and set them to 0
        predicted_depth_hw[np.isnan(predicted_depth_hw) | np.isinf(predicted_depth_hw)] = 0

        # Get the original rendered depth from the scene for alignment
        original_rendered_depth: Float[np.ndarray, "H W"] = frame.dpt

        # Align predicted depth with rendered depth using smooth connector
        aligned_depth_hw: Float[np.ndarray, "H W"] = self.smooth_connector._affine_dpt_to_GS(
            render_dpt=original_rendered_depth,
            inpaint_dpt=predicted_depth_hw,
            inpaint_msk=frame.inpaint,
        ).astype(np.float32)
        # aligned_depth_hw = predicted_depth_hw

        # Update frame depth with aligned depth
        frame.dpt = aligned_depth_hw

        # Calculate depth edges mask using aligned depth
        edges_mask: Bool[np.ndarray, "H W"] = depth_edges_mask(aligned_depth_hw, threshold=0.01)

        # Update inpaint mask without edges
        frame.inpaint_wo_edge = frame.inpaint & ~edges_mask

        # Get depth confidence mask from depth prediction confidence
        dpt_conf_mask: Float[np.ndarray, "H W"] = depth_prediction.confidence
        dpt_conf_mask_bool: Bool[np.ndarray, "H W"] = dpt_conf_mask > 0
        frame.dpt_conf_mask = dpt_conf_mask_bool

        # Further refine inpaint_wo_edge with depth confidence mask
        frame.inpaint_wo_edge = frame.inpaint_wo_edge & dpt_conf_mask_bool

        print(f"[INFO] Inpainted frame with {np.sum(frame.inpaint)}/{frame.inpaint.size} inpaint pixels")

        return frame

    def _render_splats(self):
        # render 5times frames
        nframes: int = min(len(self.scene.frames) * 25 if len(self.scene.frames) > 2 else 150, 150)
        cam_T_world_traj: Float[np.ndarray, "n_frames 4 4"] = _generate_trajectory(self.scene, nframes=nframes)
        # render
        print(f"[INFO] rendering final video with {nframes} frames...")
        rr.send_blueprint(blueprint=self._create_blueprint(tab_idx=2))
        # log the splats as point clouds
        g_xyz: Float[torch.Tensor, "n_splats 3"] = torch.cat(
            [gf.xyz.reshape(-1, 3) for gf in self.scene.gaussian_frames], dim=0
        )
        g_rgb: Float[torch.Tensor, "n_splats 3"] = torch.sigmoid(
            torch.cat([gf.rgb.reshape(-1, 3) for gf in self.scene.gaussian_frames], dim=0)
        )

        # log the gaussian scene
        rr.log(
            f"{self.final_log_path}/final_gaussian_scene",
            rr.Points3D(
                positions=g_xyz.clone().detach().float().cpu().numpy(),
                colors=g_rgb.clone().detach().float().cpu().numpy(),
            ),
            static=True,
        )

        for i, cam_T_world in enumerate(cam_T_world_traj, start=0):
            rr.set_time("time", sequence=i)
            frame = Frame(
                H=self.shared_intrinsics.height,
                W=self.shared_intrinsics.width,
                intrinsic=self.shared_intrinsics.k_matrix,
                cam_T_world=cam_T_world,
            )
            rgb, dpt, alpha = self.scene._render_RGBD(frame)
            rgb: Float[np.ndarray, "H W 3"] = rgb.detach().float().cpu().numpy()
            dpt: Float[np.ndarray, "H W"] = dpt.detach().float().cpu().numpy()

            rgb = (rgb * 255).astype(np.uint8)

            rr.set_time("time", sequence=i)
            cam_log_path: Path = self.final_log_path / "camera"
            pinhole_log_path: Path = cam_log_path / "pinhole"
            pinhole_param = PinholeParameters(
                name="camera",
                intrinsics=self.shared_intrinsics,
                extrinsics=Extrinsics(
                    cam_R_world=cam_T_world[:3, :3],
                    cam_t_world=cam_T_world[:3, 3],
                ),
            )
            rr.log(f"{pinhole_log_path}/rgb", rr.Image(rgb, color_model=rr.ColorModel.RGB).compress())
            rr.log(
                f"{pinhole_log_path}/depth",
                rr.DepthImage((deepcopy(dpt) * 1000).astype(np.uint16), meter=1000.0),
            )
            log_pinhole(
                camera=pinhole_param, cam_log_path=cam_log_path, image_plane_distance=self.image_plane_distance * 10
            )

        print(f"[INFO] DONE rendering {nframes} frames.")

    def _initialize(self) -> None:
        rr.set_time("time", sequence=0)

        input_image: Image.Image = Image.open(self.config.image_path).convert("RGB")
        # ensures image is correctly sized and processed
        input_image: Image.Image = process_image(input_image, max_dimension=self.config.max_resolution)

        if self.config.stage == "no-outpaint":
            # No outpainting, just use the input image directly
            outpaint_img: Image.Image = input_image
            outpaint_mask: Image.Image = Image.new("L", input_image.size, 0)
            # Just use the input image for depth prediction since there's no outpainting
            outpaint_rgb_hw3: UInt8[np.ndarray, "H W 3"] = np.array(outpaint_img.convert("RGB"))
            outpaint_rel_depth: RelativeDepthPrediction = self.predictor.__call__(rgb=outpaint_rgb_hw3, K_33=None)
            dpt_conf_mask: Float[np.ndarray, "h w"] = outpaint_rel_depth.confidence
            # convert to boolean mask, depth confidence mask is a binary mask where values > 0 are considered confident
            dpt_conf_mask: Bool[np.ndarray, "H W"] = dpt_conf_mask > 0

            outpaint_depth_hw: Float[np.ndarray, "H W"] = outpaint_rel_depth.depth

            # convert to numpy arrays
            outpaint_mask: Bool[np.ndarray, "H W"] = np.array(outpaint_mask).astype(np.bool_)
            outpaint_edges_mask: Bool[np.ndarray, "H W"] = depth_edges_mask(outpaint_depth_hw, threshold=0.01)
            # inpaint/outpaint mask without edges (True where inpainting is applied, False near edges and where no inpainting)
            outpaint_wo_edges: Bool[np.ndarray, "H W"] = outpaint_mask & ~outpaint_edges_mask
            # final mask
            outpaint_wo_edges = outpaint_wo_edges & dpt_conf_mask

        elif self.config.stage != "no-outpaint":
            # Auto-generate outpainting setup: user-controlled border expansion
            border_percent: float = (
                self.config.expansion_percent / 2.0
            )  # Convert to fraction per side (divide by 2 for each side)
            border_output: tuple[Image.Image, Image.Image] = add_border_and_mask(
                input_image,
                zoom_all=1.0,
                zoom_left=border_percent,
                zoom_right=border_percent,
                zoom_up=border_percent,
                zoom_down=border_percent,
                overlap=0,
            )
            outpaint_img: Image.Image = border_output[0]
            outpaint_mask: Image.Image = border_output[1]
            # Create outpainted image using Flux Inpainting
            outpaint_img: Image.Image = self.flux_inpainter(
                rgb_hw3=np.array(outpaint_img), mask=np.array(outpaint_mask)
            )

            outpaint_rgb_hw3: UInt8[np.ndarray, "H W 3"] = np.array(outpaint_img.convert("RGB"))
            outpaint_rel_depth: RelativeDepthPrediction = self.predictor.__call__(rgb=outpaint_rgb_hw3, K_33=None)
            dpt_conf_mask: Float[np.ndarray, "h w"] = outpaint_rel_depth.confidence
            # convert to boolean mask, depth confidence mask is a binary mask where values > 0 are considered confident
            dpt_conf_mask: Bool[np.ndarray, "H W"] = dpt_conf_mask > 0

            outpaint_depth_hw: Float[np.ndarray, "H W"] = outpaint_rel_depth.depth
            # remove any nans or infs and set them to 0
            outpaint_depth_hw[np.isnan(outpaint_depth_hw) | np.isinf(outpaint_depth_hw)] = 0
            # mask showing where outpainting (inpainting) is applied
            outpaint_mask: Bool[np.ndarray, "H W"] = np.array(outpaint_mask).astype(np.bool_)
            # depth edges, True near edges, False otherwise
            outpaint_edges_mask: Bool[np.ndarray, "H W"] = depth_edges_mask(outpaint_depth_hw, threshold=0.01)
            # inpaint/outpaint mask without edges (True where inpainting is applied, False near edges and where no inpainting)
            outpaint_wo_edges: Bool[np.ndarray, "H W"] = outpaint_mask & ~outpaint_edges_mask
            # final mask
            outpaint_wo_edges = outpaint_wo_edges & dpt_conf_mask

            outpaint_intri: Intrinsics = Intrinsics(
                camera_conventions="RDF",
                fl_x=outpaint_rel_depth.K_33[0, 0].item(),
                fl_y=outpaint_rel_depth.K_33[1, 1].item(),
                cx=outpaint_rel_depth.K_33[0, 2].item(),
                cy=outpaint_rel_depth.K_33[1, 2].item(),
                width=outpaint_rgb_hw3.shape[1],
                height=outpaint_rgb_hw3.shape[0],
            )
            outpaint_extri = Extrinsics(
                world_R_cam=np.eye(3, dtype=np.float32),
                world_t_cam=np.zeros(3, dtype=np.float32),
            )
            outpaint_pinhole: PinholeParameters = PinholeParameters(
                name="camera_outpaint",
                intrinsics=outpaint_intri,
                extrinsics=outpaint_extri,
            )

            outpaint_frame: Frame = Frame(
                H=outpaint_rgb_hw3.shape[0],
                W=outpaint_rgb_hw3.shape[1],
                rgb=outpaint_rgb_hw3.astype(np.float32) / 255.0,  # Convert to [0,1] range
                dpt=outpaint_depth_hw,
                intrinsic=outpaint_intri.k_matrix,
                cam_T_world=outpaint_extri.world_T_cam,  # Identity matrix for world coordinates
                inpaint=outpaint_mask,
                inpaint_wo_edge=outpaint_wo_edges,
                dpt_conf_mask=dpt_conf_mask,
            )

            log_frame(
                parent_log_path=self.parent_log_path, frame=outpaint_frame, cam_params=outpaint_pinhole, log_pcd=True
            )
            self.scene._add_trainable_frame(outpaint_frame, require_grad=True)

        input_rgb_hw3: UInt8[np.ndarray, "H W 3"] = np.array(input_image.convert("RGB"))
        # get input depth from outpaint depth, where outpaint mask is False.
        # This allows for getting the depth of the original image and only having to run the depth model once
        input_area = ~outpaint_mask
        input_depth_hw: Float[np.ndarray, "H W"] = outpaint_depth_hw[input_area].reshape(
            input_image.height, input_image.width
        )
        """
        ### Why Input Frame `inpaint` is Set to All `True`

        1.  **The Conceptual Issue**
            At first glance, this seems wrong because the input frame contains the original image data, and original pixels shouldn't need "inpainting". We'd expect `inpaint=False` for original content.

        2.  **The Training Logic Explanation**
            The `inpaint` mask in training determines which pixels should be supervised during optimization.
            - When `inpaint=True`: "Supervise this pixel - ensure the rendered result matches the target."
            - When `inpaint=False`: "Don't supervise this pixel - ignore it during training."

        3.  **Why All `True` Makes Sense for the Input Frame**
            For the input frame, setting `inpaint=True` everywhere means: "Train the Gaussians to perfectly reproduce the original image." All original pixels become supervision targets, forcing the 3D representation to learn to render the input view accurately.

        4.  **The Two-Frame Training Strategy**
            - **Input Frame**: Learns to reproduce original content. `input_frame.inpaint` is all `True`, so ALL pixels are supervised.
            - **Outpaint Frame**: Learns to reproduce only new content. `outpaint_frame.inpaint` is `True` only for outpainted areas.

        5.  **What This Achieves**
            - **Input Frame Training**: Every original pixel has `inpaint=True`, so the model learns to render them perfectly.
            - **Outpaint Frame Training**: Original pixels have `inpaint=False` (they are already learned, so we skip them), while new outpainted pixels have `inpaint=True` so the model learns to render the new content.

        6.  **The Alternative Would Be Problematic**
            If we set `input_frame.inpaint = False` everywhere, there would be no supervision on the original content. The Gaussians wouldn't learn to render the input view, leading to poor reconstruction quality for the reference image. Training would only learn the outpainted content.
        """
        input_mask: Bool[np.ndarray, "H W"] = np.full_like(input_depth_hw, True, dtype=np.bool_)
        input_edges_mask: Bool[np.ndarray, "H W"] = outpaint_edges_mask[input_area].reshape(
            input_image.height, input_image.width
        )
        input_mask_wo_edges: Bool[np.ndarray, "H W"] = ~input_edges_mask
        input_dpt_conf_mask: Bool[np.ndarray, "H W"] = dpt_conf_mask[input_area].reshape(
            input_image.height, input_image.width
        )
        input_k33: Float[np.ndarray, "3 3"] = outpaint_rel_depth.K_33.copy()
        # focal stays the same, but principal point is adjusted to center of input image
        input_k33[0, 2] = input_rgb_hw3.shape[1] / 2.0
        input_k33[1, 2] = input_rgb_hw3.shape[0] / 2.0

        input_intri: Intrinsics = Intrinsics(
            camera_conventions="RDF",
            fl_x=input_k33[0, 0].item(),
            fl_y=input_k33[1, 1].item(),
            cx=input_k33[0, 2].item(),
            cy=input_k33[1, 2].item(),
            width=input_rgb_hw3.shape[1],
            height=input_rgb_hw3.shape[0],
        )
        input_extri = Extrinsics(
            world_R_cam=np.eye(3, dtype=np.float32),
            world_t_cam=np.zeros(3, dtype=np.float32),
        )
        input_pinhole: PinholeParameters = PinholeParameters(
            name="camera_input",
            intrinsics=input_intri,
            extrinsics=input_extri,
        )
        self.shared_intrinsics = input_intri

        input_frame: Frame = Frame(
            H=input_rgb_hw3.shape[0],
            W=input_rgb_hw3.shape[1],
            rgb=input_rgb_hw3.astype(np.float32) / 255.0,  # Convert to [0,1] range
            dpt=input_depth_hw,
            intrinsic=input_k33,
            cam_T_world=input_extri.world_T_cam,  # Identity matrix for world coordinates
            inpaint=input_mask,
            inpaint_wo_edge=input_mask_wo_edges,
            dpt_conf_mask=input_dpt_conf_mask,
        )

        log_frame(parent_log_path=self.parent_log_path, frame=input_frame, cam_params=input_pinhole)
        self.scene._add_trainable_frame(input_frame, require_grad=True)

        self.scene: Gaussian_Scene = GS_Train_Tool(self.scene, iters=100)(self.scene.frames, log=False)

    def setup_rerun(self):
        self.parent_log_path: Path = Path("/world")
        self.final_log_path: Path = Path("/final")

        rr.send_blueprint(blueprint=self._create_blueprint())
        rr.log("/", rr.ViewCoordinates.RDF, static=True)

    def _create_blueprint(self, tab_idx: int = 0, debug: bool = False) -> rrb.Blueprint:
        """
        Create a rerun blueprint for the pipeline.
        """

        def create_camera_views_panel(camera_ids: list[int | str]) -> rrb.Horizontal:
            """
            Creates a horizontal panel of vertical 2D views for a list of camera identifiers.
            """
            contents = []
            for i in camera_ids:
                name_suffix = i.title() if isinstance(i, str) else str(i)
                view = rrb.Vertical(
                    rrb.Vertical(
                        rrb.Spatial2DView(
                            origin=f"{self.parent_log_path}/camera_{i}/pinhole/rgb",
                            contents=["+ $origin/**"],
                            name=f"Camera {name_suffix} RGB",
                        ),
                        rrb.Spatial2DView(
                            origin=f"{self.parent_log_path}/camera_{i}/pinhole/depth",
                            contents=["+ $origin/**"],
                            name=f"Camera {name_suffix} Depth",
                        ),
                    ),
                    rrb.Horizontal(
                        rrb.Spatial2DView(
                            origin=f"{self.parent_log_path}/camera_{i}/pinhole/dpt_conf_mask",
                            contents=["+ $origin/**"],
                            name=f"Camera {name_suffix} Depth Conf Mask",
                        ),
                        rrb.Spatial2DView(
                            origin=f"{self.parent_log_path}/camera_{i}/pinhole/inpaint_mask",
                            contents=["+ $origin/**"],
                            name=f"Camera {name_suffix} Inpaint Mask",
                        ),
                        rrb.Spatial2DView(
                            origin=f"{self.parent_log_path}/camera_{i}/pinhole/inpaint_wo_edges_mask",
                            contents=["+ $origin/**"],
                            name=f"Camera {name_suffix} Inpaint w/o Edges",
                        ),
                    ),
                    row_shares=[9, 1],
                )
                contents.append(view)
            return rrb.Horizontal(*contents)

        # create tabs
        initial_cameras = ["input", "outpaint"]
        initialization_2d_views = create_camera_views_panel(initial_cameras)

        # Adjust column shares based on image orientation
        init_column_shares: list[int] = [1, 1] if self.orientation == "landscape" else [5, 2]

        # Create content list for initialization view
        initial_cameras = ["input", "outpaint"]
        init_content_3d = [
            "+ $origin/**",
            f"- {self.final_log_path}/camera/pinhole/depth",
            *[
                f"- {self.parent_log_path}/camera_{cam}/pinhole/depth"
                for cam in initial_cameras + self.logged_cam_idx_list
            ],
            *[
                f"- {self.parent_log_path}/camera_{cam}/pinhole/dpt_conf_mask"
                for cam in initial_cameras + self.logged_cam_idx_list
            ],
            *[
                f"- {self.parent_log_path}/camera_{cam}/pinhole/inpaint_mask"
                for cam in initial_cameras + self.logged_cam_idx_list
            ],
            *[
                f"- {self.parent_log_path}/camera_{cam}/pinhole/inpaint_wo_edges_mask"
                for cam in initial_cameras + self.logged_cam_idx_list
            ],
        ]

        initialization_view = rrb.Horizontal(
            rrb.Spatial3DView(
                origin=self.parent_log_path,
                contents=init_content_3d,
            ),
            initialization_2d_views,
            column_shares=init_column_shares,
            name="Initialization",
        )

        # only show at most 5 cameras, pick them distributed
        if len(self.logged_cam_idx_list) > 5:
            # Sample 5 indices evenly from the list (including endpoints)
            idxs = np.linspace(0, len(self.logged_cam_idx_list) - 1, 5, dtype=int)
            view_cam_list: list[int] = [self.logged_cam_idx_list[i] for i in idxs]
        else:
            view_cam_list: list[int] = self.logged_cam_idx_list

        grid_view = create_camera_views_panel(view_cam_list)

        # fmt: off
        content_3d = [
            "+ $origin/**",
            *[f"- {self.parent_log_path}/camera_{i}/pinhole/depth" for i in self.logged_cam_idx_list + initial_cameras],
            *[f"- {self.parent_log_path}/camera_{i}/pinhole/dpt_conf_mask" for i in self.logged_cam_idx_list + initial_cameras],
            *[f"- {self.parent_log_path}/camera_{i}/pinhole/inpaint_mask" for i in self.logged_cam_idx_list + initial_cameras],
            *[f"- {self.parent_log_path}/camera_{i}/pinhole/inpaint_wo_edges_mask" for i in self.logged_cam_idx_list + initial_cameras],
        ]
        # fmt: on

        # coarse scene view
        coarse_scene_view = rrb.Horizontal(
            rrb.Spatial3DView(origin=self.parent_log_path, contents=content_3d),
            grid_view,
            column_shares=[5, 3],
            name="Coarse Scene",
        )

        final_scene_view = rrb.Horizontal(
            rrb.Spatial3DView(
                origin=self.final_log_path,
                contents=[
                    "+ $origin/**",
                ],
            ),
            rrb.Vertical(
                rrb.Spatial2DView(
                    origin=f"{self.final_log_path}/camera/pinhole",
                    contents=[
                        "+ $origin/**",
                    ],
                    name="Final Rendering",
                ),
            ),
            column_shares=[9, 4],
            name="Final Scene",
        )

        blueprint = rrb.Blueprint(
            rrb.Tabs(
                initialization_view,
                coarse_scene_view,
                final_scene_view,
                active_tab=tab_idx,
            ),
            collapse_panels=True,
        )
        return blueprint


def main(config: SingleImageConfig) -> None:
    """
    Main function to run the Single Image Processing Outpainting/Depth/Splat.
    """
    start_time = timer()
    vd_pipeline = SingleImagePipeline(config)
    vd_pipeline()
    end_time: float = timer()
    # Format elapsed time
    elapsed: float = end_time - start_time
    minutes: int = int(elapsed // 60)
    seconds: float = elapsed % 60
    print(f"Processing time: {minutes}m {seconds:.2f}s")
