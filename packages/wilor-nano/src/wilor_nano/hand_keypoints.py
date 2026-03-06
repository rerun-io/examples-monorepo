"""Single-person hand keypoint estimation.

This module assumes there is at most one person's hands per frame. All
predictions are computed per-hand region-of-interest (ROI): either a left
hand or a right hand at a time. As such, outcomes correspond to one hand at
a time and there is no multi-person disambiguation or tracking here.

Overview:
- WiLor model predicts 3D hand pose/shape (MANO) and weak-perspective camera
  parameters from a hand crop. Outputs are canonicalized to a right-hand
  frame by mirroring left-hand inputs when needed.
- RTMPose provides 2D keypoint locations and per-keypoint confidence scores,
  which WiLor does not output; we reuse these confidences alongside WiLor's
  geometry.
- Post-processing lifts weak-perspective camera to full-image translation,
  rescales focal length from crop to full resolution, and projects 3D joints
  back to 2D image coordinates.

Conventions:
- Arrays are annotated with jaxtyping to document dtype and shape.
- In dev environments, package-wide runtime type checking may be enabled via
  `beartype_this_package()` to validate annotated shapes/dtypes at runtime.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from jaxtyping import Float, Int, Num, UInt8
from numpy import ndarray
from rtmlib.tools.pose_estimation.rtmpose import RTMPose
from serde import from_dict, serde
from skimage.filters import gaussian

from wilor_nano.models.refinement_net import RefineNetOutput
from wilor_nano.models.wilor import WiLor
from wilor_nano.utils import utils


@dataclass
class HandKeypointDetectorConfig:
    verbose: bool
    hf_wilor_repo_id: str = "pablovela5620/wilor-nano"
    pretrained_dir: Path = Path.cwd() / "pretrained_models"
    focal_length: int = 5000
    image_size: int = 256
    use_keypoints_results: bool = True


class RefineNetNPOutput(TypedDict):
    """
    TypedDict for the output of RefineNet.

    This defines the structure and types of the dictionary returned by the RefineNet forward method.
    """

    global_orient: Float[torch.Tensor, "batch"]
    hand_pose: Float[torch.Tensor, "batch 15 3"]
    betas: Float[torch.Tensor, "batch 10"]
    pred_cam: Float[torch.Tensor, "batch 3"]


@serde
class RawWilorPred:
    """WiLor/RefineNet raw predictions for one hand (batch=1), model space."""

    global_orient: Float[ndarray, "batch=1 1 3"]
    """Root wrist global orientation (axis-angle, radians), shape (1, 1, 3)."""
    hand_pose: Float[ndarray, "batch=1 15 3"]
    """Local joint rotations for 15 MANO joints (axis-angle), shape (1, 15, 3)."""
    betas: Float[ndarray, "batch=1 10"]
    """MANO shape coefficients, shape (1, 10)."""
    pred_cam: Float[ndarray, "batch=1 3"]
    """Weak-perspective camera in crop coords [s, tx, ty], shape (1, 3)."""
    pred_keypoints_3d: Float[ndarray, "batch=1 n_joints=21 3"]
    """21 MANO joints in model space, shape (1, 21, 3)."""
    pred_vertices: Float[ndarray, "batch=1 n_verts=778 3"]
    """778 MANO mesh vertices in model space, shape (1, 778, 3)."""


@serde
class FinalWilorPred:
    """Per-hand predictions post-processed to a canonical right-hand frame."""

    global_orient: Float[ndarray, "batch=1 1 3"]
    """Canonicalized root orientation (axis-angle), shape (1, 1, 3)."""
    hand_pose: Float[ndarray, "batch=1 15 3"]
    """Canonicalized joint rotations (axis-angle), shape (1, 15, 3)."""
    betas: Float[ndarray, "batch=1 10"]
    """MANO shape coefficients, shape (1, 10)."""
    pred_cam: Float[ndarray, "batch=1 3"]
    """Weak-perspective cam [s, tx, ty] in crop coords (handedness-corrected), shape (1, 3)."""
    pred_cam_t_full: Float[ndarray, "batch=1 3"]
    """Full-image translation for perspective projection, shape (1, 3)."""
    scaled_focal_length: float
    """Focal length scaled from crop size to full image (scalar)."""
    pred_keypoints_3d: Float[ndarray, "batch=1 n_joints=21 3"]
    """Canonicalized 3D joints in model space, shape (1, 21, 3)."""
    pred_vertices: Float[ndarray, "batch=1 n_verts=778 3"]
    """Canonicalized MANO mesh vertices in model space, shape (1, 778, 3)."""
    pred_keypoints_2d: Float[ndarray, "batch=1 n_joints=21 2"]
    """2D pixel coordinates in the original image, shape (1, 21, 2)."""
    confidence_2d: Float[ndarray, "batch=1 n_joints=21"]
    """2D keypoint confidence scores, shape (1, 21)."""


@serde
class KeypointResults:
    """Keypoint detection results for a single hand (batch=1)."""

    keypoints_2d: Float[ndarray, "batch=1 n_joints=21 2"]
    """2D keypoints in pixel coordinates, shape (1, 21, 2)."""
    scores: Float[ndarray, "batch=1 n_joints=21"]
    """Confidence scores for each keypoint, shape (1, 21)."""
    global_orient: Float[ndarray, "batch=1 1 3"] | None
    """Canonicalized root orientation (axis-angle), shape (1, 1, 3)."""
    hand_pose: Float[ndarray, "batch=1 15 3"] | None
    """Canonicalized joint rotations (axis-angle), shape (1, 15, 3)."""
    betas: Float[ndarray, "batch=1 10"] | None
    """MANO shape coefficients, shape (1, 10)."""


class RTMPoseHandKeypointDetector:
    """RTMPose-based detector for 2D hand keypoints and confidences.

    This class uses RTMPose to estimate 2D keypoint locations and per-keypoint
    confidence scores from a hand region-of-interest (ROI). It assumes single-person
    detection and does not perform multi-person disambiguation or tracking.

    Overview:
    - RTMPose provides 2D keypoints and confidences, which are used alongside
      WiLor's 3D geometry in the parent detector.
    - Outputs are not canonicalized by handedness; mirroring is handled upstream.

    Conventions:
    - Arrays are annotated with jaxtyping to document dtype and shape.
    - In dev environments, package-wide runtime type checking may be enabled via
      `beartype_this_package()` to validate annotated shapes/dtypes at runtime.
    """

    def __init__(self, cfg: HandKeypointDetectorConfig) -> None:
        self.cfg: HandKeypointDetectorConfig = cfg
        self.init_model()

    def init_model(self):
        """Initialize the RTMPose model for hand keypoint detection."""
        url: str = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip"
        pose_input_size: tuple[int, int] = (256, 256)
        self.hand_model = RTMPose(onnx_model=url, model_input_size=pose_input_size, device="cuda")

    def __call__(
        self,
        rgb_hw3: UInt8[ndarray, "H W 3"],
        xyxy: Float[np.ndarray, "1 4"],
        handedness: Literal["left", "right"] | None = None,
    ) -> KeypointResults:
        """Estimate 2D keypoints and confidences for a single hand ROI.

        This method mirrors the signature of WilorHandKeypointDetector's __call__
        for consistency, though rescale_factor and handedness are not used here
        as RTMPose operates directly on the provided ROI without scaling or
        handedness adjustments.

        Steps:
        - Run RTMPose on the specified ROI to get 2D keypoints and scores.
        - Return results in a KeypointResults structure, with MANO fields set to None.

        Args:
            rgb_hw3: Original RGB image, uint8 shape (H, W, 3).
            xyxy: Bounding box for the hand (1, 4) [x1, y1, x2, y2].
            handedness: "left" or "right"; not used in this detector.

        Returns:
            KeypointResults with keypoints_2d (shape (1, 21, 2)), scores (shape (1, 21)),
            and MANO fields set to None.
        """
        xyxy_list: list[list[float]] = xyxy.tolist() if xyxy is not None else []
        rtmpose_output: tuple[Float[ndarray, "batch=1 n_kpts=21 2"], Float[ndarray, "batch=1 n_kpts=21"]] = (
            self.hand_model(rgb_hw3, bboxes=xyxy_list)
        )
        keypoints: Float[ndarray, "batch=1 n_kpts=21 2"] = rtmpose_output[0]
        scores: Float[ndarray, "batch=1 n_kpts=21"] = rtmpose_output[1]
        return KeypointResults(
            keypoints_2d=keypoints,
            scores=scores,
            global_orient=None,
            hand_pose=None,
            betas=None,
        )


class WilorHandKeypointDetector:
    def __init__(self, cfg: HandKeypointDetectorConfig) -> None:
        self.cfg: HandKeypointDetectorConfig = cfg
        self.init_model()

    def init_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16

        self.cfg.pretrained_dir.mkdir(parents=True, exist_ok=True)
        mano_mean_path: Path = self.cfg.pretrained_dir / "pretrained_models" / "mano_mean_params.npz"
        if not mano_mean_path.exists():
            downloaded_path: str = hf_hub_download(
                repo_id=self.cfg.hf_wilor_repo_id,
                subfolder="pretrained_models",
                filename="mano_mean_params.npz",
                local_dir=self.cfg.pretrained_dir,
            )
            mano_mean_path = Path(downloaded_path)

        mano_model_path: Path = self.cfg.pretrained_dir / "pretrained_models" / "mano_clean" / "MANO_RIGHT.pkl"
        if not mano_model_path.exists():
            downloaded_path = hf_hub_download(
                repo_id=self.cfg.hf_wilor_repo_id,
                subfolder="pretrained_models/mano_clean",
                filename="MANO_RIGHT.pkl",
                local_dir=self.cfg.pretrained_dir,
            )
            mano_model_path = Path(downloaded_path)
        self.wilor_model = WiLor(
            mano_root_dir=mano_model_path.parent,
            mano_mean_path=mano_mean_path,
            focal_length=self.cfg.focal_length,
            image_size=self.cfg.image_size,
        )
        wilor_model_path: Path = self.cfg.pretrained_dir / "pretrained_models" / "wilor_final.ckpt"
        if not wilor_model_path.exists():
            downloaded_path = hf_hub_download(
                repo_id=self.cfg.hf_wilor_repo_id,
                subfolder="pretrained_models",
                filename="wilor_final.ckpt",
                local_dir=self.cfg.pretrained_dir,
            )
            wilor_model_path = Path(downloaded_path)
        # wilor model for keypoint predictions, does not provide confidence values, so these are gotten from the rtmpose hand model
        self.wilor_model.load_state_dict(torch.load(wilor_model_path)["state_dict"], strict=False)
        self.wilor_model.eval()
        self.wilor_model.to(self.device, dtype=self.dtype)

        self.hand_confidence_model = RTMPoseHandKeypointDetector(self.cfg)

    @torch.no_grad()
    def __call__(
        self,
        rgb_hw3: UInt8[ndarray, "H W 3"],
        xyxy: Float[np.ndarray, "1 4"],
        handedness: Literal["left", "right"],
        rescale_factor: float = 2.5,
    ) -> FinalWilorPred | KeypointResults:
        """Run WiLor on a single-hand crop and return canonicalized outputs.

        Steps:
        - Create an anti-aliased crop around `xyxy`, optionally flipping if the
          hand is left to match the right-hand canonical frame.
        - Run WiLor to get MANO parameters, 3D joints/mesh, and weak-persp cam.
        - Use RTMPose to estimate 2D keypoint confidences (WiLor has none).
        - Post-process to adjust handedness, scale focal length, lift camera,
          and project 3D joints to 2D pixels in the full image.

        Args:
            rgb_hw3: Original RGB image, uint8 shape (H, W, 3).
            xyxy: Bounding box for the hand (1, 4) [x1, y1, x2, y2].
            handedness: "left" or "right"; controls mirroring and cam tx sign.
            rescale_factor: Crop scale relative to bbox size (default 2.5).

        Returns:
            FinalWilorPred with canonicalized right-hand outputs, including
            MANO parameters, 3D joints/vertices, camera, 2D projections, and
            2D confidences (from RTMPose).
        """
        # estimate the confidence of the keypoints using the rtmpose hand model
        rtmhand_results: KeypointResults = self.hand_confidence_model(rgb_hw3, xyxy)

        confidence_2d: Float[ndarray, "batch=1 n_kpts=21"] = rtmhand_results.scores
        # TODO implement a mechanism to ignore low-confidence keypoints and not continue with the higher cost wilor prediction
        # I can currently think of two:
        # 1. If more than X keypoints are below a certain threshold, skip wilor prediction and return empty results
        # 2. If the average confidence is below a certain threshold, skip wilor prediction
        # For now, we will just log a warning if the average confidence is below a certain threshold
        avg_confidence: float = float(confidence_2d.mean())
        if avg_confidence < 0.3 and self.cfg.verbose:
            print(
                f"Warning: Low average keypoint confidence ({avg_confidence:.2f}) for {handedness} hand at bbox {xyxy}. WiLor prediction may be unreliable."
            )

        center: Float[ndarray, "1 2"] = (xyxy[:, 2:4] + xyxy[:, 0:2]) / 2.0
        scale: Float[ndarray, "1 2"] = rescale_factor * (xyxy[:, 2:4] - xyxy[:, 0:2])
        img_patches_list: list[ndarray] = []
        img_size: Int[ndarray, "2"] = np.array([rgb_hw3.shape[1], rgb_hw3.shape[0]])

        bbox_size: float = float(scale[0].max())
        patch_width: int = self.cfg.image_size
        patch_height: int = self.cfg.image_size
        flip: bool = handedness == "left"
        box_center: Float[ndarray, "2"] = center[0]

        cvimg: Float[ndarray, "h w 3"] = rgb_hw3.copy().astype(np.float32)
        # Blur image to avoid aliasing artifacts
        downsampling_factor: float = (bbox_size * 1.0) / patch_width
        downsampling_factor: float = downsampling_factor / 2.0
        if downsampling_factor > 1.1:
            cvimg = gaussian(cvimg, sigma=(downsampling_factor - 1) / 2, channel_axis=2, preserve_range=True)

        patch_output: tuple[Float[ndarray, "h=256 w=256 3"], Float[ndarray, "2 3"]] = utils.generate_image_patch_cv2(
            cvimg,
            int(box_center[0]),
            int(box_center[1]),
            int(bbox_size),
            int(bbox_size),
            int(patch_width),
            int(patch_height),
            flip,
            1.0,
            0,
            border_mode=cv2.BORDER_CONSTANT,
        )
        img_patch_cv: Float[ndarray, "h=256 w=256 3"] = patch_output[0]
        img_patches_list.append(img_patch_cv)
        img_patches_np: Num[np.ndarray, "n_dets=1 h=256 w=256 3"] = np.stack(img_patches_list)
        img_patches: Float[torch.Tensor, "n_dets=1 h=256 w=256 3"] = torch.from_numpy(img_patches_np).to(
            device=self.device, dtype=self.dtype
        )
        # Run the WiLor model: outputs are PyTorch tensors keyed by component name
        wilor_output_raw: RefineNetOutput = self.wilor_model(img_patches)

        # Optional shape/type sanity check (e.g., with a serde-typed container).
        # Converts tensors to numpy for validation/logging, does not alter wilor_output_raw.
        raw_wilor_preds: RawWilorPred = from_dict(
            RawWilorPred, {k: v.cpu().float().numpy() for k, v in wilor_output_raw.items()}
        )

        final_preds: FinalWilorPred = self.post_process(
            raw_wilor_preds=raw_wilor_preds,
            handedness=handedness,
            box_center=box_center,
            bbox_size=bbox_size,
            img_size=img_size,
            confidence_2d=confidence_2d,
        )

        if self.cfg.use_keypoints_results:
            return KeypointResults(
                keypoints_2d=final_preds.pred_keypoints_2d,
                scores=final_preds.confidence_2d,
                global_orient=final_preds.global_orient,
                hand_pose=final_preds.hand_pose,
                betas=final_preds.betas,
            )

        return final_preds

    def post_process(
        self,
        *,
        raw_wilor_preds: RawWilorPred,
        handedness: Literal["left", "right"],
        box_center: Float[ndarray, "2"],
        bbox_size: float,
        img_size: Int[ndarray, "2"],
        confidence_2d: Float[ndarray, "batch=1 n_kpts=21"],
    ) -> FinalWilorPred:
        """Canonicalize handedness, lift camera to full image, and project 2D.

        - For left hands, mirror x-coordinates and flip axis-angle signs for
          y/z to match the right-hand canonical frame.
        - Adjust weak-perspective camera tx sign by handedness; lift to full
          image translation using scaled focal length and crop metadata.
        - Project 3D joints to 2D pixel coordinates in the original image.

        Returns a `FinalWilorPred` carrying both 3D and 2D quantities along
        with the per-keypoint 2D confidences.
        """
        # Unpack for clarity; do not mutate inputs
        pred_cam_in: Float[ndarray, "batch=1 3"] = raw_wilor_preds.pred_cam
        pred_kpts3d_in: Float[ndarray, "batch=1 n_joints=21 3"] = raw_wilor_preds.pred_keypoints_3d
        pred_verts_in: Float[ndarray, "batch=1 n_verts=778 3"] = raw_wilor_preds.pred_vertices
        global_orient_in: Float[ndarray, "batch=1 1 3"] = raw_wilor_preds.global_orient
        hand_pose_in: Float[ndarray, "batch=1 15 3"] = raw_wilor_preds.hand_pose

        # Adjust weak-perspective camera for handedness: cam = [s, tx, ty]
        sx: int = 1 if handedness == "right" else -1  # +1 for right hand, -1 for left hand
        pred_cam: Float[ndarray, "batch=1 3"] = pred_cam_in.copy()
        pred_cam[:, 1] = sx * pred_cam[:, 1]

        # Mirror to canonical frame for left hand
        if handedness == "left":
            pred_keypoints_3d: Float[ndarray, "batch=1 n_joints=21 3"] = pred_kpts3d_in.copy()
            pred_keypoints_3d[:, :, 0] = -pred_keypoints_3d[:, :, 0]
            pred_vertices: Float[ndarray, "batch=1 n_verts=778 3"] = pred_verts_in.copy()
            pred_vertices[:, :, 0] = -pred_vertices[:, :, 0]

            # Rotation vectors (axis-angle): [x, y, z] -> [x, -y, -z]
            global_orient: Float[ndarray, "batch=1 1 3"] = np.concatenate(
                (global_orient_in[:, :, 0:1], -global_orient_in[:, :, 1:3]), axis=-1
            )
            hand_pose: Float[ndarray, "batch=1 15 3"] = np.concatenate(
                (hand_pose_in[:, :, 0:1], -hand_pose_in[:, :, 1:3]), axis=-1
            )
        else:
            pred_keypoints_3d = pred_kpts3d_in.copy()
            pred_vertices = pred_verts_in.copy()
            global_orient = global_orient_in.copy()
            hand_pose = hand_pose_in.copy()

        # Scale focal length from model crop size (image_size) to full image resolution
        scaled_focal_length: float = self.cfg.focal_length / self.cfg.image_size * img_size.max()

        # Lift weak-perspective cam to full-image translation for projection to 2D
        pred_cam_t_full: Float[ndarray, "batch=1 3"] = utils.cam_crop_to_full(
            pred_cam, box_center[None], bbox_size, img_size[None], scaled_focal_length
        )

        # Project 3D joints to 2D pixel coordinates in the original image
        pred_keypoints_2d: Float[ndarray, "batch=1 n_joints=21 2"] = utils.perspective_projection(
            pred_keypoints_3d,
            translation=pred_cam_t_full,
            focal_length=np.array([scaled_focal_length] * 2)[None],
            camera_center=img_size[None] / 2,
        )

        return FinalWilorPred(
            global_orient=global_orient,
            hand_pose=hand_pose,
            betas=raw_wilor_preds.betas.copy(),
            pred_cam=pred_cam,
            pred_cam_t_full=pred_cam_t_full,
            scaled_focal_length=scaled_focal_length,
            pred_keypoints_3d=pred_keypoints_3d,
            pred_vertices=pred_vertices,
            pred_keypoints_2d=pred_keypoints_2d,
            confidence_2d=confidence_2d,
        )
