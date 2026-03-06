from pathlib import Path
from typing import NotRequired, TypedDict

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from jaxtyping import Float, Int, Num, UInt8
from skimage.filters import gaussian
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results

from wilor_nano.models.wilor import WiLor
from wilor_nano.utils import utils


class WilorPreds(TypedDict):
    """
    TypedDict for predictions produced per detected hand.

    All arrays are numpy arrays. Batch dimension is 1 since predictions
    are extracted per-hand after batched inference.
    """

    pred_cam: Float[np.ndarray, "1 3"]
    global_orient: Float[np.ndarray, "1 1 3"]
    hand_pose: Float[np.ndarray, "1 15 3"]
    betas: Float[np.ndarray, "1 10"]
    pred_keypoints_3d: Float[np.ndarray, "1 n_joints=21 3"]
    pred_vertices: Float[np.ndarray, "1 n_verts=778 3"]
    pred_cam_t_full: Float[np.ndarray, "1 3"]
    scaled_focal_length: float
    pred_keypoints_2d: Float[np.ndarray, "1 n_joints=21 2"]


class Detection(TypedDict, total=False):
    """
    TypedDict for a single detection result.

    - hand_bbox: [x1, y1, x2, y2] in image pixel coordinates
    - is_right: 1 for right hand, 0 for left hand
    - wilor_preds: optional detailed model predictions
    """

    hand_bbox: list[float]
    is_right: int
    wilor_preds: NotRequired[WilorPreds]


class WiLorHandPose3dEstimationPipeline:
    def __init__(self, **kwargs):
        self.verbose: bool = False
        self.init_models(**kwargs)

    def init_models(self, hf_repo_id: str = "pablovela5620/wilor-nano", **kwargs):
        """
        focal_length: you will need to scale the actual focal length by 256/max_image_side_length for wilor to estimate
            camera translation properly.
        """
        # default to use CPU
        self.device = kwargs.get("device", torch.device("cpu"))
        self.dtype = kwargs.get("dtype", torch.float32)
        self.FOCAL_LENGTH = kwargs.get("focal_length", 5000)
        self.IMAGE_SIZE: int = 256
        self.WILOR_MINI_REPO_ID = hf_repo_id
        wilor_pretrained_dir: Path = Path(kwargs.get("wilor_pretrained_dir", Path(__file__).parent.parent)).resolve()
        wilor_pretrained_dir.mkdir(parents=True, exist_ok=True)
        mano_mean_path: Path = wilor_pretrained_dir / "pretrained_models" / "mano_mean_params.npz"
        if not mano_mean_path.exists():
            downloaded_path: str = hf_hub_download(
                repo_id=self.WILOR_MINI_REPO_ID,
                subfolder="pretrained_models",
                filename="mano_mean_params.npz",
                local_dir=wilor_pretrained_dir,
            )
            mano_mean_path = Path(downloaded_path)
        mano_model_path: Path = wilor_pretrained_dir / "pretrained_models" / "mano_clean" / "MANO_RIGHT.pkl"
        if not mano_model_path.exists():
            downloaded_path = hf_hub_download(
                repo_id=self.WILOR_MINI_REPO_ID,
                subfolder="pretrained_models/mano_clean",
                filename="MANO_RIGHT.pkl",
                local_dir=wilor_pretrained_dir,
            )
            mano_model_path = Path(downloaded_path)
        self.wilor_model = WiLor(
            mano_root_dir=mano_model_path.parent,
            mano_mean_path=mano_mean_path,
            focal_length=self.FOCAL_LENGTH,
            image_size=self.IMAGE_SIZE,
        )
        wilor_model_path: Path = wilor_pretrained_dir / "pretrained_models" / "wilor_final.ckpt"
        if not wilor_model_path.exists():
            downloaded_path = hf_hub_download(
                repo_id=self.WILOR_MINI_REPO_ID,
                subfolder="pretrained_models",
                filename="wilor_final.ckpt",
                local_dir=wilor_pretrained_dir,
            )
            wilor_model_path = Path(downloaded_path)
        self.wilor_model.load_state_dict(torch.load(wilor_model_path)["state_dict"], strict=False)
        self.wilor_model.eval()
        self.wilor_model.to(self.device, dtype=self.dtype)

        yolo_model_path: Path = wilor_pretrained_dir / "pretrained_models" / "detector.pt"
        if not yolo_model_path.exists():
            downloaded_path = hf_hub_download(
                repo_id=self.WILOR_MINI_REPO_ID,
                subfolder="pretrained_models",
                filename="detector.pt",
                local_dir=wilor_pretrained_dir,
            )
            yolo_model_path = Path(downloaded_path)
        self.hand_detector: YOLO = YOLO(yolo_model_path)
        self.hand_detector.to(self.device)

    @torch.no_grad()
    def predict(
        self,
        rgb_hw3: UInt8[np.ndarray, "h w 3"],
        hand_conf: float = 0.3,
        rescale_factor: float = 2.5,
    ) -> list[Detection]:
        detections: Results = self.hand_detector(rgb_hw3, conf=hand_conf, verbose=self.verbose)[0]

        detect_rets: list[Detection] = []
        bbox_list: list[list[float]] = []
        is_rights: list[int] = []
        for det in detections:
            hand_bbox: Float[np.ndarray, "6"] = det.boxes.data.cpu().detach().squeeze().numpy()
            is_rights.append(int(det.boxes.cls.cpu().detach().squeeze().item()))
            bbox_list.append(hand_bbox[:4].tolist())
            detect_rets.append({"hand_bbox": bbox_list[-1], "is_right": is_rights[-1]})

        if len(bbox_list) == 0:
            return detect_rets

        bboxes: Float[np.ndarray, "n 4"] = np.stack(bbox_list)

        center: Float[np.ndarray, "n 2"] = (bboxes[:, 2:4] + bboxes[:, 0:2]) / 2.0
        scale: Float[np.ndarray, "n 2"] = rescale_factor * (bboxes[:, 2:4] - bboxes[:, 0:2])
        img_patches_list: list[np.ndarray] = []
        img_size: Int[np.ndarray, "2"] = np.array([rgb_hw3.shape[1], rgb_hw3.shape[0]])
        for i in tqdm(range(bboxes.shape[0]), disable=not self.verbose):
            bbox_size = scale[i].max()
            patch_width = patch_height = self.IMAGE_SIZE
            right: int = is_rights[i]
            flip: bool = right == 0
            box_center = center[i]

            cvimg: UInt8[np.ndarray, "h w 3"] = rgb_hw3.copy()
            # Blur image to avoid aliasing artifacts
            downsampling_factor = (bbox_size * 1.0) / patch_width
            downsampling_factor = downsampling_factor / 2.0
            if downsampling_factor > 1.1:
                cvimg = gaussian(cvimg, sigma=(downsampling_factor - 1) / 2, channel_axis=2, preserve_range=True)

            img_patch_cv, trans = utils.generate_image_patch_cv2(
                cvimg,
                box_center[0],
                box_center[1],
                bbox_size,
                bbox_size,
                patch_width,
                patch_height,
                flip,
                1.0,
                0,
                border_mode=cv2.BORDER_CONSTANT,
            )
            img_patches_list.append(img_patch_cv)
        img_patches: Num[np.ndarray, "n 256 256 3"] = np.stack(img_patches_list)
        img_patches = torch.from_numpy(img_patches).to(device=self.device, dtype=self.dtype)
        wilor_output = self.wilor_model(img_patches)
        wilor_output = {k: v.cpu().float().numpy() for k, v in wilor_output.items()}

        for i in range(len(detect_rets)):
            wilor_output_i = {key: val[[i]] for key, val in wilor_output.items()}
            pred_cam = wilor_output_i["pred_cam"]
            bbox_size = scale[i].max()
            box_center = center[i]
            right = is_rights[i]
            multiplier = 2 * right - 1
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            if right == 0:
                wilor_output_i["pred_keypoints_3d"][:, :, 0] = -wilor_output_i["pred_keypoints_3d"][:, :, 0]
                wilor_output_i["pred_vertices"][:, :, 0] = -wilor_output_i["pred_vertices"][:, :, 0]
                wilor_output_i["global_orient"] = np.concatenate(
                    (wilor_output_i["global_orient"][:, :, 0:1], -wilor_output_i["global_orient"][:, :, 1:3]), axis=-1
                )
                wilor_output_i["hand_pose"] = np.concatenate(
                    (wilor_output_i["hand_pose"][:, :, 0:1], -wilor_output_i["hand_pose"][:, :, 1:3]), axis=-1
                )
            scaled_focal_length = self.FOCAL_LENGTH / self.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = utils.cam_crop_to_full(
                pred_cam, box_center[None], bbox_size, img_size[None], scaled_focal_length
            )
            wilor_output_i["pred_cam_t_full"] = pred_cam_t_full
            wilor_output_i["scaled_focal_length"] = scaled_focal_length
            pred_keypoints_2d: Float[np.ndarray, "1 21 2"] = utils.perspective_projection(
                wilor_output_i["pred_keypoints_3d"],
                translation=pred_cam_t_full,
                focal_length=np.array([scaled_focal_length] * 2)[None],
                camera_center=img_size[None] / 2,
            )
            wilor_output_i["pred_keypoints_2d"] = pred_keypoints_2d
            detect_rets[i]["wilor_preds"] = wilor_output_i  # type: ignore[typeddict-item]

        return detect_rets

    @torch.no_grad()
    def predict_with_bboxes(
        self,
        image: UInt8[np.ndarray, "h w 3"],
        bboxes: Float[np.ndarray, "n 4"],
        is_rights: Int[np.ndarray, "n"],
        **kwargs,
    ) -> list[Detection]:
        detect_rets: list[Detection] = []
        if len(bboxes) == 0:
            return detect_rets
        for i in range(bboxes.shape[0]):
            detect_rets.append({"hand_bbox": bboxes[i, :4].tolist(), "is_right": is_rights[i]})
        rescale_factor = kwargs.get("rescale_factor", 2.5)
        center: Float[np.ndarray, "n 2"] = (bboxes[:, 2:4] + bboxes[:, 0:2]) / 2.0
        scale: Float[np.ndarray, "n 2"] = rescale_factor * (bboxes[:, 2:4] - bboxes[:, 0:2])
        img_patches_list: list[np.ndarray] = []
        img_size: Int[np.ndarray, "2"] = np.array([image.shape[1], image.shape[0]])
        for i in tqdm(range(bboxes.shape[0]), disable=not self.verbose):
            bbox_size = scale[i].max()
            patch_width = patch_height = self.IMAGE_SIZE
            right = is_rights[i]
            flip = right == 0
            box_center = center[i]

            cvimg = image.copy()
            # Blur image to avoid aliasing artifacts
            downsampling_factor = (bbox_size * 1.0) / patch_width
            downsampling_factor = downsampling_factor / 2.0
            if downsampling_factor > 1.1:
                cvimg = gaussian(cvimg, sigma=(downsampling_factor - 1) / 2, channel_axis=2, preserve_range=True)
            img_size = np.array([cvimg.shape[1], cvimg.shape[0]])

            img_patch_cv, trans = utils.generate_image_patch_cv2(
                cvimg,
                box_center[0],
                box_center[1],
                bbox_size,
                bbox_size,
                patch_width,
                patch_height,
                flip,
                1.0,
                0,
                border_mode=cv2.BORDER_CONSTANT,
            )
            img_patches_list.append(img_patch_cv)

        img_patches: Num[np.ndarray, "n 256 256 3"] = np.stack(img_patches_list)
        img_patches = torch.from_numpy(img_patches).to(device=self.device, dtype=self.dtype)
        wilor_output = self.wilor_model(img_patches)
        wilor_output = {k: v.cpu().float().numpy() for k, v in wilor_output.items()}

        for i in range(len(detect_rets)):
            wilor_output_i = {key: val[[i]] for key, val in wilor_output.items()}
            pred_cam = wilor_output_i["pred_cam"]
            bbox_size = scale[i].max()
            box_center = center[i]
            right = is_rights[i]
            multiplier = 2 * right - 1
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            if right == 0:
                wilor_output_i["pred_keypoints_3d"][:, :, 0] = -wilor_output_i["pred_keypoints_3d"][:, :, 0]
                wilor_output_i["pred_vertices"][:, :, 0] = -wilor_output_i["pred_vertices"][:, :, 0]
                wilor_output_i["global_orient"] = np.concatenate(
                    (wilor_output_i["global_orient"][:, :, 0:1], -wilor_output_i["global_orient"][:, :, 1:3]), axis=-1
                )
                wilor_output_i["hand_pose"] = np.concatenate(
                    (wilor_output_i["hand_pose"][:, :, 0:1], -wilor_output_i["hand_pose"][:, :, 1:3]), axis=-1
                )
            scaled_focal_length = self.FOCAL_LENGTH / self.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = utils.cam_crop_to_full(
                pred_cam, box_center[None], bbox_size, img_size[None], scaled_focal_length
            )
            wilor_output_i["pred_cam_t_full"] = pred_cam_t_full
            wilor_output_i["scaled_focal_length"] = scaled_focal_length
            pred_keypoints_2d: Float[np.ndarray, "1 21 2"] = utils.perspective_projection(
                wilor_output_i["pred_keypoints_3d"],
                translation=pred_cam_t_full,
                focal_length=np.array([scaled_focal_length] * 2)[None],
                camera_center=img_size[None] / 2,
            )
            wilor_output_i["pred_keypoints_2d"] = pred_keypoints_2d
            detect_rets[i]["wilor_preds"] = wilor_output_i  # type: ignore[typeddict-item]

        return detect_rets
