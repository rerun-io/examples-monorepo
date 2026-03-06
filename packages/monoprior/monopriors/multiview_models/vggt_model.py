from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from timeit import default_timer as timer
from typing import Any, Literal, TypedDict

import cv2
import numpy as np
import torch
from einops import rearrange
from jaxtyping import Bool, Float32, UInt8
from numpy import ndarray
from PIL import Image
from serde import field as serde_field
from serde import from_dict, serde
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters, rescale_intri
from torch import Tensor
from torchvision import transforms as TF
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from monopriors.depth_utils import multidepth_to_points


@contextmanager
def amp_autocast(device_type: Literal["cpu", "cuda"], dtype: torch.dtype) -> Iterator[None]:
    """Context manager that wraps torch.amp.autocast with explicit enter/exit."""

    if device_type == "cuda":
        autocast_cm: Any = torch.amp.autocast(device_type=device_type, dtype=dtype)
        autocast_cm.__enter__()
        try:
            yield None
        finally:
            autocast_cm.__exit__(None, None, None)
    else:
        yield None


class PreprocessingMetadata(TypedDict):
    original_size: tuple[int, int]  # (width, height)
    mode: Literal["crop", "pad"]  # Processing mode
    target_size: int  # Target width (usually 518px)
    padding: dict[Literal["top", "left", "right", "bottom"], int]  # Padding values
    new_size: tuple[int, int]  # (width, height) after resizing


@serde(deny_unknown_fields=True)
class VGGTPredictions:
    pose_enc: UInt8[ndarray, "*batch num_cams 9"]
    depth: Float32[ndarray, "*batch num_cams H W 1"]
    depth_conf: Float32[ndarray, "*batch num_cams H W"]
    world_points: Float32[ndarray, "*batch num_cams H W 3"]
    world_points_conf: Float32[ndarray, "*batch num_cams H W"]
    images: Float32[ndarray, "*batch num_cams 3 H W"]
    intrinsic: Float32[ndarray, "*batch num_cams 3 3"]
    cam_T_world_b34: Float32[ndarray, "*batch num_cams 3 4"] = serde_field(rename="extrinsic")

    def remove_batch_dim_if_one(self) -> "VGGTPredictions":
        """
        Removes the batch dimension from all arrays if batch size is 1.

        Returns:
            VGGTPredictions: A new instance with batch dimension removed if batch=1
        """
        if self.pose_enc.shape[0] != 1:
            return self

        result = VGGTPredictions(
            pose_enc=self.pose_enc.squeeze(0),
            depth=self.depth.squeeze(0),
            depth_conf=self.depth_conf.squeeze(0),
            world_points=self.world_points.squeeze(0),
            world_points_conf=self.world_points_conf.squeeze(0),
            images=self.images.squeeze(0),
            cam_T_world_b34=self.cam_T_world_b34.squeeze(0),
            intrinsic=self.intrinsic.squeeze(0),
        )
        return result


@dataclass
class PreprocessResults:
    images: Float32[torch.Tensor, "N 3 H W"]
    metadata: list[PreprocessingMetadata]


def preprocess_images(
    rgb_list: list[UInt8[ndarray, "H W 3"]],
    mode: Literal["crop", "pad"] = "crop",
) -> PreprocessResults:
    """
    A quick start function to preprocess images for model input.

    Args:
        rgb_list (list): List of RGB images as numpy arrays
        mode (str): Processing mode, either "crop" or "pad"

    Returns:
        tuple: (
            torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W),
            list: List of preprocessing metadata dictionaries for each image
        )

    Raises:
        ValueError: If the input list is empty

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - The function ensures width=518px while maintaining aspect ratio
        - Height is adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(rgb_list) == 0:
        raise ValueError("At least 1 image is required")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518
    metadata_list: list[PreprocessingMetadata] = []
    mode_literal: Literal["crop", "pad"] = mode

    # First process all images and collect their shapes
    for rgb in rgb_list:
        # Convert the numpy array to PIL Image to ensure identical processing
        pil_img = Image.fromarray(rgb)
        original_width, original_height = pil_img.size

        # Initialize metadata as TypedDict with explicit constructor
        metadata = PreprocessingMetadata(
            original_size=(original_width, original_height),
            mode=mode_literal,
            target_size=target_size,
            padding={"top": 0, "left": 0, "right": 0, "bottom": 0},
            new_size=(0, 0),  # Will be filled later
        )

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if original_width >= original_height:
                new_width = target_size
                new_height = round(original_height * (new_width / original_width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width: int = (
                    round(original_width * (new_height / original_height) / 14) * 14
                )  # Make divisible by 14

            metadata["new_size"] = (new_width, new_height)
            # Calculate padding
            pad_top: int = (target_size - new_height) // 2
            pad_bottom: int = target_size - new_height - pad_top
            pad_left: int = (target_size - new_width) // 2
            pad_right: int = target_size - new_width - pad_left

            metadata["padding"] = {"top": pad_top, "bottom": pad_bottom, "left": pad_left, "right": pad_right}

            # Resize with new dimensions using PIL's BICUBIC
            pil_img = pil_img.resize((new_width, new_height), Image.Resampling.BICUBIC)

            # Convert to tensor
            img = to_tensor(pil_img)

            # Apply padding
            img = torch.nn.functional.pad(
                img,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=1.0,
            )
        else:  # mode == "crop"
            # Original behavior: set width to target_size
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(original_height * (new_width / original_width) / 14) * 14
            metadata["new_size"] = (new_width, new_height)

            # Resize with new dimensions using PIL's BICUBIC for exact matching
            pil_img = pil_img.resize((new_width, new_height), Image.Resampling.BICUBIC)

            # Convert to tensor using the same to_tensor transform
            img = to_tensor(pil_img)  # Convert to tensor (0, 1)

            # Center crop height if it's larger than target_size
            if new_height > target_size:
                start_y = (new_height - target_size) // 2
                metadata["padding"]["top"] = -start_y  # Negative value indicates cropping
                img = img[:, start_y : start_y + target_size, :]
                metadata["new_size"] = (new_width, target_size)

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)
        metadata_list.append(metadata)

    # Check if we have different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for i, img in enumerate(images):
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Update metadata with additional padding
                metadata_list[i]["padding"]["top"] += pad_top
                metadata_list[i]["padding"]["bottom"] += pad_bottom
                metadata_list[i]["padding"]["left"] += pad_left
                metadata_list[i]["padding"]["right"] += pad_right

                img = torch.nn.functional.pad(
                    img,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image is (1, C, H, W)
    if len(rgb_list) == 1 and images.dim() == 3:
        images = images.unsqueeze(0)

    return PreprocessResults(images=images, metadata=metadata_list)


def remove_padding_from_prediction(
    pred: np.ndarray,
    metadata: PreprocessingMetadata,
) -> np.ndarray:
    """
    Remove padding from a prediction tensor based on preprocessing metadata.

    Args:
        pred: The prediction tensor/array with padding
        metadata: Dictionary containing padding information

    Returns:
        The unpadded array/tensor
    """
    # Get padding values
    pad_top: int = metadata["padding"]["top"]
    pad_left: int = metadata["padding"]["left"]
    new_width: int = metadata["new_size"][0]
    new_height: int = metadata["new_size"][1]

    if metadata["mode"] == "pad":
        # For pad mode, we need to crop out the padding
        if pred.ndim == 2:  # For 2D arrays like depth maps or masks
            return pred[pad_top : pad_top + new_height, pad_left : pad_left + new_width]
        elif pred.ndim == 3:  # For RGB images (H, W, C)
            return pred[pad_top : pad_top + new_height, pad_left : pad_left + new_width, :]
        else:
            raise ValueError(f"Unsupported tensor dimension: {pred.ndim}")
    else:  # For crop mode
        # In crop mode, padding values are used differently and might be negative
        # But we generally don't need to uncrop - we just need to resize later
        return pred


def filter_confidences(confidence: Float32[ndarray, "H W"], keep_top_percent: int | float) -> UInt8[ndarray, "H W"]:
    """
    Create a confidence mask by keeping the top percentage of pixels.

    Args:
        confidence: 2D confidence map where higher values indicate higher confidence
        keep_top_percent: Percentage of pixels to keep (0-100). E.g., 25.0 keeps top 25%

    Returns:
        Binary mask as UInt8 array with values 0 (filtered out) or 255 (kept)

    Notes:
        - Uses percentile-based thresholding: pixels >= (100 - keep_top_percent) percentile
        - Also filters out very low confidence values (< 1e-5) regardless of percentile
    """
    conf_threshold: float = float(np.percentile(confidence, 100.0 - keep_top_percent))
    mask: Bool[ndarray, "H W"] = (confidence >= conf_threshold) & (confidence > 1e-5)
    mask: UInt8[ndarray, "H W"] = (mask * 255).astype(np.uint8)
    return mask


def robust_filter_confidences(
    confidence: Float32[ndarray, "H W"], keep_top_percent: int | float
) -> UInt8[ndarray, "H W"]:
    """
    Robust confidence filtering that handles edge cases in percentile-based thresholding.

    This function addresses the issue where standard percentile thresholding can overshoot
    the target percentage when many pixels share the same confidence value as the percentile
    pivot (e.g., in nearly constant confidence maps).

    Args:
        confidence: 2D confidence map where higher values indicate higher confidence
        keep_top_percent: Target percentage of pixels to keep (0-100)

    Returns:
        Binary mask as UInt8 array with values 0 (filtered out) or 255 (kept)

    Notes:
        - Iteratively reduces the percentile threshold until the actual kept percentage
          is within tolerance (±10 percentage points) of the target
        - Uses filter_confidences() internally for the actual filtering
        - Handles degenerate cases like uniform confidence distributions
        - May keep slightly fewer pixels than requested to avoid significant overshoot
    """

    keep_top_percent: int | float = keep_top_percent
    mask: UInt8[ndarray, "H W"] = filter_confidences(confidence, keep_top_percent)
    # check percentage of values in masks that are 255
    percentage_255: float = float(100.0 * np.sum(mask == 255) / mask.size)
    tol: float = 10.0  # allowable deviation in percentage points
    target: float = float(keep_top_percent)
    while percentage_255 >= target + tol:
        keep_top_percent -= 1.0
        mask: UInt8[ndarray, "H W"] = filter_confidences(confidence, keep_top_percent)
        percentage_255: float = float(100.0 * np.sum(mask == 255) / mask.size)

    return mask


@dataclass
class MultiviewPred:
    """
    Multiview Consistent Depth Prediction.

    Attributes:
        cam_name (str): Name of the camera.
        rgb_image (UInt8[ndarray, "H W 3"]): RGB image.
        depth_map (UInt16[ndarray, "H W"]): Depth map computed from multi-view structure-from-motion.
            The depth values are scale-consistent across views, but only accurate up to an unknown global scale factor.
        pinhole_param (PinholeParameters): Pinhole camera parameters.
    """

    cam_name: str
    rgb_image: UInt8[ndarray, "H W 3"]
    depth_map: Float32[ndarray, "H W"]
    confidence_mask: Float32[ndarray, "H W"]
    pinhole_param: PinholeParameters


def generate_multiview_pred(
    pred_class: VGGTPredictions,
    img_tensors: Float32[Tensor, "num_img 3 resized_h resized_w"],
    rgb_list: list[UInt8[ndarray, "original_h original_w 3"]],
    metadata_list: list[PreprocessingMetadata] | None = None,
) -> list[MultiviewPred]:
    pred_class = pred_class.remove_batch_dim_if_one()
    assert len(pred_class.cam_T_world_b34.shape) == 3, "Currently batch size of 1 is only supported"

    # Generate world points from depth map, this is usually more accurate than the world points from pose encoding
    depth_maps: Float32[ndarray, "num_cams resized_h resized_w 1"] = pred_class.depth
    # Convert batch of (3x4) cam_T_world matrices to homogeneous (4x4)
    cam_T_world_b34: Float32[ndarray, "num_cams 3 4"] = pred_class.cam_T_world_b34.astype(np.float32)
    num_cams = cam_T_world_b34.shape[0]
    # Create bottom homogeneous row [0,0,0,1] for each camera
    bottom_row: Float32[ndarray, "num_cams 1 4"] = np.tile(np.array([[0, 0, 0, 1]], dtype=np.float32), (num_cams, 1, 1))
    # multidepth_to_points requires world_T_cam not cam_T_world
    cam_T_world_b44: Float32[ndarray, "num_cams 4 4"] = np.concatenate([cam_T_world_b34, bottom_row], axis=1)
    world_T_cam_b44: Float32[ndarray, "num_cams 4 4"] = np.linalg.inv(cam_T_world_b44)
    world_points: Float32[ndarray, "b h w 3"] = multidepth_to_points(
        depth_maps=depth_maps, world_T_cam_batch=world_T_cam_b44, K_b33=pred_class.intrinsic
    )

    # Get colors from original images and reshape them to match points
    processed_imgs: Float32[ndarray, "num_cams 3 resized_h resized_w"] = img_tensors.numpy(force=True)
    # Rearrange to match point shape expectation
    processed_imgs: Float32[ndarray, "num_cams resized_h resized_w 3"] = rearrange(
        processed_imgs,
        "num_cams C resized_h resized_w -> num_cams resized_h resized_w C",
    )

    # Process each image's data first - remove padding if metadata is available
    if metadata_list:
        unpadded_depth_maps = []
        unpadded_world_points = []
        unpadded_processed_imgs = []
        unpadded_depth_confs = []

        for i in range(len(processed_imgs)):
            # Remove padding from depths, world points, processed images, and confidence maps
            unpadded_depth_maps.append(remove_padding_from_prediction(depth_maps[i], metadata_list[i]))
            unpadded_world_points.append(remove_padding_from_prediction(world_points[i], metadata_list[i]))
            unpadded_processed_imgs.append(remove_padding_from_prediction(processed_imgs[i], metadata_list[i]))
            unpadded_depth_confs.append(remove_padding_from_prediction(pred_class.depth_conf[i], metadata_list[i]))

            # Also need to update camera intrinsics to account for removed padding
            if metadata_list[i]["mode"] == "pad":
                pad_left: int = metadata_list[i]["padding"]["left"]
                pad_top: int = metadata_list[i]["padding"]["top"]

                # Adjust principal point to account for removed padding
                pred_class.intrinsic[i, 0, 2] -= pad_left
                pred_class.intrinsic[i, 1, 2] -= pad_top

        # Replace the padded data with unpadded versions
        depth_maps = np.array(unpadded_depth_maps)
        world_points = np.array(unpadded_world_points)
        processed_imgs = np.array(unpadded_processed_imgs)
        depth_confs: Float32[ndarray, "num_cams _ _"] = np.array(unpadded_depth_confs)
    else:
        depth_confs: Float32[ndarray, "num_cams _ _"] = pred_class.depth_conf

    mv_pred_list: list[MultiviewPred] = []
    for idx, (intri, extri, processed_img, original_img, depth_map, depth_conf) in enumerate(
        zip(
            pred_class.intrinsic,
            pred_class.cam_T_world_b34,
            processed_imgs,
            rgb_list,
            depth_maps,
            depth_confs,
            strict=True,
        )
    ):
        cam_name: str = f"camera_{idx}"
        intri_param = Intrinsics(
            camera_conventions="RDF",
            fl_x=float(intri[0, 0]),
            fl_y=float(intri[1, 1]),
            cx=float(intri[0, 2]),
            cy=float(intri[1, 2]),
            width=processed_img.shape[1],
            height=processed_img.shape[0],
        )
        extri: Extrinsics = Extrinsics(cam_R_world=extri[:, :3], cam_t_world=extri[:, 3])

        pinhole_param = PinholeParameters(name=cam_name, intrinsics=intri_param, extrinsics=extri)

        depth_map = depth_map.squeeze()
        # Use INTER_LINEAR for the processed RGB image (standard for color images)
        processed_img: Float32[ndarray, "orig_h orig_w 3"] = cv2.resize(
            processed_img, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_LINEAR
        )
        # Use INTER_NEAREST for the confidence mask to preserve binary values
        conf_mask: Float32[ndarray, "orig_h orig_w"] = cv2.resize(
            depth_conf.astype(np.float32),
            (original_img.shape[1], original_img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        # Use INTER_NEAREST for depth map to preserve discontinuities and avoid floating artifacts
        depth_map: Float32[ndarray, "orig_h orig_w"] = cv2.resize(
            depth_map, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST
        )

        # rescale camera parameters to original image size
        pinhole_param.intrinsics = rescale_intri(
            pinhole_param.intrinsics,
            target_width=original_img.shape[1],
            target_height=original_img.shape[0],
        )

        # Normalize the processed image to [0, 1] range
        normalized: Float32[ndarray, "orig_h orig_w 3"] = (processed_img - processed_img.min()) / (
            processed_img.max() - processed_img.min()
        )
        rgb_image: UInt8[ndarray, "orig_h orig_w 3"] = (normalized * 255).clip(0, 255).astype(np.uint8)
        mv_pred_list.append(
            MultiviewPred(
                cam_name=cam_name,
                rgb_image=rgb_image,
                depth_map=depth_map,  # convert to uint16
                # confidence_mask=(conf_mask * 255).astype(np.uint8),
                confidence_mask=conf_mask,
                # pointcloud=pcd,
                # pointcloud_conf=pc_conf_mask,
                pinhole_param=pinhole_param,
            )
        )

    return mv_pred_list


class VGGTPredictor:
    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        preprocessing_mode: Literal["crop", "pad"] = "crop",
        local_files_only: bool = False,
    ) -> None:
        self.device = device
        self.preprocessing_mode: Literal["crop", "pad"] = preprocessing_mode
        load_start: float = timer()
        print("Loading model...")
        self.model: VGGT | None = VGGT.from_pretrained("facebook/VGGT-1B", local_files_only=local_files_only).to(
            self.device
        )
        print("Model loaded in", timer() - load_start, "seconds")
        self.dtype: torch.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    def __call__(self, rgb_list: list[UInt8[ndarray, "H W 3"]]) -> list[MultiviewPred]:
        preprocess_results: PreprocessResults = preprocess_images(rgb_list, mode=self.preprocessing_mode)
        img_tensors: Float32[torch.Tensor, "N 3 H W"] = preprocess_results.images.to(self.device)

        # Run inference
        print("Running inference...")
        with torch.no_grad(), amp_autocast(device_type=self.device, dtype=self.dtype):
            # run model and convert to dataclass for type validaton + easy access
            if self.model is not None:
                predictions: dict = self.model(img_tensors)
            else:
                raise RuntimeError("Model is not loaded. Please reload the model before inference.")

        # Convert pose encoding to extrinsic and intrinsic matrices
        print("Converting pose encoding to extrinsic and intrinsic matrices...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], img_tensors.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        # Tensor -> Numpy conversion
        for key in predictions:
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].numpy(force=True)

        # Convert from dict to dataclass and performs runtime type validation for easy access
        pred_class: VGGTPredictions = from_dict(VGGTPredictions, predictions)

        calibration_data: list[MultiviewPred] = generate_multiview_pred(
            pred_class,
            img_tensors=img_tensors,
            rgb_list=rgb_list,
            metadata_list=preprocess_results.metadata if self.preprocessing_mode == "pad" else None,
        )
        return calibration_data


def unload_vggt_model(vggt_predictor: VGGTPredictor) -> None:
    """
    Unload the VGGT model from memory by moving it to CPU and deleting references.
    This helps free GPU memory more effectively.

    Args:
        vggt_predictor: The VGGTPredictor instance containing the model to unload
    """
    # Move model to CPU first (helps with GPU memory cleanup)
    if hasattr(vggt_predictor, "model") and vggt_predictor.model is not None:
        vggt_predictor.model.cpu()
        # Remove reference to model
        vggt_predictor.model = None
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("VGGT model unloaded from memory")
