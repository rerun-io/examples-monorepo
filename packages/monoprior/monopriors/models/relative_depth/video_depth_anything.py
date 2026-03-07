import gc
from collections.abc import Iterator
from dataclasses import dataclass
from timeit import default_timer as timer
from typing import Literal

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub import hf_hub_download
from jaxtyping import Float, Float32, UInt8
from torchvision.transforms import Compose
from tqdm import tqdm

from monopriors.depth_utils import disparity_to_depth, estimate_intrinsics
from monopriors.scale_utils import compute_scale_and_shift, get_interpolate_frames
from monopriors.third_party.depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize
from monopriors.third_party.video_depth_anything.video_depth import (
    INFER_LEN,
    INTERP_LEN,
    KEYFRAMES,
    OVERLAP,
    VideoDepthAnything,
)

from .base_relative_depth import BaseVideoRelativePredictor, RelativeDepthPrediction

model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}
encoder2name: dict[str, str] = {
    "vits": "Small",
    "vitb": "Base",
    "vitl": "Large",
    "vitg": "Giant",
}


# Define data classes for yields
@dataclass
class DepthChunkIntermediate:
    chunk_index: int
    raw_depth_list: list[np.ndarray]  # List of raw depth maps for the chunk.
    progress: float


@dataclass
class FinalDepthResult:
    aligned_depth: list[RelativeDepthPrediction]  # Final aligned depth predictions.


# Create a union type for the yields
DepthInferenceYield = DepthChunkIntermediate | FinalDepthResult


def pad_frames(
    frame_list: list[UInt8[np.ndarray, "H W 3"]],
    infer_len: int,
    overlap: int,
) -> tuple[list[UInt8[np.ndarray, "H W 3"]], int, int]:
    """
    Pads the frame list so that the video length fits an integer number of chunks.

    Each chunk has infer_len frames with an overlapping region of 'overlap' frames.
    This function computes how many extra frames to add (by copying the last frame) so that
    chunking can be done without index issues.

    Args:
        frame_list: List of original video frames.
        infer_len: Number of frames per chunk (e.g., 32).
        overlap: Number of overlapping frames between consecutive chunks (e.g., 10).

    Returns:
        padded_frame_list: The padded list of frames.
        org_video_len: The original number of frames.
        frame_step: The step size between chunk starts (infer_len - overlap).
    """
    frame_step: int = infer_len - overlap
    org_video_len: int = len(frame_list)
    append_frame_len: int = (frame_step - (org_video_len % frame_step)) % frame_step + (infer_len - frame_step)
    padded_frame_list = frame_list + [frame_list[-1].copy()] * append_frame_len
    return padded_frame_list, org_video_len, frame_step


def align_depths(
    depth_list: list[Float32[np.ndarray, "H W"]],
    keyframes: list[int],
    infer_len: int,
    overlap: int,
    interp_len: int,
) -> list[Float32[np.ndarray, "H W"]]:
    """
    Aligns depth maps across video chunks to ensure consistent depth scales.

    The alignment is performed by computing scale and shift parameters between keyframes of consecutive chunks,
    applying those parameters to the new chunk, and then interpolating the overlapping regions for a smooth transition.

    Process Details:
      1. Process video in chunks of infer_len frames with an overlapping region of 'overlap' frames.
      2. For each new chunk, the first 'overlap' frames are meant to replace the last overlapping frames from the previous chunk.
      3. After depth prediction, compute scale and shift alignment between keyframes from consecutive chunks.
      4. Apply the computed alignment parameters to the new chunk.
      5. In the overlapping region, create smooth transitions by interpolating between the aligned depths.
      6. This ensures consistent depth scales across the entire video.

    Args:
        depth_list: List of inferred depth maps for all frames.
        keyframes: List of keyframe indices used for alignment.
        infer_len: Number of frames per chunk.
        overlap: Number of overlapping frames between chunks.
        interp_len: Number of frames used for interpolation in the overlapping region.

    Returns:
        A list of aligned depth maps.
    """
    depth_list_aligned = []
    ref_align = []
    align_len = overlap - interp_len
    kf_align_list: list[int] = keyframes[:align_len]

    for frame_id in range(0, len(depth_list), infer_len):
        if not depth_list_aligned:
            # First chunk: no previous alignment, so use depths directly.
            depth_list_aligned += depth_list[frame_id : frame_id + infer_len]
            for kf_id in kf_align_list:
                ref_align.append(depth_list[frame_id + kf_id])
        else:
            # Gather keyframes from current chunk.
            curr_align = [depth_list[frame_id + i] for i in range(len(kf_align_list))]
            # Compute scale and shift parameters between current keyframes and reference keyframes.
            scale, shift = compute_scale_and_shift(
                np.concatenate(curr_align), np.concatenate(ref_align), np.concatenate(np.ones_like(ref_align) == 1)
            )
            # Interpolate the overlapping region.
            pre_depth_list = depth_list_aligned[-interp_len:]
            post_depth_list = depth_list[frame_id + align_len : frame_id + overlap]
            for i in range(len(post_depth_list)):
                post_depth_list[i] = post_depth_list[i] * scale + shift
                post_depth_list[i][post_depth_list[i] < 0] = 0
            depth_list_aligned[-interp_len:] = get_interpolate_frames(pre_depth_list, post_depth_list)
            # Align the remaining frames of the current chunk.
            for i in range(overlap, infer_len):
                new_depth = depth_list[frame_id + i] * scale + shift
                new_depth[new_depth < 0] = 0
                depth_list_aligned.append(new_depth)
            # Update reference keyframes for the next chunk.
            ref_align = ref_align[:1]
            for kf_id in kf_align_list[1:]:
                new_depth = depth_list[frame_id + kf_id] * scale + shift
                new_depth[new_depth < 0] = 0
                ref_align.append(new_depth)

    return depth_list_aligned


class VideoDepthAnythingPredictor(BaseVideoRelativePredictor):
    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        encoder: Literal["vits", "vitl"] = "vits",
        input_size: int = 518,
    ) -> None:
        super().__init__()
        print("Loading DepthAnythingV2 model...")
        start = timer()
        model_name: str = encoder2name[encoder]

        self.device = device

        self.model = VideoDepthAnything(**model_configs[encoder])
        filepath: str = hf_hub_download(
            repo_id=f"depth-anything/Video-Depth-Anything-{model_name}",
            filename=f"video_depth_anything_{encoder}.pth",
            repo_type="model",
        )
        self.model.load_state_dict(
            torch.load(filepath, map_location="cpu"),
            strict=True,
        )
        self.model = self.model.to(device).eval()
        print(f"VideoDepthAnything model loaded. Time: {timer() - start:.2f}s")
        self.input_size = input_size

    def __call__(
        self, rgb_frames: UInt8[np.ndarray, "T H W 3"], K_33: Float[np.ndarray, "3 3"] | None
    ) -> list[RelativeDepthPrediction]:
        frame_height: int = rgb_frames.shape[1]
        frame_width: int = rgb_frames.shape[2]

        if K_33 is None:
            K_33: Float32[np.ndarray, "3 3"] = estimate_intrinsics(H=frame_height, W=frame_width)

        transform = Compose(
            [
                Resize(
                    width=self.input_size,
                    height=self.input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

        # Adjust input size based on aspect ratio.
        ratio: float = max(frame_height, frame_width) / min(frame_height, frame_width)
        if ratio > 1.78:
            self.input_size = int(self.input_size * 1.777 / ratio)
            self.input_size = round(self.input_size / 14) * 14

        transform = Compose(
            [
                Resize(
                    width=self.input_size,
                    height=self.input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

        # Build frame list from the input array.
        frame_list: list[UInt8[np.ndarray, "H W 3"]] = [rgb_frames[i] for i in range(rgb_frames.shape[0])]

        # Pad frames so that the video length is compatible with chunking.
        padded_frame_list, org_video_len, frame_step = pad_frames(frame_list, INFER_LEN, OVERLAP)

        depth_list = []
        pre_input = None
        for frame_id in tqdm(
            range(0, org_video_len, frame_step),
            desc="Processing video chunks",
            unit="chunk",
        ):
            chunk_depth, pre_input = self.process_video_chunk(
                padded_frame_list,
                frame_id,
                transform,
                self.device,
                False,
                pre_input,
                frame_height,
                frame_width,
            )
            depth_list += chunk_depth

        del frame_list
        gc.collect()

        # Align the depths across chunks to ensure consistency.
        aligned_depth_list = align_depths(depth_list, KEYFRAMES, INFER_LEN, OVERLAP, INTERP_LEN)
        # remove the padded frames
        aligned_depth_list = aligned_depth_list[:org_video_len]

        relative_depth_list = [
            RelativeDepthPrediction(
                disparity=disparity,
                depth=disparity_to_depth(disparity, focal_length=int(K_33[0, 0])),
                confidence=np.ones_like(disparity),
                K_33=K_33,
            )
            for disparity in aligned_depth_list
        ]

        return relative_depth_list

    def infer_video_depth_iter(
        self, rgb_frames: UInt8[np.ndarray, "T H W 3"], K_33: Float32[np.ndarray, "3 3"] | None = None
    ) -> Iterator[DepthInferenceYield]:
        """
        An iterator-based inference function that yields intermediate raw (unaligned) depth chunks
        for visualization and then yields the final aligned result.

        Yields:
            DepthChunkIntermediate: For each processed chunk, yields an object containing:
                - chunk_index: Starting frame index of the chunk.
                - raw_depth: List of raw depth maps (numpy arrays) for that chunk.
                - progress: Fraction of original frames processed so far.
            FinalDepthResult: After processing all chunks, yields the final aligned result.
        """
        frame_height: int = rgb_frames.shape[1]
        frame_width: int = rgb_frames.shape[2]

        if K_33 is None:
            K_33 = estimate_intrinsics(H=frame_height, W=frame_width)

        # Adjust input size based on aspect ratio.
        ratio: float = max(frame_height, frame_width) / min(frame_height, frame_width)
        if ratio > 1.78:
            self.input_size = int(self.input_size * 1.777 / ratio)
            self.input_size = round(self.input_size / 14) * 14

        transform = Compose(
            [
                Resize(
                    width=self.input_size,
                    height=self.input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

        frame_list: list[UInt8[np.ndarray, "H W 3"]] = [rgb_frames[i] for i in range(rgb_frames.shape[0])]
        padded_frame_list, org_video_len, frame_step = pad_frames(frame_list, INFER_LEN, OVERLAP)

        depth_list = []
        pre_input = None
        # Process each chunk and yield raw (unaligned) depth results.
        for frame_id in tqdm(
            range(0, org_video_len, frame_step),
            desc="Processing video chunks (iterator)",
            unit="chunk",
        ):
            output: tuple[list[Float32[np.ndarray, "H W"]], torch.Tensor] = self.process_video_chunk(
                padded_frame_list,
                frame_id,
                transform,
                self.device,
                fp32=False,
                pre_input=pre_input,
                frame_height=frame_height,
                frame_width=frame_width,
            )
            chunk_depth: list[Float32[np.ndarray, "H W"]] = output[0]
            pre_input = output[1]
            depth_list += chunk_depth

            yield DepthChunkIntermediate(
                chunk_index=frame_id,
                raw_depth_list=chunk_depth,
                progress=frame_id / org_video_len,
            )

        del frame_list
        gc.collect()

        # Once all chunks are processed, perform alignment.
        aligned_depth_list: list[Float32[np.ndarray, "H W"]] = align_depths(
            depth_list, KEYFRAMES, INFER_LEN, OVERLAP, INTERP_LEN
        )
        aligned_depth_list = aligned_depth_list[:org_video_len]

        relative_depth_list: list[RelativeDepthPrediction] = [
            RelativeDepthPrediction(
                disparity=disparity,
                depth=disparity_to_depth(disparity, focal_length=int(K_33[0, 0])),
                confidence=np.ones_like(disparity),
                K_33=K_33,
            )
            for disparity in aligned_depth_list
        ]

        # Yield the final aligned result so the visualization can be updated.
        yield FinalDepthResult(aligned_depth=relative_depth_list)

    def process_video_chunk(
        self,
        frame_list: list[UInt8[np.ndarray, "H W 3"]],
        frame_id: int,
        transform: object,
        device: Literal["cpu", "cuda"],
        fp32: bool,
        pre_input: torch.Tensor | None,
        frame_height: int,
        frame_width: int,
    ) -> tuple[list[Float32[np.ndarray, "H W"]], torch.Tensor]:
        """
        Processes a single chunk of video frames to infer depth maps.

        This function preprocesses each frame in the chunk, prepares a tensor for the model,
        optionally replaces the first overlapping frames with keyframes from the previous chunk,
        and performs inference. The resulting depth maps are then resized back to the original frame dimensions.

        Args:
            frame_list: List of video frames.
            frame_id: Starting index for the current chunk.
            transform: Preprocessing transform (resize, normalization, etc.).
            device: Device to run inference on ("cuda" or "cpu").
            fp32: Whether to use FP32 precision (False uses mixed precision).
            pre_input: The previous chunk’s processed input tensor (used for overlapping frame replacement).
            frame_height: Original frame height.
            frame_width: Original frame width.

        Returns:
            A tuple containing:
                - depth_chunk: List of inferred depth maps (numpy arrays) for the current chunk.
                - cur_input: The processed input tensor for the current chunk, to be reused as pre_input in the next iteration.
        """
        cur_list = []
        for i in range(INFER_LEN):
            # Preprocess the current frame.
            preprocessed_frame: Float32[np.ndarray, "3 H W"] = transform(
                {"image": frame_list[frame_id + i].astype(np.float32) / 255.0}
            )["image"]
            img_tensor: Float32[torch.Tensor, "3 H W"] = torch.from_numpy(preprocessed_frame)
            # Rearrange to add batch and time dimensions.
            img_chunk: Float32[torch.Tensor, "1 1 3 H W"] = rearrange(img_tensor, "c h w -> 1 1 c h w")
            cur_list.append(img_chunk)

        # Concatenate frames along the time dimension.
        cur_input: Float32[torch.Tensor, "1 T 3 H W"] = torch.cat(cur_list, dim=1).to(device)

        # Replace the first OVERLAP frames with keyframes from previous chunk if available.
        if pre_input is not None:
            cur_input[:, :OVERLAP, ...] = pre_input[:, KEYFRAMES, ...]

        with torch.no_grad(), torch.autocast(device_type=device, enabled=(not fp32)):
            depth: Float32[torch.Tensor, "1 T H W"] = self.model.forward(cur_input)  # Inference

        depth = depth.to(cur_input.dtype)
        # Resize the predicted depth maps to match the original frame size.
        depth = F.interpolate(
            depth.flatten(0, 1).unsqueeze(1),
            size=(frame_height, frame_width),
            mode="bilinear",
            align_corners=True,
        )
        # Convert each depth map to a numpy array.
        depth_chunk = [depth[i][0].numpy(force=True) for i in range(depth.shape[0])]

        return depth_chunk, cur_input
