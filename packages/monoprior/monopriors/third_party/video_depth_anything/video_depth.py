# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gc
from typing import Literal

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float32, UInt8
from torchvision.transforms import Compose
from tqdm import tqdm

from monopriors.scale_utils import compute_scale_and_shift, get_interpolate_frames
from monopriors.third_party.depth_anything_v2.dinov2 import DINOv2
from monopriors.third_party.depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize

from .dpt_temporal import DPTHeadTemporal

# infer settings, do not change
INFER_LEN = 32
OVERLAP = 10
KEYFRAMES = [0, 12, 24, 25, 26, 27, 28, 29, 30, 31]
INTERP_LEN = 8


class VideoDepthAnything(nn.Module):
    def __init__(
        self,
        encoder="vitl",
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        num_frames=32,
        pe="ape",
    ):
        super(VideoDepthAnything, self).__init__()

        self.intermediate_layer_idx = {"vits": [2, 5, 8, 11], "vitl": [4, 11, 17, 23]}

        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)

        self.head = DPTHeadTemporal(
            self.pretrained.embed_dim,
            features,
            use_bn,
            out_channels=out_channels,
            use_clstoken=use_clstoken,
            num_frames=num_frames,
            pe=pe,
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        patch_h, patch_w = H // 14, W // 14
        features = self.pretrained.get_intermediate_layers(
            x.flatten(0, 1), self.intermediate_layer_idx[self.encoder], return_class_token=True
        )
        depth = self.head(features, patch_h, patch_w, T)
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        return depth.squeeze(1).unflatten(0, (B, T))  # return shape [B, T, H, W]

    def infer_video_depth(
        self,
        frames: UInt8[np.ndarray, "T H W 3"],
        input_size: int = 518,
        device: Literal["cpu", "cuda"] = "cuda",
        fp32: bool = False,
    ) -> Float32[np.ndarray, "T H W"]:
        """
        Infers depth maps from video frames using a pre-trained deep learning model.

        This method processes a sequence of video frames to produce corresponding depth maps.
        It handles large videos by processing them in chunks with overlapping segments,
        ensuring temporal consistency across chunk boundaries through alignment and interpolation.

        Args:
            frames (numpy.ndarray): Array of rgb frames with shape [num_frames, height, width, 3]
            input_size (int, optional): Base input size for the neural network. Defaults to 518.
                May be adjusted based on aspect ratio.
            device (str, optional): Device to run inference on ("cuda" or "cpu"). Defaults to "cuda".
            fp32 (bool, optional): Whether to use FP32 precision instead of mixed precision.
                Defaults to False (uses mixed precision).

        Returns:
            - numpy.ndarray: Depth maps with shape [num_frames, height, width]

        Notes:
            - Video processing is optimized for aspect ratios up to 16:9. Videos with more extreme
              aspect ratios will have input_size automatically adjusted to fit memory constraints.
            - The method employs a sliding window approach with overlapping frames to maintain
              temporal consistency.
            - Scale and shift alignment is performed between consecutive chunks to ensure
              smooth transitions.
        """
        frame_height: int = frames.shape[1]
        frame_width: int = frames.shape[2]

        ratio: float = max(frame_height, frame_width) / min(frame_height, frame_width)
        if ratio > 1.78:  # we recommend to process video with ratio smaller than 16:9 due to memory limitation
            input_size = int(input_size * 1.777 / ratio)
            input_size: int = round(input_size / 14) * 14

        transform = Compose(
            [
                Resize(
                    width=input_size,
                    height=input_size,
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

        frame_list: list[UInt8[np.ndarray, "H W 3"]] = [frames[i] for i in range(frames.shape[0])]
        # Calculate step size between chunks (22 frames): each chunk processes INFER_LEN (32) frames
        # with OVERLAP (10) frames shared between consecutive chunks for temporal consistency
        frame_step: int = INFER_LEN - OVERLAP
        org_video_len: int = len(frame_list)
        append_frame_len: int = (frame_step - (org_video_len % frame_step)) % frame_step + (INFER_LEN - frame_step)
        # Pad video with copies of last frame to ensure proper length for chunk processing
        frame_list = frame_list + [frame_list[-1].copy()] * append_frame_len

        depth_list = []
        pre_input = None
        for frame_id in tqdm(range(0, org_video_len, frame_step), desc="Processing video chunks", unit="chunk"):
            cur_list: list[Float32[torch.Tensor, "1 1 3 H W"]] = []
            # generate frame_chunks of size INFER_LEN
            # each chunk has a size of INFER_LEN, with the first OVERLAP frames shared with the previous chunk
            for i in range(INFER_LEN):
                preprocessed_frame: Float32[np.ndarray, "3 H W"] = transform(
                    {"image": frame_list[frame_id + i].astype(np.float32) / 255.0}
                )["image"]
                img_tensor: Float32[torch.Tensor, "3 H W"] = torch.from_numpy(preprocessed_frame)
                img_chunk: Float32[torch.Tensor, "1 1 3 H W"] = rearrange(img_tensor, "c h w -> 1 1 c h w")

                cur_list.append(img_chunk)
            # concatenate the chunks along the time dimension
            cur_input: Float32[torch.Tensor, "1 T 3 H W"] = torch.cat(cur_list, dim=1).to(device)
            if pre_input is not None:
                cur_input[:, :OVERLAP, ...] = pre_input[:, KEYFRAMES, ...]

            with torch.no_grad():
                with torch.autocast(device_type=device, enabled=(not fp32)):
                    depth: Float32[torch.Tensor, "1 T H W"] = self.forward(cur_input)  # depth shape: [1, T, H, W]

            depth = depth.to(cur_input.dtype)
            depth: Float32[torch.Tensor, "T 1 H W"] = F.interpolate(
                depth.flatten(0, 1).unsqueeze(1), size=(frame_height, frame_width), mode="bilinear", align_corners=True
            )
            depth_list += [depth[i][0].numpy(force=True) for i in range(depth.shape[0])]

            pre_input = cur_input

        del frame_list
        gc.collect()

        depth_list_aligned = []  # List to accumulate aligned depth maps for all video frames.
        ref_align = []  # List to hold reference keyframe depths from the previous chunk for alignment.

        # Determine the number of frames used for alignment in the overlapping region.
        # OVERLAP frames are shared between chunks; INTERP_LEN of these are used for smooth interpolation.
        align_len = OVERLAP - INTERP_LEN

        # Select keyframes (by index) from the initial set for depth alignment.
        kf_align_list: list[int] = KEYFRAMES[:align_len]

        # Process the video in chunks of INFER_LEN frames (e.g., 32 frames per chunk)
        # with an overlapping region of OVERLAP frames (e.g., 10 frames) between consecutive chunks.
        for frame_id in range(0, len(depth_list), INFER_LEN):
            if len(depth_list_aligned) == 0:
                # --- First Chunk Processing ---
                # For the first chunk, there is no previous data to align with.
                # Directly append the first INFER_LEN frames as-is.
                depth_list_aligned += depth_list[:INFER_LEN]
                # Save keyframes from the first chunk to use as a reference for aligning the next chunk.
                for kf_id in kf_align_list:
                    ref_align.append(depth_list[frame_id + kf_id])
            else:
                # --- Subsequent Chunk Processing ---
                # Step 1 & 2: For each new chunk, the first OVERLAP frames are meant to replace the last OVERLAP
                # frames from the previous processing step to create continuity.
                # Gather keyframes from the current chunk (using the initial indices defined in kf_align_list)
                curr_align = []
                for i in range(len(kf_align_list)):
                    curr_align.append(depth_list[frame_id + i])

                # Step 3: Compute alignment parameters (scale and shift) between keyframes of the current chunk (curr_align)
                # and the reference keyframes from the previous chunk (ref_align).
                scale, shift = compute_scale_and_shift(
                    np.concatenate(curr_align), np.concatenate(ref_align), np.concatenate(np.ones_like(ref_align) == 1)
                )

                # Step 5: Create smooth transitions in the overlapping region.
                # Extract the last INTERP_LEN frames from the already aligned previous chunk.
                pre_depth_list = depth_list_aligned[-INTERP_LEN:]
                # Extract the corresponding frames from the current chunk that will be aligned and interpolated.
                post_depth_list = depth_list[frame_id + align_len : frame_id + OVERLAP]
                # Apply the computed scale and shift to the overlapping frames from the new chunk.
                for i in range(len(post_depth_list)):
                    post_depth_list[i] = post_depth_list[i] * scale + shift
                    post_depth_list[i][post_depth_list[i] < 0] = 0  # Clamp negative depth values to 0.
                # Interpolate between the aligned previous chunk and the current chunk's overlapping region to ensure smooth transitions.
                depth_list_aligned[-INTERP_LEN:] = get_interpolate_frames(pre_depth_list, post_depth_list)

                # Step 4: Apply the alignment parameters to the rest of the new chunk (frames beyond the overlapping region).
                for i in range(OVERLAP, INFER_LEN):
                    new_depth = depth_list[frame_id + i] * scale + shift
                    new_depth[new_depth < 0] = 0
                    depth_list_aligned.append(new_depth)

                # Step 6: Update the reference keyframes (ref_align) for the next chunk.
                # Retain the first keyframe from the previous alignment and update subsequent keyframes using the current chunk.
                ref_align = ref_align[:1]
                for kf_id in kf_align_list[1:]:
                    new_depth = depth_list[frame_id + kf_id] * scale + shift
                    new_depth[new_depth < 0] = 0
                    ref_align.append(new_depth)

        # At the end, the original depth_list is replaced with the fully aligned depths,
        # ensuring consistent depth scales across the entire video despite chunk-based processing.
        depth_list: list[Float32[np.ndarray, "H W"]] = depth_list_aligned

        return np.stack(depth_list[:org_video_len], axis=0)
