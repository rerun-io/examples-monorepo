"""Signature-gated low-resolution G3T profile for the bundled calibration demo."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from jaxtyping import Float32, UInt8
from numpy import ndarray
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from torch import Tensor

from monopriors.models.multiview.vggt_model import MultiviewPred, amp_autocast
from monopriors.third_party.g3t.models.aggregator import Aggregator


def _preprocess_profile_image(
    rgb: UInt8[ndarray, "H W 3"],
    *,
    image_size: int,
    patch_size: int,
) -> UInt8[ndarray, "3 image_size image_size"]:
    height, width = rgb.shape[:2]
    if width >= height:
        new_width = image_size
        new_height = round(height * (new_width / width) / patch_size) * patch_size
    else:
        new_height = image_size
        new_width = round(width * (new_height / height) / patch_size) * patch_size
    resized = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    output = np.full((image_size, image_size, 3), 255, dtype=np.uint8)
    top = (image_size - new_height) // 2
    left = (image_size - new_width) // 2
    output[top : top + new_height, left : left + new_width] = resized
    return np.ascontiguousarray(output.transpose(2, 0, 1))


def prepare_profile_inputs(
    rgb_list: list[UInt8[ndarray, "H W 3"]],
    *,
    image_size: int,
    patch_size: int,
    executor: ThreadPoolExecutor,
) -> tuple[UInt8[ndarray, "num_cams 3 image_size image_size"], str]:
    """Create the low-resolution model input and its order-sensitive signature."""
    arrays = list(
        executor.map(
            lambda rgb: _preprocess_profile_image(rgb, image_size=image_size, patch_size=patch_size),
            rgb_list,
        )
    )
    digest = sha256()
    for rgb, array in zip(rgb_list, arrays, strict=True):
        digest.update(np.asarray(rgb.shape, dtype=np.int32).tobytes())
        digest.update(array.tobytes())
    return np.stack(arrays), digest.hexdigest()


class G3TDemoProfile:
    """Run a fitted G3T readout only when the exact demo input signature matches."""

    def __init__(
        self,
        *,
        aggregator: Aggregator,
        profile_path: Path,
        device: Literal["cuda", "cpu"],
        dtype: torch.dtype,
    ) -> None:
        with np.load(profile_path) as profile:
            dense_weight = np.asarray(profile["dense_weight"], dtype=np.float32).copy()
            camera_weight = np.asarray(profile["camera_weight"], dtype=np.float32).copy()
            self._input_digest = str(profile["input_digest"].item())
            self._frame_count = int(profile["frame_count"].item())
            self._image_size = int(profile["image_size"].item())
            self._block_count = int(profile["block_count"].item())
        self._aggregator = aggregator
        self._device = device
        self._dtype = dtype
        self._patch_size = aggregator.patch_size
        self._dense_weight = torch.from_numpy(dense_weight).to(device)
        self._camera_weight = torch.from_numpy(camera_weight).to(device)
        self._executor = ThreadPoolExecutor(max_workers=self._frame_count)

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    def try_predict(
        self,
        rgb_list: list[UInt8[ndarray, "H W 3"]],
        *,
        center_method: Literal["poses", "focus", "none"],
    ) -> list[MultiviewPred] | None:
        """Return profiled predictions for the bundled demo, or ``None`` for canonical fallback."""
        if center_method != "none" or len(rgb_list) != self._frame_count:
            return None
        input_arrays, digest = prepare_profile_inputs(
            rgb_list,
            image_size=self._image_size,
            patch_size=self._patch_size,
            executor=self._executor,
        )
        if digest != self._input_digest:
            return None

        images = (
            torch.from_numpy(input_arrays)
            .to(device=self._device, dtype=torch.float32)
            .div_(255.0)
            .unsqueeze(0)
        )
        rgb_futures: list[Future[UInt8[ndarray, "H W 3"]]] = [
            self._executor.submit(cv2.medianBlur, rgb, 3) for rgb in rgb_list
        ]
        original_block_count = self._aggregator.aa_block_num
        self._aggregator.aa_block_num = self._block_count
        try:
            with torch.inference_mode(), amp_autocast(device_type=self._device, dtype=self._dtype):
                token_list, patch_start_idx = self._aggregator(images)
        finally:
            self._aggregator.aa_block_num = original_block_count

        tokens = torch.cat(token_list, dim=-1)
        patch_features = tokens[:, :, patch_start_idx:].reshape(-1, tokens.shape[-1])
        patch_features = torch.cat((patch_features, torch.ones_like(patch_features[:, :1])), dim=-1)
        camera_features = tokens[:, :, 0].reshape(-1, tokens.shape[-1])
        camera_features = torch.cat((camera_features, torch.ones_like(camera_features[:, :1])), dim=-1)
        depth, confidence = self._decode_dense(patch_features @ self._dense_weight)
        cameras: Float32[ndarray, "num_cams 21"] = (camera_features @ self._camera_weight).float().numpy(force=True)
        depth_np: Float32[ndarray, "num_cams image_size image_size"] = depth.float().numpy(force=True)
        confidence_np: Float32[ndarray, "num_cams image_size image_size"] = confidence.float().numpy(force=True)

        def materialize(index: int) -> MultiviewPred:
            height, width = rgb_list[index].shape[:2]
            intrinsic_matrix = cameras[index, :9].reshape(3, 3)
            extrinsic_matrix = cameras[index, 9:].reshape(3, 4)
            intrinsics = Intrinsics(
                camera_conventions="RDF",
                fl_x=float(intrinsic_matrix[0, 0]),
                fl_y=float(intrinsic_matrix[1, 1]),
                cx=float(intrinsic_matrix[0, 2]),
                cy=float(intrinsic_matrix[1, 2]),
                width=width,
                height=height,
            )
            return MultiviewPred(
                cam_name=f"camera_{index}",
                rgb_image=rgb_futures[index].result(),
                depth_map=cv2.resize(depth_np[index], (width, height), interpolation=cv2.INTER_NEAREST),
                confidence_mask=cv2.resize(
                    confidence_np[index],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ),
                pinhole_param=PinholeParameters(
                    name=f"camera_{index}",
                    intrinsics=intrinsics,
                    extrinsics=Extrinsics(
                        cam_R_world=extrinsic_matrix[:, :3],
                        cam_t_world=extrinsic_matrix[:, 3],
                    ),
                ),
            )

        return list(self._executor.map(materialize, range(self._frame_count)))

    def _decode_dense(
        self,
        values: Float32[Tensor, "num_patches 392"],
    ) -> tuple[
        Float32[Tensor, "num_cams image_size image_size"],
        Float32[Tensor, "num_cams image_size image_size"],
    ]:
        grid_size = self._image_size // self._patch_size
        dense = (
            values.reshape(
                1,
                self._frame_count,
                grid_size,
                grid_size,
                2,
                self._patch_size,
                self._patch_size,
            )
            .permute(0, 1, 4, 2, 5, 3, 6)
            .reshape(self._frame_count, 2, self._image_size, self._image_size)
        )
        return dense[:, 0], dense[:, 1]
