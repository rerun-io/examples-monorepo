"""Typed multi-view predictor with backend-owned model and pose conventions."""

import gc
from _thread import LockType
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from timeit import default_timer as timer
from typing import Literal, TypeAlias, cast

import numpy as np
import torch
from jaxtyping import Float, Float32, UInt8
from numpy import ndarray
from simplecv.camera_orient_utils import auto_orient_and_center_poses, rotation_matrix_between
from simplecv.ops.conventions import CameraConventions, convert_pose
from torch import Tensor
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri as decode_vggt_camera_head

from monopriors.models.multiview.vggt_model import (
    MultiviewModelPredictions,
    MultiviewPred,
    PreprocessResults,
    amp_autocast,
    generate_multiview_pred,
    preprocess_images,
)
from monopriors.third_party.g3t.layers.attention import Attention
from monopriors.third_party.g3t.models.g3t import G3T
from monopriors.third_party.g3t.utils.pose_enc import pose_encoding_to_extri_intri as decode_g3t_pose_encoding

MultiviewModelName: TypeAlias = Literal["vggt", "g3t"]
CenterMethod: TypeAlias = Literal["poses", "focus", "none"]
BackendFactory: TypeAlias = Callable[["MultiviewPredictorConfig"], "MultiviewBackend"]

G3T_REPO_ID: str = "thatbrguy/g3t"
G3T_CHECKPOINT_REVISION: str = "c55e91a04f1cbbad67359072536351201fd19e8b"


@dataclass(frozen=True, slots=True)
class MultiviewPredictorConfig:
    """Construction settings that uniquely identify a predictor instance."""

    model_name: MultiviewModelName = "vggt"
    """Model backend: standard VGGT or gravity-aligned G3T."""
    device: Literal["cuda", "cpu"] = "cuda"
    """Torch execution device."""
    preprocessing_mode: Literal["crop", "pad"] = "pad"
    """Image preprocessing strategy."""
    local_files_only: bool = False
    """Require checkpoints to be present in the local Hugging Face cache."""
    g3t_compile: bool = True
    """Compile G3T on CUDA for lower warm inference latency; ignored by VGGT and CPU runs."""


@dataclass(slots=True)
class MultiviewTensorPredictions:
    """Backend-neutral tensors needed by Monopriors."""

    depth: Float32[Tensor, "batch num_cams H W 1"]
    """Per-camera depth maps."""
    depth_conf: Float32[Tensor, "batch num_cams H W"]
    """Per-pixel depth confidence."""
    intrinsic: Float32[Tensor, "batch num_cams 3 3"]
    """Per-camera pinhole intrinsics."""
    cam_T_world_b34: Float32[Tensor, "batch num_cams 3 4"]
    """World-to-camera extrinsics in Monopriors' canonical +Z-up frame."""


class MultiviewBackend(ABC):
    """Model adapter that owns checkpoint loading, decoding, and world convention."""

    @abstractmethod
    def predict(
        self,
        images: Float32[Tensor, "num_cams 3 H W"],
        *,
        center_method: CenterMethod,
    ) -> MultiviewTensorPredictions:
        """Predict the minimal common tensor contract in Monopriors' +Z-up world frame."""

    @abstractmethod
    def close(self) -> None:
        """Release model-owned accelerator memory."""


def decode_g3t_camera_heads(
    *,
    local_pose_encoding: Float32[Tensor, "batch num_cams 6"],
    relative_pose_encoding: Float32[Tensor, "batch num_cams 5"],
    image_size_hw: tuple[int, int],
) -> tuple[Float32[Tensor, "batch num_cams 3 4"], Float32[Tensor, "batch num_cams 3 3"]]:
    """Decode and compose G3T's gravity-to-camera and world-to-gravity heads."""
    cam_T_gravity_b34, intrinsic_b33 = decode_g3t_pose_encoding(
        local_pose_encoding,
        image_size_hw,
        pose_encoding_type="noT_quaR_FoV",
    )
    gravity_T_world_b34, _ = decode_g3t_pose_encoding(
        relative_pose_encoding,
        image_size_hw,
        pose_encoding_type="absT_quaRy_noFoV",
    )
    assert intrinsic_b33 is not None

    bottom_row: Float32[Tensor, "batch num_cams 1 4"] = torch.zeros(
        (*cam_T_gravity_b34.shape[:-2], 1, 4),
        dtype=cam_T_gravity_b34.dtype,
        device=cam_T_gravity_b34.device,
    )
    bottom_row[..., 0, 3] = 1.0
    cam_T_gravity_b44: Float32[Tensor, "batch num_cams 4 4"] = torch.cat([cam_T_gravity_b34, bottom_row], dim=-2)
    gravity_T_world_b44: Float32[Tensor, "batch num_cams 4 4"] = torch.cat(
        [gravity_T_world_b34, bottom_row], dim=-2
    )
    cam_T_world_b44: Float32[Tensor, "batch num_cams 4 4"] = cam_T_gravity_b44 @ gravity_T_world_b44
    return cam_T_world_b44[..., :3, :], intrinsic_b33


def _world_poses_from_camera_extrinsics(
    cam_T_world_b34: Float32[Tensor, "batch num_cams 3 4"],
) -> Float[ndarray, "num_cams 4 4"]:
    """Convert a singleton batch of RDF world-to-camera poses to RUB camera-to-world poses."""
    cam_T_world_np: Float32[ndarray, "batch num_cams 3 4"] = cam_T_world_b34.numpy(force=True).astype(np.float32)
    if cam_T_world_np.shape[0] != 1:
        raise ValueError("Only batch size 1 is supported for multi-view inference.")
    num_cams: int = cam_T_world_np.shape[1]
    bottom_row: Float32[ndarray, "num_cams 1 4"] = np.broadcast_to(
        np.array([[[0.0, 0.0, 0.0, 1.0]]], dtype=np.float32),
        (num_cams, 1, 4),
    )
    cam_T_world_b44: Float32[ndarray, "num_cams 4 4"] = np.concatenate(
        [cam_T_world_np[0], bottom_row], axis=1
    )
    world_T_cam_cv: Float32[ndarray, "num_cams 4 4"] = np.linalg.inv(cam_T_world_b44).astype(np.float32)
    return convert_pose(world_T_cam_cv, CameraConventions.CV, CameraConventions.GL)


def _camera_extrinsics_from_world_poses(
    world_T_cam_gl_b34: Float[ndarray, "num_cams 3 4"],
) -> Float32[Tensor, "batch num_cams 3 4"]:
    """Convert RUB camera-to-world poses back to a tensor batch of RDF world-to-camera extrinsics."""
    num_cams: int = world_T_cam_gl_b34.shape[0]
    bottom_row: Float[ndarray, "num_cams 1 4"] = np.broadcast_to(
        np.array([[[0.0, 0.0, 0.0, 1.0]]]),
        (num_cams, 1, 4),
    )
    world_T_cam_gl_b44: Float[ndarray, "num_cams 4 4"] = np.concatenate(
        [world_T_cam_gl_b34, bottom_row], axis=1
    )
    world_T_cam_cv_b44: Float[ndarray, "num_cams 4 4"] = convert_pose(
        world_T_cam_gl_b44, CameraConventions.GL, CameraConventions.CV
    )
    cam_T_world_b44: Float32[ndarray, "num_cams 4 4"] = np.linalg.inv(world_T_cam_cv_b44).astype(np.float32)
    return torch.from_numpy(cam_T_world_b44[None, :, :3, :])


class VGGTBackend(MultiviewBackend):
    """VGGT adapter that estimates up from cameras and returns the common contract."""

    def __init__(self, config: MultiviewPredictorConfig) -> None:
        self.device: Literal["cuda", "cpu"] = config.device
        self.dtype: torch.dtype = (
            torch.bfloat16
            if config.device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )
        self.model: VGGT = VGGT.from_pretrained(
            "facebook/VGGT-1B", local_files_only=config.local_files_only
        ).to(config.device)
        self.model.eval()

    def predict(
        self,
        images: Float32[Tensor, "num_cams 3 H W"],
        *,
        center_method: CenterMethod,
    ) -> MultiviewTensorPredictions:
        with torch.no_grad(), amp_autocast(device_type=self.device, dtype=self.dtype):
            raw_predictions: dict[str, Tensor] = self.model(images)
        image_size_hw: tuple[int, int] = (images.shape[-2], images.shape[-1])
        decoded = decode_vggt_camera_head(raw_predictions["pose_enc"], image_size_hw)
        cam_T_world_b34: Float32[Tensor, "batch num_cams 3 4"] = decoded[0]
        intrinsic_b33: Float32[Tensor, "batch num_cams 3 3"] | None = decoded[1]
        assert intrinsic_b33 is not None
        world_T_cam_gl: Float[ndarray, "num_cams 4 4"] = _world_poses_from_camera_extrinsics(cam_T_world_b34)
        oriented_world_T_cam_b34, _ = auto_orient_and_center_poses(
            world_T_cam_gl.astype(np.float64), method="up", center_method=center_method
        )
        canonical_cam_T_world_b34: Float32[Tensor, "batch num_cams 3 4"] = _camera_extrinsics_from_world_poses(
            oriented_world_T_cam_b34
        ).to(device=cam_T_world_b34.device)
        return MultiviewTensorPredictions(
            depth=raw_predictions["depth"],
            depth_conf=raw_predictions["depth_conf"],
            intrinsic=intrinsic_b33,
            cam_T_world_b34=canonical_cam_T_world_b34,
        )

    def close(self) -> None:
        self.model.cpu()


class G3TBackend(MultiviewBackend):
    """G3T adapter that preserves predicted gravity and normalizes it to +Z up."""

    def __init__(self, config: MultiviewPredictorConfig) -> None:
        self.device: Literal["cuda", "cpu"] = config.device
        self.dtype: torch.dtype = (
            torch.bfloat16
            if config.device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )
        self.model: G3T = G3T.from_pretrained(
            G3T_REPO_ID,
            revision=G3T_CHECKPOINT_REVISION,
            local_files_only=config.local_files_only,
        ).to(config.device)
        self.model.eval()
        self.model.point_head = None
        self._compiled_model: Callable[[Tensor], dict[str, Tensor]] | None = None
        self._warmed_input_shapes: set[tuple[int, int, int]] = set()
        if config.g3t_compile and config.device == "cuda":
            self._compiled_model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)

    def _warm_position_caches(self, images: Float32[Tensor, "num_cams 3 H W"]) -> None:
        """Populate shape-only RoPE caches outside CUDA graph capture."""
        num_cams = images.shape[0]
        height, width = images.shape[-2:]
        input_shape = (num_cams, height, width)
        if input_shape in self._warmed_input_shapes:
            return
        aggregator = self.model.aggregator
        patch_height = height // aggregator.patch_size
        patch_width = width // aggregator.patch_size
        position_getter = aggregator.position_getter
        if position_getter is not None:
            position_getter(num_cams, patch_height, patch_width, device=images.device)
        if aggregator.rope is not None:
            attention = cast(Attention, aggregator.frame_blocks[0].attn)
            rope_feature_dim = attention.head_dim // 2
            frame_token_count = patch_height * patch_width + aggregator.patch_start_idx
            for token_count in (frame_token_count, num_cams * frame_token_count):
                aggregator.rope._compute_frequency_components(
                    rope_feature_dim,
                    token_count,
                    images.device,
                    images.dtype,
                )
        self._warmed_input_shapes.add(input_shape)

    def predict(
        self,
        images: Float32[Tensor, "num_cams 3 H W"],
        *,
        center_method: CenterMethod,
    ) -> MultiviewTensorPredictions:
        inference_model: Callable[[Tensor], dict[str, Tensor]] = self.model
        if self._compiled_model is not None:
            self._warm_position_caches(images)
            torch.compiler.cudagraph_mark_step_begin()
            inference_model = self._compiled_model
        with torch.inference_mode(), amp_autocast(device_type=self.device, dtype=self.dtype):
            raw_predictions: dict[str, Tensor] = inference_model(images)
        image_size_hw: tuple[int, int] = (images.shape[-2], images.shape[-1])
        decoded = decode_g3t_camera_heads(
            local_pose_encoding=raw_predictions["local_pose_enc"],
            relative_pose_encoding=raw_predictions["global_pose_enc"],
            image_size_hw=image_size_hw,
        )
        cam_T_world_b34: Float32[Tensor, "batch num_cams 3 4"] = decoded[0]
        intrinsic_b33: Float32[Tensor, "batch num_cams 3 3"] = decoded[1]
        world_T_cam_gl: Float[ndarray, "num_cams 4 4"] = _world_poses_from_camera_extrinsics(cam_T_world_b34)
        if center_method == "none":
            centered_world_T_cam_b34: Float[ndarray, "num_cams 3 4"] = world_T_cam_gl[:, :3, :].astype(np.float64)
        else:
            centered_world_T_cam_b34, _ = auto_orient_and_center_poses(
                world_T_cam_gl.astype(np.float64), method="none", center_method=center_method
            )
        gravity_rotation_33: Float[ndarray, "3 3"] = rotation_matrix_between(
            np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])
        )
        canonical_world_T_cam_b34: Float[ndarray, "num_cams 3 4"] = (
            gravity_rotation_33 @ centered_world_T_cam_b34
        )
        canonical_cam_T_world_b34: Float32[Tensor, "batch num_cams 3 4"] = _camera_extrinsics_from_world_poses(
            canonical_world_T_cam_b34
        ).to(device=cam_T_world_b34.device)
        return MultiviewTensorPredictions(
            depth=raw_predictions["depth"],
            depth_conf=raw_predictions["depth_conf"],
            intrinsic=intrinsic_b33,
            cam_T_world_b34=canonical_cam_T_world_b34,
        )

    def close(self) -> None:
        self.model.cpu()


BACKEND_FACTORIES: dict[MultiviewModelName, BackendFactory] = {
    "vggt": VGGTBackend,
    "g3t": G3TBackend,
}


class MultiviewPredictor:
    """Preprocess images and adapt one selected backend to ``MultiviewPred``."""

    def __init__(self, config: MultiviewPredictorConfig) -> None:
        self.config: MultiviewPredictorConfig = config
        load_start: float = timer()
        print(f"Loading {config.model_name.upper()} model...")
        self.backend: MultiviewBackend = BACKEND_FACTORIES[config.model_name](config)
        self._inference_lock: LockType = Lock()
        print("Model loaded in", timer() - load_start, "seconds")

    def __call__(
        self,
        rgb_list: list[UInt8[ndarray, "H W 3"]],
        *,
        center_method: CenterMethod = "none",
    ) -> list[MultiviewPred]:
        preprocess_results: PreprocessResults = preprocess_images(rgb_list, mode=self.config.preprocessing_mode)
        images: Float32[Tensor, "num_cams 3 H W"] = preprocess_results.images.to(self.config.device)
        print("Running inference...")
        with self._inference_lock:
            tensor_predictions: MultiviewTensorPredictions = self.backend.predict(
                images,
                center_method=center_method,
            )
        predictions: MultiviewModelPredictions = MultiviewModelPredictions(
            depth=tensor_predictions.depth.numpy(force=True),
            depth_conf=tensor_predictions.depth_conf.numpy(force=True),
            intrinsic=tensor_predictions.intrinsic.numpy(force=True),
            cam_T_world_b34=tensor_predictions.cam_T_world_b34.numpy(force=True),
        )
        return generate_multiview_pred(
            predictions,
            img_tensors=images,
            rgb_list=rgb_list,
            metadata_list=preprocess_results.metadata if self.config.preprocessing_mode == "pad" else None,
            fast_rgb=self.config.model_name == "g3t",
        )

    def close(self) -> None:
        """Release this predictor's backend resources."""
        self.backend.close()


class MultiviewPredictorCache:
    """Single-slot predictor cache with atomic acquire/use/replacement."""

    def __init__(self, factory: Callable[[MultiviewPredictorConfig], MultiviewPredictor] = MultiviewPredictor) -> None:
        self._factory: Callable[[MultiviewPredictorConfig], MultiviewPredictor] = factory
        self._lock: LockType = Lock()
        self._config: MultiviewPredictorConfig | None = None
        self._predictor: MultiviewPredictor | None = None

    @contextmanager
    def acquire(self, config: MultiviewPredictorConfig) -> Generator[MultiviewPredictor, None, None]:
        """Yield the exact requested predictor while preventing concurrent replacement."""
        with self._lock:
            if self._config != config:
                if self._predictor is not None:
                    self._predictor.close()
                self._predictor = None
                self._config = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._predictor = self._factory(config)
                self._config = config
            if self._predictor is None:
                raise RuntimeError("Predictor cache failed to construct a predictor.")
            yield self._predictor
