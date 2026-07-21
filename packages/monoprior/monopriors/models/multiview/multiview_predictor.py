"""Typed multi-view predictor with backend-owned model and pose conventions."""

import gc
from _thread import LockType
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from timeit import default_timer as timer
from typing import Literal, TypeAlias, TypeGuard, get_args

import numpy as np
import torch
from jaxtyping import Float, Float32, UInt8
from numpy import ndarray
from simplecv.camera_orient_utils import auto_orient_and_center_poses, rotation_matrix_between
from simplecv.ops.conventions import CameraConventions, convert_pose
from torch import Tensor
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri as decode_vggt_camera_head

from monopriors.models.multiview.multiview_model import (
    MultiviewModelPredictions,
    MultiviewPred,
    PreprocessResults,
    amp_autocast,
    generate_multiview_pred,
    preprocess_images,
)
from monopriors.third_party.g3t.models.g3t import G3T
from monopriors.third_party.g3t.utils.pose_enc import pose_encoding_to_extri_intri as decode_g3t_pose_encoding

MultiviewModelName: TypeAlias = Literal["vggt", "g3t"]
CenterMethod: TypeAlias = Literal["poses", "focus", "none"]
ImagePreprocessingMode: TypeAlias = Literal["crop", "pad"]
BackendFactory: TypeAlias = Callable[["MultiviewPredictorConfig"], "MultiviewBackend"]

MULTIVIEW_MODEL_NAMES: tuple[str, ...] = get_args(MultiviewModelName)
IMAGE_PREPROCESSING_MODES: tuple[str, ...] = get_args(ImagePreprocessingMode)

G3T_REPO_ID: str = "thatbrguy/g3t"
G3T_CHECKPOINT_REVISION: str = "c55e91a04f1cbbad67359072536351201fd19e8b"


def is_multiview_model_name(value: str) -> TypeGuard[MultiviewModelName]:
    """Return whether a string names a supported multi-view backend."""
    return value in MULTIVIEW_MODEL_NAMES


def is_image_preprocessing_mode(value: str) -> TypeGuard[ImagePreprocessingMode]:
    """Return whether a string names a supported preprocessing mode."""
    return value in IMAGE_PREPROCESSING_MODES


@dataclass(frozen=True, slots=True)
class MultiviewPredictorConfig:
    """Construction settings requested for a predictor instance."""

    model_name: MultiviewModelName = "vggt"
    """Model backend: standard VGGT or gravity-aligned G3T."""
    device: Literal["cuda", "cpu"] = "cuda"
    """Torch execution device."""
    local_files_only: bool = False
    """Require checkpoints to be present in the local Hugging Face cache."""
    g3t_compile: bool = False
    """Compile G3T for fixed-shape CUDA workloads; ignored by VGGT and CPU runs."""

    @property
    def cache_key(self) -> "MultiviewPredictorKey":
        """Return the settings that change the resident predictor."""
        return MultiviewPredictorKey(
            model_name=self.model_name,
            device=self.device,
            g3t_compile=self.model_name == "g3t" and self.device == "cuda" and self.g3t_compile,
        )


@dataclass(frozen=True, slots=True)
class MultiviewPredictorKey:
    """Canonical runtime identity for a cached predictor."""

    model_name: MultiviewModelName
    device: Literal["cuda", "cpu"]
    g3t_compile: bool


@dataclass(slots=True)
class DecodedBackendOutput:
    """Raw model output and decoded camera tensors from one backend."""

    raw_predictions: dict[str, Tensor]
    """Backend output containing the shared depth and confidence fields."""
    cam_T_world_b34: Float32[Tensor, "batch num_cams 3 4"]
    """Decoded RDF world-to-camera extrinsics."""
    intrinsic_b33: Float32[Tensor, "batch num_cams 3 3"]
    """Decoded per-camera pinhole intrinsics."""


@dataclass(slots=True)
class MultiviewBackendPredictions:
    """Backend-neutral predictions at the Torch-to-NumPy boundary."""

    depth: Float32[Tensor, "batch num_cams H W 1"]
    """Per-camera depth maps."""
    depth_conf: Float32[Tensor, "batch num_cams H W"]
    """Per-pixel depth confidence."""
    intrinsic: Float32[ndarray, "batch num_cams 3 3"]
    """Per-camera pinhole intrinsics."""
    cam_T_world_b34: Float32[ndarray, "batch num_cams 3 4"]
    """World-to-camera extrinsics in Monopriors' canonical +Z-up frame."""


class MultiviewBackend(ABC):
    """Model adapter that owns checkpoint loading, decoding, and world convention."""

    def predict(
        self,
        images: Float32[Tensor, "num_cams 3 H W"],
        *,
        center_method: CenterMethod,
    ) -> MultiviewBackendPredictions:
        """Predict the common depth and camera contract in the canonical +Z-up frame."""
        decoded: DecodedBackendOutput = self._run_and_decode(images)
        world_T_cam_gl: Float[ndarray, "num_cams 4 4"] = _world_poses_from_camera_extrinsics(
            decoded.cam_T_world_b34
        )
        canonical_world_T_cam_b34: Float[ndarray, "num_cams 3 4"] = self._canonicalize_world_poses(
            world_T_cam_gl,
            center_method=center_method,
        )
        return MultiviewBackendPredictions(
            depth=decoded.raw_predictions["depth"],
            depth_conf=decoded.raw_predictions["depth_conf"],
            intrinsic=decoded.intrinsic_b33.numpy(force=True).astype(np.float32),
            cam_T_world_b34=_camera_extrinsics_from_world_poses(canonical_world_T_cam_b34),
        )

    @abstractmethod
    def _run_and_decode(
        self,
        images: Float32[Tensor, "num_cams 3 H W"],
    ) -> DecodedBackendOutput:
        """Run a backend and decode its native camera heads."""

    @abstractmethod
    def _canonicalize_world_poses(
        self,
        world_T_cam_gl: Float[ndarray, "num_cams 4 4"],
        *,
        center_method: CenterMethod,
    ) -> Float[ndarray, "num_cams 3 4"]:
        """Apply the backend's world-orientation policy."""

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


def _inference_dtype(device: Literal["cuda", "cpu"]) -> torch.dtype:
    """Select the shared mixed-precision dtype for a model device."""
    if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


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
) -> Float32[ndarray, "batch num_cams 3 4"]:
    """Convert RUB camera-to-world poses to an RDF world-to-camera NumPy batch."""
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
    return cam_T_world_b44[None, :, :3, :]


class VGGTBackend(MultiviewBackend):
    """VGGT adapter that estimates up from cameras and returns the common contract."""

    def __init__(self, config: MultiviewPredictorConfig) -> None:
        self.device: Literal["cuda", "cpu"] = config.device
        self.dtype: torch.dtype = _inference_dtype(config.device)
        self.model: VGGT = VGGT.from_pretrained(
            "facebook/VGGT-1B", local_files_only=config.local_files_only
        ).to(config.device)
        self.model.eval()

    def _run_and_decode(
        self,
        images: Float32[Tensor, "num_cams 3 H W"],
    ) -> DecodedBackendOutput:
        with torch.no_grad(), amp_autocast(device_type=self.device, dtype=self.dtype):
            raw_predictions: dict[str, Tensor] = self.model(images)
        image_size_hw: tuple[int, int] = (images.shape[-2], images.shape[-1])
        decoded = decode_vggt_camera_head(raw_predictions["pose_enc"], image_size_hw)
        cam_T_world_b34: Float32[Tensor, "batch num_cams 3 4"] = decoded[0]
        intrinsic_b33: Float32[Tensor, "batch num_cams 3 3"] | None = decoded[1]
        assert intrinsic_b33 is not None
        return DecodedBackendOutput(
            raw_predictions=raw_predictions,
            cam_T_world_b34=cam_T_world_b34,
            intrinsic_b33=intrinsic_b33,
        )

    def _canonicalize_world_poses(
        self,
        world_T_cam_gl: Float[ndarray, "num_cams 4 4"],
        *,
        center_method: CenterMethod,
    ) -> Float[ndarray, "num_cams 3 4"]:
        """Estimate VGGT's up direction from its camera poses."""
        oriented_world_T_cam_b34, _ = auto_orient_and_center_poses(
            world_T_cam_gl.astype(np.float64), method="up", center_method=center_method
        )
        return oriented_world_T_cam_b34

    def close(self) -> None:
        self.model.cpu()


class G3TBackend(MultiviewBackend):
    """G3T adapter that preserves predicted gravity and normalizes it to +Z up."""

    def __init__(self, config: MultiviewPredictorConfig) -> None:
        self.device: Literal["cuda", "cpu"] = config.device
        self.dtype: torch.dtype = _inference_dtype(config.device)
        self.model: G3T = G3T.from_pretrained(
            G3T_REPO_ID,
            revision=G3T_CHECKPOINT_REVISION,
            local_files_only=config.local_files_only,
        ).to(config.device)
        self.model.eval()
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
        self.model.aggregator.warm_shape_caches(
            num_cams=num_cams,
            height=height,
            width=width,
            device=images.device,
            dtype=images.dtype,
        )
        self._warmed_input_shapes.add(input_shape)

    def _run_and_decode(
        self,
        images: Float32[Tensor, "num_cams 3 H W"],
    ) -> DecodedBackendOutput:
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
        return DecodedBackendOutput(
            raw_predictions=raw_predictions,
            cam_T_world_b34=cam_T_world_b34,
            intrinsic_b33=intrinsic_b33,
        )

    def _canonicalize_world_poses(
        self,
        world_T_cam_gl: Float[ndarray, "num_cams 4 4"],
        *,
        center_method: CenterMethod,
    ) -> Float[ndarray, "num_cams 3 4"]:
        """Preserve G3T's gravity direction while applying optional centering."""
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
        return canonical_world_T_cam_b34

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
        print("Model loaded in", timer() - load_start, "seconds")

    def __call__(
        self,
        rgb_list: list[UInt8[ndarray, "H W 3"]],
        *,
        preprocessing_mode: ImagePreprocessingMode = "pad",
        center_method: CenterMethod = "none",
    ) -> list[MultiviewPred]:
        preprocess_results: PreprocessResults = preprocess_images(rgb_list, mode=preprocessing_mode)
        images: Float32[Tensor, "num_cams 3 H W"] = preprocess_results.images.to(self.config.device)
        print("Running inference...")
        backend_predictions: MultiviewBackendPredictions = self.backend.predict(
            images,
            center_method=center_method,
        )
        predictions: MultiviewModelPredictions = MultiviewModelPredictions(
            depth=backend_predictions.depth.numpy(force=True),
            depth_conf=backend_predictions.depth_conf.numpy(force=True),
            intrinsic=backend_predictions.intrinsic,
            cam_T_world_b34=backend_predictions.cam_T_world_b34,
        )
        return generate_multiview_pred(
            predictions,
            img_tensors=images,
            rgb_list=rgb_list,
            metadata_list=preprocess_results.metadata if preprocessing_mode == "pad" else None,
        )

    def close(self) -> None:
        """Release this predictor's backend resources."""
        self.backend.close()


class MultiviewPredictorCache:
    """Single-slot predictor cache with atomic acquire/use/replacement."""

    def __init__(self, factory: Callable[[MultiviewPredictorConfig], MultiviewPredictor] = MultiviewPredictor) -> None:
        self._factory: Callable[[MultiviewPredictorConfig], MultiviewPredictor] = factory
        self._lock: LockType = Lock()
        self._key: MultiviewPredictorKey | None = None
        self._predictor: MultiviewPredictor | None = None

    @contextmanager
    def acquire(self, config: MultiviewPredictorConfig) -> Generator[MultiviewPredictor, None, None]:
        """Yield the exact requested predictor while preventing concurrent replacement."""
        with self._lock:
            requested_key: MultiviewPredictorKey = config.cache_key
            if self._key != requested_key:
                if self._predictor is not None:
                    self._predictor.close()
                self._predictor = None
                self._key = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._predictor = self._factory(config)
                self._key = requested_key
            if self._predictor is None:
                raise RuntimeError("Predictor cache failed to construct a predictor.")
            yield self._predictor
