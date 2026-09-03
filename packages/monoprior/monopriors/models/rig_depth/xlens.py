"""X-Lens predictor adapter for calibrated multi-camera rigs."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import torch
from einops import rearrange
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from jaxtyping import Float32, Float64, Int64, UInt8
from numpy import ndarray
from torch import Tensor

from monopriors.models.rig_depth.base_rig_depth import BaseRigDepthPredictor, BaseRigDepthPredictorConfig, RigDepthPrediction
from monopriors.third_party.xlens.inference.pipeline import XLensInference
from monopriors.third_party.xlens.inference.preprocess import IMAGENET_MEAN, IMAGENET_STD, build_cam_types, build_ray_map, canonicalize_c2w
from monopriors.third_party.xlens.models.dinov2.vision_transformer import FrozenRigGeometry
from monopriors.third_party.xlens.models.net import XLensNet, XLensNetOutput

XLENS_HF_REPO: str = "henryzhou998/X-Lens"
XLENS_HF_REVISION: str = "1d0c96353b69464addad12389fadbb816e3978ae"
XLENS_CHECKPOINT: str = "model.safetensors"

AmpDtype: TypeAlias = Literal["bf16", "fp16", "fp32"]
"""CUDA autocast dtype; ``fp32`` disables autocast."""
RigKey: TypeAlias = tuple[str, bytes, bytes | None]
"""Content key of one rig: ray-field digest, camera-type bytes, pose bytes."""

AMP_DTYPES: dict[AmpDtype, torch.dtype | None] = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}


def download_xlens_checkpoint() -> Path:
    """Download the gated non-commercial release from its pinned revision.

    Returns:
        Local path to ``model.safetensors``.

    Raises:
        RuntimeError: If the user's Hugging Face login lacks access to the gated
            ``henryzhou998/X-Lens`` repository.
    """
    try:
        return Path(hf_hub_download(repo_id=XLENS_HF_REPO, filename=XLENS_CHECKPOINT, revision=XLENS_HF_REVISION))
    except HfHubHTTPError as error:
        status_code: int | None = error.response.status_code if error.response is not None else None
        if status_code in (401, 403):
            raise RuntimeError(
                f"X-Lens weights are gated at {XLENS_HF_REPO}; log in with `hf auth login`, accept the repository terms, and retry"
            ) from error
        raise


def load_xlens_model(checkpoint: Path | None, device: Literal["cpu", "cuda"]) -> tuple[XLensNet, Path]:
    """Load the released ViT-S architecture with a strictly matching state dict.

    Args:
        checkpoint: Local released state dict; downloaded with the user's Hugging Face login when None.
        device: Device that owns the model.

    Returns:
        The evaluation-mode model and the checkpoint path it was loaded from.
    """
    checkpoint_path: Path = checkpoint or download_xlens_checkpoint()
    config_path: Path = Path(__file__).parents[2] / "third_party" / "xlens" / "xlens_vits.yaml"
    pipeline: XLensInference = XLensInference(checkpoint_path=str(checkpoint_path), device=device, config=str(config_path))
    return pipeline.model, checkpoint_path


def validate_rig_inputs(images: UInt8[ndarray, "s h w 3"], rays: Float32[ndarray, "s h w 3"], cam_types: Int64[ndarray, "s"]) -> None:
    """Reject rigs X-Lens cannot process.

    Raises:
        ValueError: If fewer than two views are supplied, the resolution is
            smaller than 28 pixels or not divisible by 14, or the arrays disagree.
    """
    views: int = images.shape[0]
    height: int = images.shape[1]
    width: int = images.shape[2]
    if views < 2:
        raise ValueError(f"X-Lens needs at least two views, got {views}")
    if height < 28 or width < 28 or height % 14 != 0 or width % 14 != 0:
        raise ValueError(f"X-Lens height and width must be multiples of 14 and at least 28, got {height}x{width}")
    if rays.shape != (views, height, width, 3) or cam_types.shape != (views,):
        raise ValueError("X-Lens images, rays, and camera types must have matching view and image dimensions")


def normalize_images(images: UInt8[ndarray, "s h w 3"], device: torch.device) -> Float32[Tensor, "1 s 3 h w"]:
    """ImageNet-normalise uint8 RGB views on the inference device.

    Same operation order as the upstream numpy ``normalize_image`` (divide,
    subtract, divide), so the float32 results are identical.

    Args:
        images: RGB views, ``UInt8[ndarray, "s h w 3"]``.
        device: Device that owns the result.

    Returns:
        Normalised network input, ``Float32[Tensor, "1 s 3 h w"]``.
    """
    rgb: UInt8[Tensor, "s h w 3"] = torch.from_numpy(np.ascontiguousarray(images)).to(device)
    mean: Float32[Tensor, "3"] = torch.as_tensor(IMAGENET_MEAN, device=device)
    std: Float32[Tensor, "3"] = torch.as_tensor(IMAGENET_STD, device=device)
    normalized: Float32[Tensor, "s h w 3"] = (rgb.to(torch.float32) / 255.0 - mean) / std
    return rearrange(normalized, "s h w c -> 1 s c h w").contiguous()  # pyrefly: ignore  # bad-argument-type — einops stub false positive


@dataclass(frozen=True, slots=True)
class RigTensors:
    """Model-ready geometry of one rig on the inference device."""

    d_cam: Float32[Tensor, "1 s 3 h w"]
    """Per-pixel camera-frame unit rays."""
    ray_map: Float32[Tensor, "1 s 6 h w"] | None
    """World-frame rays plus normalised translations; None without poses."""
    cam_types: Int64[Tensor, "1 s"]
    """X-Lens camera type ids."""


def rig_tensors(
    rays: Float32[ndarray, "s h w 3"],
    cam_types: Int64[ndarray, "s"],
    cam_T_ref: Float64[ndarray, "s 4 4"] | None,
    device: torch.device,
) -> RigTensors:
    """Build the geometry tensors of ``assemble_batch`` without touching images.

    Args:
        rays: Camera-frame unit rays, ``Float32[ndarray, "s h w 3"]``.
        cam_types: X-Lens camera ids, ``Int64[ndarray, "s"]``.
        cam_T_ref: Optional camera-to-reference poses, ``Float64[ndarray, "s 4 4"]``; X-Lens canonicalises view 0.
        device: Device that owns the tensors.

    Returns:
        Rays, optional ray map, and camera types on ``device``.
    """
    d_cam: Float32[Tensor, "1 s 3 h w"] = rearrange(torch.from_numpy(np.ascontiguousarray(rays)).to(device), "s h w c -> 1 s c h w").contiguous()  # pyrefly: ignore  # bad-argument-type — einops stub false positive
    ray_map: Float32[Tensor, "1 s 6 h w"] | None = None
    if cam_T_ref is not None:
        c2w: Float32[Tensor, "1 s 4 4"] = canonicalize_c2w(torch.from_numpy(np.asarray(cam_T_ref, np.float32))[None].to(device))
        ray_map = build_ray_map(d_cam, c2w)
    return RigTensors(d_cam=d_cam, ray_map=ray_map, cam_types=build_cam_types(cam_types.tolist(), device=device))


class RigKeyMemo:
    """Content key of the current rig, hashing the ray field only when a new array object arrives.

    Hashing a four-view 896x504 ray field costs about 15 ms, so the digest is
    memoised on the array object: callers must not mutate ``rays`` in place
    between calls. Camera types and poses are small and hashed every call.
    """

    def __init__(self) -> None:
        """Start with no rig seen."""
        self._rays: ndarray | None = None
        self._digest: str = ""

    def key(self, rays: Float32[ndarray, "s h w 3"], cam_types: Int64[ndarray, "s"], cam_T_ref: Float64[ndarray, "s 4 4"] | None) -> RigKey:
        """Return the content key of one rig."""
        if rays is not self._rays:
            digest = hashlib.sha256(str(rays.shape).encode())
            digest.update(np.ascontiguousarray(rays))
            self._digest = digest.hexdigest()
            self._rays = rays
        poses: bytes | None = None if cam_T_ref is None else np.ascontiguousarray(cam_T_ref, dtype=np.float64).tobytes()
        return (self._digest, np.ascontiguousarray(cam_types, dtype=np.int64).tobytes(), poses)


def rig_depth_prediction(output: XLensNetOutput) -> RigDepthPrediction:
    """Take view-0 batch outputs as an owning float32 prediction."""
    return RigDepthPrediction(
        depth_m=output["depth_metric"][0].float().clone(),
        confidence=output["depth_conf"][0].float().clone(),
        mask=output["mask"][0].float().clone(),
        scale=float(output["metric_scaling_factor"][0]),
    )


@dataclass
class XLensConfig(BaseRigDepthPredictorConfig):
    """Configuration for the released X-Lens ViT-S model."""

    checkpoint: Path | None = None
    """Local safetensors state dict, or None to download the pinned gated release."""
    amp: AmpDtype = "bf16"
    """CUDA autocast dtype; BF16 is the upstream default and fp32 disables autocast."""
    freeze_geometry: bool = True
    """Compute the rig's attention geometry once per distinct rays/camera types/poses and reuse it; False rebuilds it every call."""

    def setup(self, device: Literal["cpu", "cuda"]) -> "XLensPredictor":
        """Build the configured X-Lens predictor on one device."""
        return XLensPredictor(device=device, checkpoint=self.checkpoint, amp=self.amp, freeze_geometry=self.freeze_geometry)


class XLensPredictor(BaseRigDepthPredictor):
    """Released X-Lens model with upstream preprocessing and output handling."""

    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        checkpoint: Path | None = None,
        amp: AmpDtype = "bf16",
        freeze_geometry: bool = True,
    ) -> None:
        """Load X-Lens and its frozen ViT-S architecture.

        Args:
            device: Where model inference runs.
            checkpoint: Local released state dict; downloaded with the user's
                Hugging Face login when None.
            amp: CUDA autocast dtype, or ``fp32`` for no autocast.
            freeze_geometry: Cache the rig's pixel-independent attention inputs
                across calls; rays must not be mutated in place between calls.
        """
        loaded: tuple[XLensNet, Path] = load_xlens_model(checkpoint, device)
        self.model: XLensNet = loaded[0]
        self.device: torch.device = torch.device(device)
        self.amp_dtype: torch.dtype | None = AMP_DTYPES[amp]
        self.freeze_geometry: bool = freeze_geometry
        self._memo: RigKeyMemo = RigKeyMemo()
        self._frozen_key: RigKey | None = None
        self._frozen: FrozenRigGeometry | None = None

    def frozen_geometry(
        self,
        rays: Float32[ndarray, "s h w 3"],
        cam_types: Int64[ndarray, "s"],
        cam_T_ref: Float64[ndarray, "s 4 4"] | None,
    ) -> FrozenRigGeometry:
        """Return this rig's frozen geometry, recomputing it when rays, camera types, or poses change.

        Under autocast the attention masks are stored in the autocast dtype:
        scaled-dot-product attention casts them there anyway, so the values
        the blocks see are unchanged and the largest tensor halves in size.
        """
        key: RigKey = self._memo.key(rays, cam_types, cam_T_ref)
        if self._frozen is None or key != self._frozen_key:
            tensors: RigTensors = rig_tensors(rays, cam_types, cam_T_ref, self.device)
            with torch.inference_mode():
                frozen: FrozenRigGeometry = self.model.freeze_geometry(tensors.d_cam, tensors.cam_types, tensors.ray_map)
            if self.amp_dtype is not None and self.device.type == "cuda":
                frozen = frozen.with_mask_dtype(self.amp_dtype)
            self._frozen = frozen
            self._frozen_key = key
        return self._frozen

    def __call__(
        self,
        images: UInt8[ndarray, "s h w 3"],
        rays: Float32[ndarray, "s h w 3"],
        cam_types: Int64[ndarray, "s"],
        cam_T_ref: Float64[ndarray, "s 4 4"] | None,
    ) -> RigDepthPrediction:
        """Predict per-view camera-frame metric z-depth.

        Args:
            images: Shared-resolution RGB views, ``UInt8[ndarray, "s h w 3"]``.
            rays: Camera-frame unit rays, ``Float32[ndarray, "s h w 3"]``.
            cam_types: X-Lens camera ids, ``Int64[ndarray, "s"]``.
            cam_T_ref: Optional camera-to-reference poses,
                ``Float64[ndarray, "s 4 4"]``. X-Lens canonicalizes view 0.

        Returns:
            Metric depth, confidence, mask, and scale at input resolution on the model device.

        Raises:
            ValueError: If fewer than two views are supplied or the resolution
                is smaller than 28 pixels or not divisible by 14.
        """
        validate_rig_inputs(images, rays, cam_types)
        image_tensor: Float32[Tensor, "1 s 3 h w"] = normalize_images(images, self.device)
        # Geometry is built outside autocast, like upstream's ``assemble_batch``: the
        # pose canonicalisation and ray-map matmuls stay float32. The frozen path
        # therefore also evaluates the ray encoder and distortion-bias MLP in
        # float32 once per rig, whereas the unfrozen path rebuilds them under
        # autocast at every call exactly as the released pipeline does.
        frozen: FrozenRigGeometry | None = self.frozen_geometry(rays, cam_types, cam_T_ref) if self.freeze_geometry else None
        tensors: RigTensors | None = None if self.freeze_geometry else rig_tensors(rays, cam_types, cam_T_ref, self.device)
        autocast_enabled: bool = self.amp_dtype is not None and self.device.type == "cuda"
        with torch.inference_mode(), torch.autocast("cuda", enabled=autocast_enabled, dtype=self.amp_dtype or torch.bfloat16):
            output: XLensNetOutput
            if tensors is None:
                output = self.model(image_tensor, frozen=frozen)
            else:
                output = self.model(image_tensor, ray_map=tensors.ray_map, d_cam=tensors.d_cam, cam_types=tensors.cam_types)
        return rig_depth_prediction(output)
