from abc import ABC, abstractmethod
from dataclasses import dataclass
from timeit import default_timer as timer
from typing import Literal

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float, UInt8

from monopriors.models.metric_depth import BaseMetricPredictor, MetricDepthPrediction, get_metric_predictor
from monopriors.models.metric_depth.moge_v2 import MOGE_V2_CHECKPOINTS
from monopriors.models.surface_normal import (
    BaseNormalPredictor,
    SurfaceNormalPrediction,
    get_normal_predictor,
)
from monopriors.third_party.moge.model._inference import InferenceOutput
from monopriors.third_party.moge.model.v2 import MoGeModel


@dataclass
class OldMonoPriorPrediction:
    """Legacy batched tensor representation of monocular priors."""

    depth_b1hw: Float[torch.Tensor, "b 1 h w"]
    """Predicted metric depth in BCHW layout."""
    normal_b3hw: Float[torch.Tensor, "b 3 h w"]
    """Predicted surface normals in BCHW layout."""
    K_b33: Float[torch.Tensor, "b 3 3"] | None = None
    """Optional camera intrinsics for each batch item."""
    depth_conf_b1hw: Float[torch.Tensor, "b 1 h w"] | None = None
    """Optional per-pixel depth confidence."""
    normal_conf_b1hw: Float[torch.Tensor, "b 1 h w"] | None = None
    """Optional per-pixel normal confidence."""

    def to_numpy(
        self,
    ) -> tuple[
        Float[np.ndarray, "b h w 1"],
        Float[np.ndarray, "b h w 3"],
        Float[np.ndarray, "b 3 3"] | None,
        Float[np.ndarray, "b h w 1"] | None,
        Float[np.ndarray, "b h w 1"] | None,
    ]:
        """Convert legacy Torch predictions to channel-last NumPy arrays.

        Returns:
            Float depth, normal, intrinsics, depth-confidence, and normal-confidence arrays with their documented shapes.
        """
        depth_np_bhw1: Float[np.ndarray, "b h w 1"] = rearrange(self.depth_b1hw, "b c h w -> b h w c").numpy(force=True)
        normal_np_bhw3: Float[np.ndarray, "b h w 3"] = rearrange(self.normal_b3hw, "b c h w -> b h w c").numpy(force=True)
        K_np_b33: Float[np.ndarray, "b 3 3"] | None = self.K_b33.numpy(force=True) if self.K_b33 is not None else None
        depth_conf_np_bhw1: Float[np.ndarray, "b h w 1"] | None = (
            rearrange(self.depth_conf_b1hw, "b c h w -> b h w c").numpy(force=True) if self.depth_conf_b1hw is not None else None
        )
        normal_conf_np_bhw1: Float[np.ndarray, "b h w 1"] | None = (
            rearrange(self.normal_conf_b1hw, "b c h w -> b h w c").numpy(force=True) if self.normal_conf_b1hw is not None else None
        )
        return (
            depth_np_bhw1,
            normal_np_bhw3,
            K_np_b33,
            depth_conf_np_bhw1,
            normal_conf_np_bhw1,
        )


@dataclass
class MonoPriorPrediction:
    """Metric-depth and surface-normal predictions for one image."""

    metric_pred: MetricDepthPrediction
    """Metric depth prediction."""
    normal_pred: SurfaceNormalPrediction
    """Surface-normal prediction."""


class MonoPriorModel(ABC):
    """Interface for a paired metric-depth and surface-normal predictor."""

    def __init__(self) -> None:
        self.device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def __call__(self, rgb: UInt8[np.ndarray, "h w 3"], K_33: Float[np.ndarray, "3 3"] | None) -> MonoPriorPrediction:
        """Predict metric depth and surface normals for a uint8 RGB image shaped ``h w 3``."""
        raise NotImplementedError


class DsineAndUnidepth(MonoPriorModel):
    """Pair UniDepth metric depth with DSINE surface normals."""

    def __init__(self) -> None:
        super().__init__()
        # Keep this composite aligned with its name by using UniDepth for metric depth.
        self.depth_model: BaseMetricPredictor = get_metric_predictor("UniDepthMetricPredictor")(device=self.device)
        self.surface_model: BaseNormalPredictor = get_normal_predictor("DSineNormalPredictor")(device=self.device)

    def __call__(
        self,
        rgb: UInt8[np.ndarray, "h w 3"],
        K_33: Float[np.ndarray, "3 3"] | None = None,
    ) -> MonoPriorPrediction:
        """Predict metric depth and normals with separate models.

        Args:
            rgb: uint8 RGB image shaped ``h w 3``.
            K_33: Optional float camera intrinsics shaped ``3 3``.

        Returns:
            Combined metric-depth and surface-normal prediction.
        """
        metric_pred: MetricDepthPrediction = self.depth_model.__call__(rgb, K_33)
        normal_pred: SurfaceNormalPrediction = self.surface_model(rgb, K_33)

        return MonoPriorPrediction(metric_pred=metric_pred, normal_pred=normal_pred)


class MoGeV2MonoPrior(MonoPriorModel):
    """Composite model using a single MoGe v2 forward pass for both metric depth and normals.

    Unlike ``DsineAndUnidepth`` which loads two separate models, this runs one
    model that produces both outputs simultaneously.
    """

    def __init__(
        self,
        encoder: Literal["vits", "vitb", "vitl"] = "vitl",
    ) -> None:
        """Load a pinned normal-capable MoGe v2 checkpoint.

        Args:
            encoder: DINOv2 encoder size.
        """
        super().__init__()
        checkpoint: tuple[str, str] = MOGE_V2_CHECKPOINTS[(encoder, True)]
        repo_id: str = checkpoint[0]
        revision: str = checkpoint[1]
        print(f"Loading MoGe v2 mono-prior model ({repo_id})...")
        start: float = timer()
        self.model: MoGeModel = MoGeModel.from_pretrained(repo_id, revision=revision).to(self.device)
        print(f"MoGe v2 mono-prior model loaded. Time: {timer() - start:.2f}s")

    def __call__(
        self,
        rgb: UInt8[np.ndarray, "h w 3"],
        K_33: Float[np.ndarray, "3 3"] | None = None,
    ) -> MonoPriorPrediction:
        """Predict metric depth and normals with one MoGe v2 forward pass.

        Args:
            rgb: uint8 RGB image shaped ``h w 3``.
            K_33: Optional float camera intrinsics shaped ``3 3``; MoGe v2 estimates its own intrinsics.

        Returns:
            Combined metric-depth and surface-normal prediction.
        """
        h: int
        w: int
        h, w, _ = rgb.shape
        input_image: Float[torch.Tensor, "3 h w"] = rearrange(torch.from_numpy(rgb), "h w c -> c h w").to(
            device=self.device,
            dtype=torch.float32,
        )
        input_image.div_(255.0)

        output: InferenceOutput = self.model.infer(input_image, force_projection=False)

        # Build intrinsics from normalized values
        normalized_k: Float[np.ndarray, "3 3"] = output["intrinsics"].numpy(force=True)
        fx: float = float(normalized_k[0, 0] * w)
        fy: float = float(normalized_k[1, 1] * h)
        cx: float = float(normalized_k[0, 2] * w)
        cy: float = float(normalized_k[1, 2] * h)
        intrinsics: Float[np.ndarray, "3 3"] = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        depth_meters: Float[np.ndarray, "h w"] = output["depth"].numpy(force=True)
        confidence: Float[np.ndarray, "h w"] = output["mask"].numpy(force=True).astype(np.float32)
        normal_hw3: Float[np.ndarray, "h w 3"] = output["normal"].numpy(force=True)
        confidence_hw1: Float[np.ndarray, "h w 1"] = confidence[:, :, np.newaxis]

        metric_pred: MetricDepthPrediction = MetricDepthPrediction(
            depth_meters=depth_meters,
            confidence=confidence,
            K_33=intrinsics,
        )
        normal_pred: SurfaceNormalPrediction = SurfaceNormalPrediction(
            normal_hw3=normal_hw3,
            confidence_hw1=confidence_hw1,
        )

        return MonoPriorPrediction(metric_pred=metric_pred, normal_pred=normal_pred)
