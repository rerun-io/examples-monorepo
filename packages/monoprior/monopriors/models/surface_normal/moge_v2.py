"""Single-image surface-normal adapter for the unified MoGe v2 core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from jaxtyping import Float, UInt8

from monopriors.models.moge_v2.export import MOGE_V2_CHECKPOINTS, Encoder
from monopriors.models.moge_v2.predictor import MoGeV2NormalOutput, MoGeV2TorchPredictor
from monopriors.models.surface_normal.base_normal_model import BaseNormalPredictor, BaseNormalPredictorConfig, SurfaceNormalPrediction
from monopriors.third_party.moge.model.v2 import MoGeModel

MOGE_V2_NORMAL_CHECKPOINTS: dict[Encoder, tuple[str, str]] = MOGE_V2_CHECKPOINTS
"""Backward-compatible name for the unified normal-capable checkpoint table."""


@dataclass
class MoGeV2NormalConfig(BaseNormalPredictorConfig):
    """Configuration for MoGe v2 surface-normal prediction."""

    encoder: Encoder = "vitl"
    """DINOv2 encoder size."""

    def setup(self, device: Literal["cpu", "cuda"]) -> MoGeV2NormalPredictor:
        """Build the configured MoGe v2 normal predictor on one device."""
        return MoGeV2NormalPredictor(device=device, encoder=self.encoder)


class MoGeV2NormalPredictor(BaseNormalPredictor[MoGeModel]):
    """MoGe v2 surface-normal adapter using the normal-only heads."""

    def __init__(self, device: Literal["cpu", "cuda"], encoder: Encoder = "vitl") -> None:
        """Load the unified normal-only core.

        Args:
            device: Must be ``"cuda"``; MoGe v2 inference is GPU-only.
            encoder: DINOv2 encoder size.

        Raises:
            ValueError: If CPU is requested.
        """
        super().__init__()
        if device == "cpu":
            raise ValueError("MoGe v2 predictors require CUDA; device='cpu' is unsupported.")
        self.predictor: MoGeV2TorchPredictor = MoGeV2TorchPredictor(encoder=encoder, heads="normal")
        self.model: MoGeModel = self.predictor.model
        self.device: Literal["cpu", "cuda"] = device

    def __call__(
        self,
        rgb: UInt8[np.ndarray, "h w 3"],
        K_33: Float[np.ndarray, "3 3"] | None = None,
    ) -> SurfaceNormalPrediction:
        """Predict surface normals and binary confidence.

        Args:
            rgb: uint8 NumPy RGB image shaped ``h w 3``.
            K_33: Ignored camera intrinsics; the normal head does not use them.

        Returns:
            Float32 normals zeroed outside the binary-valid mask and confidence.
        """
        from monopriors.models.moge_v2.adapters import normal_to_surface_prediction, rgb_to_cuda_batch

        output: MoGeV2NormalOutput = self.predictor.predict_normals(rgb_to_cuda_batch(rgb))
        return normal_to_surface_prediction(output)
