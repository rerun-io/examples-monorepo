from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

import numpy as np
from jaxtyping import Float, UInt8

from monopriors.models._protocols import DeviceMovable

ModelT = TypeVar("ModelT", bound=DeviceMovable)


@dataclass
class SurfaceNormalPrediction:
    normal_hw3: Float[np.ndarray, "h w 3"]
    # surface normal prediction TODO make into a consistent coordinate system
    confidence_hw1: Float[np.ndarray, "h w 1"]
    # confidence values


class BaseNormalPredictor(ABC, Generic[ModelT]):
    model: ModelT

    @abstractmethod
    def __call__(
        self, rgb: UInt8[np.ndarray, "h w 3"], K_33: Float[np.ndarray, "3 3"] | None
    ) -> SurfaceNormalPrediction:
        raise NotImplementedError

    def set_model_device(self, device: Literal["cpu", "cuda"] = "cuda") -> None:
        self.model.to(device)
