from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

import numpy as np
from jaxtyping import Float32, UInt8

from monopriors.models._protocols import DeviceMovable

ModelT = TypeVar("ModelT", bound=DeviceMovable)


@dataclass
class RelativeDepthPrediction:
    disparity: Float32[np.ndarray, "h w"]
    # relative disparity
    depth: Float32[np.ndarray, "h w"]
    # relative depth
    confidence: Float32[np.ndarray, "h w"]
    # confidence values
    K_33: Float32[np.ndarray, "3 3"]
    # intrinsics


class BaseRelativePredictor(ABC, Generic[ModelT]):
    model: ModelT

    @abstractmethod
    def __call__(
        self, rgb: UInt8[np.ndarray, "h w 3"], K_33: Float32[np.ndarray, "3 3"] | None
    ) -> RelativeDepthPrediction:
        raise NotImplementedError

    def set_model_device(self, device: Literal["cpu", "cuda"] = "cuda") -> None:
        self.model.to(device)


class BaseVideoRelativePredictor(ABC, Generic[ModelT]):
    model: ModelT

    @abstractmethod
    def __call__(
        self, rgb_frames: UInt8[np.ndarray, "T H W 3"], K_33: Float32[np.ndarray, "3 3"] | None
    ) -> list[RelativeDepthPrediction]:
        raise NotImplementedError

    def set_model_device(self, device: Literal["cpu", "cuda"] = "cuda") -> None:
        self.model.to(device)
