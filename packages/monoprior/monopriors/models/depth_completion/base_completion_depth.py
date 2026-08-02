from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

import numpy as np
from jaxtyping import Float, UInt8, UInt16

from monopriors.models._protocols import DeviceMovable

ModelT = TypeVar("ModelT", bound=DeviceMovable)


@dataclass
class CompletionDepthPrediction:
    """
    Dataclass for storing depth completion predictions
    """

    depth_mm: UInt16[np.ndarray, "h w"]
    # metric depth in meters
    confidence: Float[np.ndarray, "h w"]
    # confidence values


class BaseCompletionPredictor(ABC, Generic[ModelT]):
    model: ModelT

    @abstractmethod
    def __call__(self, rgb: UInt8[np.ndarray, "h w 3"]) -> CompletionDepthPrediction:
        raise NotImplementedError

    def set_model_device(self, device: Literal["cpu", "cuda"] = "cuda") -> None:
        self.model.to(device)
