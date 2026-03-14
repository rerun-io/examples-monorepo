from collections.abc import Callable
from typing import Literal, get_args

from .base_relative_depth import BaseRelativePredictor, RelativeDepthPrediction
from .depth_anything_v1 import DepthAnythingV1Predictor
from .depth_anything_v2 import DepthAnythingV2Predictor
from .moge import MogeV1Predictor
from .unidepth import UniDepthRelativePredictor

# Define predictor names as a list of strings
RELATIVE_PREDICTORS = Literal[
    "DepthAnythingV1Predictor",
    "DepthAnythingV2Predictor",
    "UniDepthRelativePredictor",
    "MogeV1Predictor",
]

# Use the list to generate the __all__ list
__all__: list[str] = list(get_args(RELATIVE_PREDICTORS)) + [
    "RelativeDepthPrediction",
]


def get_relative_predictor(
    predictor_type: RELATIVE_PREDICTORS,
) -> Callable[..., BaseRelativePredictor]:
    match predictor_type:
        case "UniDepthRelativePredictor":
            return UniDepthRelativePredictor
        case "DepthAnythingV2Predictor":
            return DepthAnythingV2Predictor
        case "DepthAnythingV1Predictor":
            return DepthAnythingV1Predictor
        case "MogeV1Predictor":
            return MogeV1Predictor
        case _:
            raise ValueError(f"Unknown predictor type: {predictor_type}")
