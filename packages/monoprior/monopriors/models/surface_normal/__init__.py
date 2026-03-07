from collections.abc import Callable
from typing import Literal, get_args

from .base_normal_model import BaseNormalPredictor, SurfaceNormalPrediction
from .dsine_model import DSineNormalPredictor
from .moge_v2 import MoGeV2NormalPredictor
from .omni_normal_model import OmniNormalPredictor

# Define predictor names as a list of strings
NORMAL_PREDICTORS = Literal["DSineNormalPredictor", "OmniNormalPredictor", "MoGeV2NormalPredictor"]

# Use the list to generate the __all__ list
__all__: list[str] = list(get_args(NORMAL_PREDICTORS)) + [
    "SurfaceNormalPrediction",
]


def get_normal_predictor(
    predictor_type: NORMAL_PREDICTORS,
) -> Callable[..., BaseNormalPredictor]:
    match predictor_type:
        case "DSineNormalPredictor":
            return DSineNormalPredictor
        case "OmniNormalPredictor":
            return OmniNormalPredictor
        case "MoGeV2NormalPredictor":
            return MoGeV2NormalPredictor
        case _:
            raise ValueError(f"Unknown predictor type: {predictor_type}")
