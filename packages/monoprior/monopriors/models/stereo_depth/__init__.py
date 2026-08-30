from collections.abc import Callable
from typing import Literal, get_args

from .base_stereo_depth import BaseStereoPredictor, StereoDepthPrediction, disparity_to_metric_depth
from .fast_foundationstereo import FastFoundationStereoPredictor
from .liteanystereo import LiteAnyStereoPredictor

STEREO_PREDICTORS = Literal["LiteAnyStereoPredictor", "FastFoundationStereoPredictor"]

__all__: list[str] = list(get_args(STEREO_PREDICTORS)) + [
    "BaseStereoPredictor",
    "StereoDepthPrediction",
    "disparity_to_metric_depth",
]


def get_stereo_predictor(predictor_type: STEREO_PREDICTORS) -> Callable[..., BaseStereoPredictor]:
    match predictor_type:
        case "LiteAnyStereoPredictor":
            return LiteAnyStereoPredictor
        case "FastFoundationStereoPredictor":
            return FastFoundationStereoPredictor
        case _:
            raise ValueError(f"Unknown predictor type: {predictor_type}")
