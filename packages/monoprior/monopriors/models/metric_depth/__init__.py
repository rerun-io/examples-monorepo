from collections.abc import Callable
from typing import Literal, get_args

from .base_metric_depth import BaseMetricPredictor, MetricDepthPrediction
from .metric3d import Metric3DPredictor
from .moge_v2 import MoGeV2MetricPredictor
from .unidepth import UniDepthMetricPredictor

# Define predictor names as a list of strings
METRIC_PREDICTORS = Literal["UniDepthMetricPredictor", "Metric3DPredictor", "MoGeV2MetricPredictor"]

# Use the list to generate the __all__ list
__all__: list[str] = list(get_args(METRIC_PREDICTORS)) + [
    "MetricDepthPrediction",
]


def get_metric_predictor(
    predictor_type: METRIC_PREDICTORS,
) -> Callable[..., BaseMetricPredictor]:
    match predictor_type:
        case "UniDepthMetricPredictor":
            return UniDepthMetricPredictor
        case "Metric3DPredictor":
            return Metric3DPredictor
        case "MoGeV2MetricPredictor":
            return MoGeV2MetricPredictor
        case _:
            raise ValueError(f"Unknown predictor type: {predictor_type}")
