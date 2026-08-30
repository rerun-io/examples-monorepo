from typing import TYPE_CHECKING

import tyro

from monopriors.models.metric_depth.base_metric_depth import BaseMetricPredictor, BaseMetricPredictorConfig, MetricDepthPrediction
from monopriors.models.metric_depth.moge_v2 import MoGeV2MetricConfig, MoGeV2MetricPredictor
from monopriors.models.metric_depth.unidepth import UniDepthMetricConfig, UniDepthMetricPredictor

metric_predictor_defaults: dict[str, BaseMetricPredictorConfig] = {
    "unidepth-metric": UniDepthMetricConfig(),
    "moge-v2-metric": MoGeV2MetricConfig(),
}

if TYPE_CHECKING:
    MetricPredictorUnion = BaseMetricPredictorConfig
else:
    MetricPredictorUnion = tyro.extras.subcommand_type_from_defaults(metric_predictor_defaults, prefix_names=False)

AnnotatedMetricPredictorUnion = tyro.conf.OmitSubcommandPrefixes[MetricPredictorUnion]

__all__: list[str] = [
    "AnnotatedMetricPredictorUnion",
    "BaseMetricPredictor",
    "BaseMetricPredictorConfig",
    "MetricDepthPrediction",
    "MetricPredictorUnion",
    "MoGeV2MetricConfig",
    "MoGeV2MetricPredictor",
    "UniDepthMetricConfig",
    "UniDepthMetricPredictor",
    "metric_predictor_defaults",
]
