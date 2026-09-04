from typing import TYPE_CHECKING

import tyro

from monopriors.models.stereo_depth.base_stereo_depth import (
    BaseStereoPredictor,
    BaseStereoPredictorConfig,
    StereoDepthPrediction,
    disparity_to_metric_depth,
)
from monopriors.models.stereo_depth.fast_foundationstereo import FastFoundationStereoConfig, FastFoundationStereoPredictor
from monopriors.models.stereo_depth.liteanystereo import LiteAnyStereoConfig, LiteAnyStereoPredictor

stereo_predictor_defaults: dict[str, BaseStereoPredictorConfig] = {
    "liteanystereo": LiteAnyStereoConfig(),
    "fast-foundationstereo": FastFoundationStereoConfig(),
}

if TYPE_CHECKING:
    StereoPredictorUnion = BaseStereoPredictorConfig
else:
    StereoPredictorUnion = tyro.extras.subcommand_type_from_defaults(stereo_predictor_defaults, prefix_names=False)

AnnotatedStereoPredictorUnion = tyro.conf.OmitSubcommandPrefixes[StereoPredictorUnion]

__all__: list[str] = [
    "AnnotatedStereoPredictorUnion",
    "BaseStereoPredictor",
    "BaseStereoPredictorConfig",
    "FastFoundationStereoConfig",
    "FastFoundationStereoPredictor",
    "LiteAnyStereoConfig",
    "LiteAnyStereoPredictor",
    "StereoPredictorUnion",
    "StereoDepthPrediction",
    "disparity_to_metric_depth",
    "stereo_predictor_defaults",
]
