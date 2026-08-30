from typing import TYPE_CHECKING

import tyro

from monopriors.models.relative_depth.base_relative_depth import (
    BaseRelativePredictor,
    BaseRelativePredictorConfig,
    BaseVideoRelativePredictor,
    BaseVideoRelativePredictorConfig,
    RelativeDepthPrediction,
)
from monopriors.models.relative_depth.depth_anything_v1 import DepthAnythingV1Config, DepthAnythingV1Predictor
from monopriors.models.relative_depth.depth_anything_v2 import DepthAnythingV2Config, DepthAnythingV2Predictor
from monopriors.models.relative_depth.moge_v1 import MoGeV1Config, MoGeV1Predictor
from monopriors.models.relative_depth.unidepth import UniDepthRelativeConfig, UniDepthRelativePredictor
from monopriors.models.relative_depth.video_depth_anything import VideoDepthAnythingConfig, VideoDepthAnythingPredictor
from monopriors.models.relative_depth.zipdepth import ZipDepthConfig, ZipDepthPredictor

relative_predictor_defaults: dict[str, BaseRelativePredictorConfig] = {
    "moge-v1": MoGeV1Config(),
    "unidepth-relative": UniDepthRelativeConfig(),
    "depth-anything-v1": DepthAnythingV1Config(),
    "depth-anything-v2": DepthAnythingV2Config(),
    "zipdepth": ZipDepthConfig(),
}

if TYPE_CHECKING:
    RelativePredictorUnion = BaseRelativePredictorConfig
else:
    RelativePredictorUnion = tyro.extras.subcommand_type_from_defaults(relative_predictor_defaults, prefix_names=False)

AnnotatedRelativePredictorUnion = tyro.conf.OmitSubcommandPrefixes[RelativePredictorUnion]

__all__: list[str] = [
    "AnnotatedRelativePredictorUnion",
    "BaseRelativePredictor",
    "BaseRelativePredictorConfig",
    "BaseVideoRelativePredictor",
    "BaseVideoRelativePredictorConfig",
    "DepthAnythingV1Config",
    "DepthAnythingV1Predictor",
    "DepthAnythingV2Config",
    "DepthAnythingV2Predictor",
    "MoGeV1Config",
    "MoGeV1Predictor",
    "RelativeDepthPrediction",
    "RelativePredictorUnion",
    "UniDepthRelativeConfig",
    "UniDepthRelativePredictor",
    "VideoDepthAnythingConfig",
    "VideoDepthAnythingPredictor",
    "ZipDepthConfig",
    "ZipDepthPredictor",
    "relative_predictor_defaults",
]
