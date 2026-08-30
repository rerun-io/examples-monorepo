from typing import TYPE_CHECKING

import tyro

from monopriors.models.surface_normal.base_normal_model import BaseNormalPredictor, BaseNormalPredictorConfig, SurfaceNormalPrediction
from monopriors.models.surface_normal.dsine_model import DSineNormalConfig, DSineNormalPredictor
from monopriors.models.surface_normal.moge_v2 import MoGeV2NormalConfig, MoGeV2NormalPredictor
from monopriors.models.surface_normal.omni_normal_model import OmniNormalConfig, OmniNormalPredictor

normal_predictor_defaults: dict[str, BaseNormalPredictorConfig] = {
    "dsine-normal": DSineNormalConfig(),
    "moge-v2-normal": MoGeV2NormalConfig(),
    "omni-normal": OmniNormalConfig(),
}

if TYPE_CHECKING:
    NormalPredictorUnion = BaseNormalPredictorConfig
else:
    NormalPredictorUnion = tyro.extras.subcommand_type_from_defaults(normal_predictor_defaults, prefix_names=False)

AnnotatedNormalPredictorUnion = tyro.conf.OmitSubcommandPrefixes[NormalPredictorUnion]

__all__: list[str] = [
    "AnnotatedNormalPredictorUnion",
    "BaseNormalPredictor",
    "BaseNormalPredictorConfig",
    "DSineNormalConfig",
    "DSineNormalPredictor",
    "MoGeV2NormalConfig",
    "MoGeV2NormalPredictor",
    "NormalPredictorUnion",
    "OmniNormalConfig",
    "OmniNormalPredictor",
    "SurfaceNormalPrediction",
    "normal_predictor_defaults",
]
