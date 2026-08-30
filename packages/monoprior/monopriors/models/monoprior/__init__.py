from typing import TYPE_CHECKING

import tyro

from monopriors.models.monoprior.monoprior_models import (
    DsineAndUnidepth,
    DsineAndUnidepthConfig,
    MoGeV2MonoPrior,
    MoGeV2MonoPriorConfig,
    MonoPriorModel,
    MonoPriorModelConfig,
    MonoPriorPrediction,
)

monoprior_model_defaults: dict[str, MonoPriorModelConfig] = {
    "moge-v2-monoprior": MoGeV2MonoPriorConfig(),
    "dsine-and-unidepth": DsineAndUnidepthConfig(),
}

if TYPE_CHECKING:
    MonoPriorModelUnion = MonoPriorModelConfig
else:
    MonoPriorModelUnion = tyro.extras.subcommand_type_from_defaults(monoprior_model_defaults, prefix_names=False)

AnnotatedMonoPriorModelUnion = tyro.conf.OmitSubcommandPrefixes[MonoPriorModelUnion]

__all__: list[str] = [
    "AnnotatedMonoPriorModelUnion",
    "DsineAndUnidepth",
    "DsineAndUnidepthConfig",
    "MoGeV2MonoPrior",
    "MoGeV2MonoPriorConfig",
    "MonoPriorModel",
    "MonoPriorModelConfig",
    "MonoPriorModelUnion",
    "MonoPriorPrediction",
    "monoprior_model_defaults",
]
