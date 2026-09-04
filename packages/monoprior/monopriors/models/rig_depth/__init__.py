"""Registry and public contract for calibrated multi-camera rig depth."""

from typing import TYPE_CHECKING

import tyro

from monopriors.models.rig_depth.base_rig_depth import BaseRigDepthPredictor, BaseRigDepthPredictorConfig, RigDepthPrediction, unproject
from monopriors.models.rig_depth.rays import camera_type, unit_rays
from monopriors.models.rig_depth.xlens import XLensConfig, XLensPredictor
from monopriors.models.rig_depth.xlens_trt import XLensTrtConfig, XLensTrtPredictor

rig_depth_predictor_defaults: dict[str, BaseRigDepthPredictorConfig] = {"xlens": XLensConfig(), "xlens-trt": XLensTrtConfig()}

if TYPE_CHECKING:
    RigDepthPredictorUnion = BaseRigDepthPredictorConfig
else:
    RigDepthPredictorUnion = tyro.extras.subcommand_type_from_defaults(rig_depth_predictor_defaults, prefix_names=False)

AnnotatedRigDepthPredictorUnion = tyro.conf.OmitSubcommandPrefixes[RigDepthPredictorUnion]

__all__: list[str] = [
    "AnnotatedRigDepthPredictorUnion",
    "BaseRigDepthPredictor",
    "BaseRigDepthPredictorConfig",
    "RigDepthPrediction",
    "RigDepthPredictorUnion",
    "XLensConfig",
    "XLensPredictor",
    "XLensTrtConfig",
    "XLensTrtPredictor",
    "camera_type",
    "rig_depth_predictor_defaults",
    "unit_rays",
    "unproject",
]
