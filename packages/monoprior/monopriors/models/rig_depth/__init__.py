"""Registry and public contract for calibrated multi-camera rig depth."""

from typing import TYPE_CHECKING

import tyro

from monopriors.models.rig_depth.base_rig_depth import BaseRigDepthPredictor, BaseRigDepthPredictorConfig, RigDepthPrediction, unproject
from monopriors.models.rig_depth.rays import camera_type, unit_rays
from monopriors.models.rig_depth.xlens import XLensConfig, XLensPredictor

rig_depth_predictor_defaults: dict[str, BaseRigDepthPredictorConfig] = {"xlens": XLensConfig()}

if TYPE_CHECKING:
    RigDepthPredictorUnion = BaseRigDepthPredictorConfig
else:
    # tyro requires at least two defaults to construct a subcommand union. Keep
    # the same public registry shape while X-Lens is the family's sole model.
    RigDepthPredictorUnion = XLensConfig

AnnotatedRigDepthPredictorUnion = tyro.conf.OmitSubcommandPrefixes[RigDepthPredictorUnion]

__all__: list[str] = [
    "AnnotatedRigDepthPredictorUnion",
    "BaseRigDepthPredictor",
    "BaseRigDepthPredictorConfig",
    "RigDepthPrediction",
    "RigDepthPredictorUnion",
    "XLensConfig",
    "XLensPredictor",
    "camera_type",
    "rig_depth_predictor_defaults",
    "unit_rays",
    "unproject",
]
