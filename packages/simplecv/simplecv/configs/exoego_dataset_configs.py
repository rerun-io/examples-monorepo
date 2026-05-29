from __future__ import annotations

from typing import TYPE_CHECKING

import tyro

from simplecv.data.exoego.aria_gen2_pilot import AriaGen2PilotConfig
from simplecv.data.exoego.assembly101 import Assembly101Config
from simplecv.data.exoego.base_exoego import BaseExoEgoDatasetConfig
from simplecv.data.exoego.ego100k import Egocentric100KConfig
from simplecv.data.exoego.ego_dex import EgoDexConfig
from simplecv.data.exoego.epfl_smart_kitchen import EpflSmartKitchenConfig
from simplecv.data.exoego.hocap import HocapConfig
from simplecv.data.exoego.hot3d import Hot3dConfig
from simplecv.data.exoego.robocap import RobocapConfig
from simplecv.data.exoego.rrd_exoego import RRDExoEgoConfig
from simplecv.data.exoego.umetrack import UmeTrackConfig

# ───────────────────── registry → union ─────────────────── #
dataset_defaults = {
    "aria-gen2": AriaGen2PilotConfig(),
    "assembly101": Assembly101Config(),
    "ego100k": Egocentric100KConfig(),
    "epfl-smart-kitchen": EpflSmartKitchenConfig(),
    "hocap": HocapConfig(),
    "ego-dex": EgoDexConfig(),
    "hot3d": Hot3dConfig(),
    "robocap": RobocapConfig(),
    "rrd": RRDExoEgoConfig(),
    "umetrack": UmeTrackConfig(),
}


if TYPE_CHECKING:  # for IDEs / mypy
    EgoDatasetUnion = BaseExoEgoDatasetConfig
else:
    EgoDatasetUnion = tyro.extras.subcommand_type_from_defaults(dataset_defaults, prefix_names=False)

AnnotatedExoEgoDatasetUnion = tyro.conf.OmitSubcommandPrefixes[EgoDatasetUnion]
