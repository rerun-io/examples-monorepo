from __future__ import annotations

from typing import TYPE_CHECKING

import tyro

from simplecv.data.exoego.assembly101 import Assembly101Config
from simplecv.data.exoego.base_exoego import BaseExoEgoDatasetConfig
from simplecv.data.exoego.hocap import HocapConfig
from simplecv.data.exoego.robocap import RobocapConfig

# ───────────────────── registry → union ─────────────────── #
dataset_defaults = {
    "assembly101": Assembly101Config(),
    "hocap": HocapConfig(),
    "robocap": RobocapConfig(),
}


if TYPE_CHECKING:  # for IDEs / mypy
    EgoDatasetUnion = BaseExoEgoDatasetConfig
else:
    EgoDatasetUnion = tyro.extras.subcommand_type_from_defaults(dataset_defaults, prefix_names=False)

AnnotatedExoEgoDatasetUnion = tyro.conf.OmitSubcommandPrefixes[EgoDatasetUnion]
