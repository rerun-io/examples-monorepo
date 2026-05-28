from dataclasses import dataclass
from pathlib import Path

from simplecv.configs.base_config import InstantiateConfig


@dataclass
class BaseExoEgoDatasetConfig(InstantiateConfig):
    load_labels: bool = True
    parent_log_path: Path = Path("world")
    verbose: bool = False
