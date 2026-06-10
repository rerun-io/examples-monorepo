from dataclasses import dataclass
from timeit import default_timer as timer

import numpy as np
from simplecv.configs.exoego_dataset_configs import AnnotatedExoEgoDatasetUnion

np.set_printoptions(suppress=True)


@dataclass
class BatchConvertConfig:
    """Configuration for batch conversion from raw exo/ego data to Rerun artifacts."""

    # rrd_save_dir: Path
    dataset: AnnotatedExoEgoDatasetUnion
    """Dataset configuration used to construct the exo/ego sequence iterator."""


def batch_raw_to_rrd(config: BatchConvertConfig) -> None:
    start_time: float = timer()
    _ = config

    print("No ingest work configured; use egoexo_forge.api.batch_raw_to_rrd for RRD conversion.")
    print(f"Total time taken: {timer() - start_time:.2f} seconds")
