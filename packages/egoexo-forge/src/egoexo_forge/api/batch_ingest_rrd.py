from dataclasses import dataclass
from timeit import default_timer as timer

import numpy as np
from simplecv.configs.exoego_dataset_configs import AnnotatedExoEgoDatasetUnion
from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence
from tqdm import tqdm

np.set_printoptions(suppress=True)


@dataclass
class BatchConvertConfig:
    # rrd_save_dir: Path
    dataset: AnnotatedExoEgoDatasetUnion


def batch_raw_to_rrd(config: BatchConvertConfig):
    start_time: float = timer()
    exoego_sequence: BaseExoEgoSequence = config.dataset.setup()

    for idx, current_exoego_sequence in enumerate(
        tqdm(exoego_sequence.iter_dataset(), desc="Processing sequences", leave=False)
    ):
        # If num_sequences_to_convert is set, only process that many sequences
        ego_sequence: BaseEgoSequence | None = current_exoego_sequence.ego_sequence
        exo_sequence: BaseExoSequence | None = current_exoego_sequence.exo_sequence

    print(f"Total time taken: {timer() - start_time:.2f} seconds")
