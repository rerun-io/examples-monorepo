

from torch.utils.data import ConcatDataset

from .base import RGBDDataset

# RGBD-Dataset
from .tartan import TartanAir


def dataset_factory(dataset_list: list[str], **kwargs: object) -> ConcatDataset:
    """ create a combined dataset """

    dataset_map: dict[str, tuple[type[RGBDDataset], ...]] = {
        'tartan': (TartanAir, ),
    }

    db_list: list[RGBDDataset] = []
    for key in dataset_list:
        # cache datasets for faster future loading
        db: RGBDDataset = dataset_map[key][0](**kwargs)

        print(f"Dataset {key} has {len(db)} images")
        db_list.append(db)

    return ConcatDataset(db_list)
