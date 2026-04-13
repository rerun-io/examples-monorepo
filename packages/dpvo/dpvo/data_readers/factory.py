"""Factory function for constructing combined RGBD training datasets.

Maps short dataset names (e.g. ``"tartan"``) to their concrete
:class:`~mini_dpvo.data_readers.base.RGBDDataset` subclass and returns a
:class:`~torch.utils.data.ConcatDataset` that merges them.
"""

from torch.utils.data import ConcatDataset

from .base import RGBDDataset

# RGBD-Dataset
from .tartan import TartanAir


def dataset_factory(dataset_list: list[str], **kwargs: object) -> ConcatDataset:
    """Create a combined :class:`ConcatDataset` from named dataset keys.

    Args:
        dataset_list: List of dataset identifiers. Currently supported:
            ``"tartan"`` (TartanAir).
        **kwargs: Keyword arguments forwarded to each dataset constructor
            (e.g. ``datapath``, ``n_frames``, ``crop_size``).

    Returns:
        A :class:`ConcatDataset` wrapping all requested datasets.

    Raises:
        KeyError: If a dataset key is not found in the registry.
    """
    dataset_map: dict[str, tuple[type[RGBDDataset], ...]] = {
        'tartan': (TartanAir, ),
    }

    db_list: list[RGBDDataset] = []
    for key in dataset_list:
        db: RGBDDataset = dataset_map[key][0](**kwargs)

        print(f"Dataset {key} has {len(db)} images")
        db_list.append(db)

    return ConcatDataset(db_list)
