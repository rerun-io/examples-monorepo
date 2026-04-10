"""Data reading and augmentation utilities for DPVO training.

This package provides:

- :mod:`.augmentation` -- spatial and colour augmentations for RGBD video clips.
- :mod:`.base` -- :class:`RGBDDataset`, the abstract base ``Dataset`` class.
- :mod:`.tartan` -- :class:`TartanAir` dataset loader.
- :mod:`.factory` -- :func:`dataset_factory` for building combined datasets.
- :mod:`.frame_utils` -- low-level I/O for optical flow, depth, and camera files.
- :mod:`.rgbd_utils` -- TUM-format loading, pose conversion, and distance matrices.
"""
