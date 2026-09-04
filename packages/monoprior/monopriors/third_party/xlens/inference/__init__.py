"""Self-contained inference utilities for X-Lens.

This package has **no dependency** on the training engine, datasets, or losses.
It provides everything needed to run the released checkpoints:

    preprocess.py  - build model inputs (images, d_cam, ray_map, cam_types)
                     for pinhole / fisheye / heterogeneous camera rigs.
    geometry.py    - unproject predicted depth to a fused world point cloud (.ply).
    pipeline.py    - XLensInference: load a checkpoint and run all 3 modes.
"""
from monopriors.third_party.xlens.inference.pipeline import XLensInference  # noqa: F401
