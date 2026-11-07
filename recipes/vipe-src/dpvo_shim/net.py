# [vipe-src] Replaces ViPE's vendored DPVO network (see __init__.py).
"""Re-exports :mod:`dpvo.net`: ``VONet`` (``slam.py``), ``Patchifier``, ``DIM`` and ``coords_grid_with_index`` (``dino_fusion.py``)."""

from dpvo.net import DIM as DIM
from dpvo.net import Patchifier as Patchifier
from dpvo.net import VONet as VONet
from dpvo.net import coords_grid_with_index as coords_grid_with_index
