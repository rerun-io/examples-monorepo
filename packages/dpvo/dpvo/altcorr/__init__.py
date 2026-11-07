"""Correlation and patchification on the kernels of :mod:`dpvo.kernels`.

Re-exports the two main public functions:

- :func:`corr` -- compute dot-product correlation volumes between feature
  maps within a local search neighbourhood.
- :func:`patchify` -- extract feature patches at sub-pixel coordinates with
  optional bilinear interpolation.
"""

from .correlation import corr as corr
from .correlation import patchify as patchify