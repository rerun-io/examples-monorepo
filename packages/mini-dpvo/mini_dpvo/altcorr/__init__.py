"""Alternative correlation module wrapping CUDA kernels for correlation and patchification.

Re-exports the two main public functions:

- :func:`corr` -- compute dot-product correlation volumes between feature
  maps within a local search neighbourhood.
- :func:`patchify` -- extract feature patches at sub-pixel coordinates with
  optional bilinear interpolation.
"""

from .correlation import corr, patchify