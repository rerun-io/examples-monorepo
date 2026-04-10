"""Fast bundle adjustment module wrapping CUDA Gauss-Newton solver.

Re-exports:

- :func:`BA` -- run Gauss-Newton bundle adjustment with Schur complement.
- :func:`neighbors` -- query the co-visibility neighbourhood of a patch.
- :func:`reproject` -- reproject 3-D patches into target frames.
"""

from .ba import BA, neighbors, reproject