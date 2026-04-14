"""Fast bundle adjustment module wrapping CUDA Gauss-Newton solver.

Re-exports:

- :func:`BA` -- run Gauss-Newton bundle adjustment with Schur complement.
- :func:`neighbors` -- query the co-visibility neighbourhood of a patch.
- :func:`reproject` -- reproject 3-D patches into target frames.
- :func:`solve_system` -- sparse PGO solve for loop closure.
"""

from .ba import BA as BA
from .ba import neighbors as neighbors
from .ba import reproject as reproject
from .ba import solve_system as solve_system
