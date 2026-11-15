"""Native hand-pose fit for handtrack (numpy API over the Rust core in ``handfit._core``)."""

from handfit import _core

__version__: str = _core.__version__

# The checked numpy API is implemented in Rust; importing this package needs
# only numpy, and never imports torch. Shape/dtype signatures live in _core.pyi.
HandFitter = _core.HandFitter
FitOutput = _core.FitOutput
