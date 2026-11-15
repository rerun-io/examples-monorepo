import sys
from typing import Annotated

import numpy as np
from jaxtyping import Float
from numpy import ndarray


def _is_rgb_normalized(rgb: ndarray) -> bool:
    return bool(
        rgb.ndim == 3
        and rgb.shape[-1] == 3
        and np.issubdtype(rgb.dtype, np.floating)
        and np.all((rgb >= 0.0) & (rgb <= 1.0))
    )


def _range_check() -> object:
    """beartype's [0, 1] range validator under pytest (only the package conftest imports beartype), else a plain note."""
    if "beartype.claw" not in sys.modules:
        return "values in [0, 1]"
    from beartype.vale import Is

    return Is[_is_rgb_normalized]


RgbNormalized = Annotated[Float[ndarray, "h w 3"], _range_check()]
