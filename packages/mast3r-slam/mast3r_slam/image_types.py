from typing import Annotated

import numpy as np
from beartype.vale import Is
from jaxtyping import Float
from numpy import ndarray


def _is_rgb_normalized(rgb: ndarray) -> bool:
    return bool(
        rgb.ndim == 3
        and rgb.shape[-1] == 3
        and np.issubdtype(rgb.dtype, np.floating)
        and np.all((rgb >= 0.0) & (rgb <= 1.0))
    )


RgbNormalized = Annotated[Float[ndarray, "h w 3"], Is[_is_rgb_normalized]]
BgrNormalized = Annotated[Float[ndarray, "h w 3"], Is[_is_rgb_normalized]]
