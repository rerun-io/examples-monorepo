from typing import Literal

import numpy as np
from jaxtyping import Float64, Int64
from numpy import ndarray
from serde import serde


@serde
class MANOData:
    """Structure mirroring the data loaded from MANO .pkl files."""

    hands_components: Float64[ndarray, "45 45"]  # PCA components for hand pose
    f: Float64[ndarray, "1538 3"]  # Faces (indices)
    J_regressor: Float64[ndarray, "16 778"]  # Joint regressor (now dense)
    kintree_table: Int64[ndarray, "2 16"]  # Kinematic tree definition
    J: Float64[ndarray, "16 3"]  # Template joint locations
    bs_style: Literal["lbs"]  # Blend shape style (e.g., 'lbs')
    hands_coeffs: Float64[ndarray, "1554 45"]  # Coefficients for hand pose PCA (if applicable)
    weights: Float64[ndarray, "778 16"]  # Skinning weights
    posedirs: Float64[ndarray, "778 3 135"]  # Pose blend shapes
    hands_mean: Float64[ndarray, "45"]  # Mean hand pose (axis-angle)
    v_template: Float64[ndarray, "778 3"]  # Template vertices
    shapedirs: Float64[ndarray, "778 3 10"]  # Shape blend shapes
    bs_type: Literal["lrotmin"]  # Blend shape type (often same as bs_style)

    def __post_init__(self):
        # Ensure that the data is in the expected format
        self.betas: Float64[ndarray, "10"] = np.zeros(self.shapedirs.shape[-1])


# convert joints between mediapipe and mano format
mp_to_mano: list[int] = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 8, 12, 20, 16, 4]
mano_to_mp: list[int] = [0, 13, 14, 15, 20, 1, 2, 3, 16, 4, 5, 6, 17, 10, 11, 12, 19, 7, 8, 9, 18]
