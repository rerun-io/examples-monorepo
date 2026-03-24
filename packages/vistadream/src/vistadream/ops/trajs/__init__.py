from typing import Literal

import numpy as np
from jaxtyping import Float

from vistadream.ops.gs.basic import Gaussian_Scene

from .interp import Interp
from .rot import Rot
from .spiral import Spiral
from .wobble import Wobble


def _generate_trajectory(scene: Gaussian_Scene, nframes: int) -> Float[np.ndarray, "n_frames 4 4"]:
    """
    Generate camera trajectory for rendering.

    Args:
        cfg: Configuration object containing trajectory parameters
        scene: Gaussian scene containing trajectory type and parameters
        nframes: Number of frames to generate (optional)

    Returns:
        Float[np.ndarray, "n_frames 4 4"]: Camera trajectory as transformation matrices
            in cam_T_world format (world-to-camera transformations).
    """
    method: Literal["rot", "wobble", "spiral", "interp"] = scene.traj_type
    if method == "rot":
        runner = Rot(scene, nframes)
    elif method == "wobble":
        runner = Wobble(scene, nframes)
    elif method == "spiral":
        runner = Spiral(scene, nframes)
    elif method == "interp":
        runner = Interp(scene, nframes)
    else:
        raise TypeError("method = rot / spiral / wobble / interp")
    return runner()
