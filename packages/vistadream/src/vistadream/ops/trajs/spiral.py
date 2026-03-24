import numpy as np
from jaxtyping import Float
from numpy import ndarray

from vistadream.ops.gs.basic import Frame, Gaussian_Scene

from .basic import Traj_Base


class Spiral(Traj_Base):
    """
    Generates a spiral camera trajectory for 3D scene visualization.

    The Spiral trajectory creates a smooth camera path that moves in a spiral pattern around
    the scene center, with orientation-aware damping to optimize movement for different image
    aspect ratios. The camera follows a sinusoidal spiral path with configurable parameters
    for radius, rotation speed, depth variation, and movement damping.

    Key Behaviors:
    - **Landscape images**: Full horizontal movement, damped vertical movement (0.3x)
    - **Portrait images**: Damped horizontal (0.6x) and vertical movement (0.3x)
    - **Spiral pattern**: Camera orbits in a sinusoidal spiral around scene center
    - **Depth variation**: Camera moves forward/backward with configurable ratios

    Mathematical Pattern:
    ```
    t = [0, 1] over nframe steps
    r = sin(2πt) * radius * rot_ratio  # Radial distance oscillates
    θ = 2πt * nframe                   # Angular rotation
    x = r * cos(θ) * x_damping         # Horizontal position
    y = r * sin(θ) * y_damping         # Vertical position
    z = -r * (forward_ratio or backward_ratio)  # Depth position
    ```

    Attributes:
        x_damping (float): Horizontal movement damping factor (1.0 landscape, 0.6 portrait)
        y_damping (float): Vertical movement damping factor (0.3 both orientations)
        rot_ratio (float): Controls spiral tightness/amplitude (0.3 = moderate spiral)
        look_at_ratio (float): Distance to look-at target relative to radius (0.5 = half radius)
        forward_ratio (float): Depth scaling when moving forward (z > 0) from scene (0.3)
        backward_ratio (float): Depth scaling when moving backward (z < 0) from scene (0.4)
        radius (float): Base movement radius calculated from scene 3D extent
        nframe (int): Number of trajectory frames to generate

    Inherited Parameters (from scene):
        traj_min_percentage (int): Percentile for scene bounds calculation (5th percentile)
        traj_max_percentage (int): Percentile for scene bounds calculation (50th percentile)

    The radius is automatically calculated from the scene's 3D point cloud:
    1. Convert depth map to 3D coordinates using camera intrinsics
    2. Calculate 5th and 50th percentiles of x, y, z coordinates
    3. Set radius = mean(max_percentile - min_percentile) across all axes

    This ensures the trajectory scale adapts to each scene's actual spatial extent.
    """

    def __init__(self, scene: Gaussian_Scene, nframe: int = 100) -> None:
        super().__init__(scene, nframe)
        # special parameters for spiral
        self._set_orientation_aware_damping()
        self.rot_ratio: float = 0.3
        self.look_at_ratio: float = 0.5
        self.forward_ratio: float = self.scene.traj_forward_ratio
        self.backward_ratio: float = self.scene.traj_backward_ratio

    def _set_orientation_aware_damping(self) -> None:
        """
        Set damping factors based on image orientation.

        This method analyzes the first frame's aspect ratio to determine optimal
        camera movement damping for the given image orientation:

        **Landscape Images (width > height):**
        - x_damping = 1.0: Allow full horizontal movement to showcase wider scenes
        - y_damping = 0.3: Reduce vertical movement to maintain stability and keep
          important elements (floor, sky) in view

        **Portrait Images (height >= width):**
        - x_damping = 0.6: Moderately reduce horizontal movement since the narrower
          field of view means less horizontal content to explore
        - y_damping = 0.3: Reduce vertical movement to prevent losing subject focus

        The damping factors multiply the calculated x,y positions, effectively scaling
        the camera movement amplitude in each axis.
        """
        first_frame: Frame = self.scene.frames[0]
        height: int = first_frame.H
        width: int = first_frame.W

        aspect_ratio = width / height
        if aspect_ratio > 1.0:  # Landscape
            self.x_damping: float = 1.0
            self.y_damping: float = 0.3  # damp y axis for landscape
        else:  # Portrait or square
            self.x_damping: float = 0.6  # damp both axes for portrait
            self.y_damping: float = 0.3

    def camera_target_up(self) -> list[tuple[Float[ndarray, "3"], Float[ndarray, "3"], Float[ndarray, "3"]]]:
        """
        Generates a list of camera configuration tuples for each frame along a spiral trajectory.

        Each tuple contains:
            - The camera position as a 3D vector.
            - The camera target (look-at point) as a 3D vector.
            - The camera up direction as a 3D vector.

        The camera moves along a spiral path, with configurable radius, rotation, and forward/backward ratios.
        The up direction is fixed to point downward along the Y-axis, and the target is fixed in front of the camera.

        Returns:
            list[tuple[Float[ndarray, "3"], Float[ndarray, "3"], Float[ndarray, "3"]]]:
                A list of (position, target, up) tuples for each frame.
        """
        # === SPIRAL TRAJECTORY CALCULATION ===

        # Step 1: Generate time parameter from 0 to 1 over all frames
        t: Float[ndarray, "n_frames"] = np.linspace(0, 1, self.nframe)

        # Step 2: Calculate radial distance - oscillates sinusoidally from 0 to max
        # r determines how far from center the camera moves at each time step
        r: Float[ndarray, "n_frames"] = np.sin(2 * np.pi * t) * self.radius * self.rot_ratio

        # Step 3: Calculate rotation angle - camera spins around scene center
        # theta determines the angular position around the spiral
        theta: Float[ndarray, "n_frames"] = 2 * np.pi * t * self.nframe

        # Step 4: Convert polar coordinates (r, theta) to cartesian (x, y)
        # Apply orientation-aware damping to constrain movement based on image aspect ratio
        x: Float[ndarray, "n_frames"] = r * np.cos(theta) * self.x_damping
        y: Float[ndarray, "n_frames"] = r * np.sin(theta) * self.y_damping

        # Step 5: Calculate depth (z) position - negative means camera is in front of scene
        z: Float[ndarray, "n_frames"] = -r

        # Step 6: Apply asymmetric depth scaling based on camera position relative to scene
        z[z < 0] *= self.forward_ratio  # When camera is in front of scene (negative z)
        z[z > 0] *= self.backward_ratio  # When camera is behind scene (positive z)

        # Step 7: Combine x, y, z into 3D position array
        pos: Float[ndarray, "n_frames 3"] = np.vstack([x, y, z]).T

        # Step 8: Define camera orientation - up vector points down in image space
        camera_ups: Float[ndarray, "n_frames 3"] = np.array([[0, -1.0, 0]]).repeat(self.nframe, axis=0)

        # Step 9: Define look-at targets - fixed point in front of scene center
        targets: Float[ndarray, "n_frames 3"] = np.array([[0, 0, self.radius * self.look_at_ratio]]).repeat(
            self.nframe, axis=0
        )

        # Step 10: Package camera poses as (position, target, up) tuples
        cameras = []
        for i in range(self.nframe):
            cam: tuple[Float[ndarray, "3"], Float[ndarray, "3"], Float[ndarray, "3"]] = (
                pos[i],
                targets[i],
                camera_ups[i],
            )
            cameras.append(cam)
        return cameras

    def __call__(self) -> Float[ndarray, "n_frames 4 4"]:
        """
        Generates a spiral camera trajectory with a look-down perspective and enhanced sway.

        Returns:
            Float[ndarray, "n_frames 4 4"]:
                An array of shape (n_frames, 4, 4) representing the camera-to-world transformation matrices
                for each frame along the spiral trajectory.
        """
        cameras: list[tuple[Float[ndarray, "3"], Float[ndarray, "3"], Float[ndarray, "3"]]] = self.camera_target_up()
        world_T_cam_trajs: Float[np.ndarray, "n_frames 4 4"] = self.trans_by_look_at(cameras)
        cam_T_world_trajs: list[Float[np.ndarray, "1 4 4"]] = [
            np.linalg.inv(world_T_cam)[None] for world_T_cam in world_T_cam_trajs
        ]
        cam_T_world_trajs.reverse()
        cam_T_world_traj: Float[ndarray, "n_frames 4 4"] = np.concatenate(cam_T_world_trajs, axis=0)
        return cam_T_world_traj
