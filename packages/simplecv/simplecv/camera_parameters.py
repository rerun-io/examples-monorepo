from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np
from einops import rearrange
from jaxtyping import Float, Float64
from numpy import ndarray

from simplecv.se3 import SE3


@dataclass
class BrownConradyDistortion:
    """Brown–Conrady distortion model with optional thin-prism and tilt terms."""

    k1: float
    k2: float
    p1: float
    p2: float
    k3: float
    k4: float = 0.0
    k5: float = 0.0
    k6: float = 0.0
    s1: float = 0.0
    s2: float = 0.0
    s3: float = 0.0
    s4: float = 0.0
    tau_x: float = 0.0
    tau_y: float = 0.0


@dataclass
class KannalaBrandtDistortion:
    """Kannala–Brandt fisheye distortion model (odd-order radial polynomial)."""

    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    k5: float = 0.0
    k6: float = 0.0
    p1: float = 0.0
    p2: float = 0.0


@dataclass
class Extrinsics:
    # Rotation and translation can be provided for both world-to-camera
    # and camera-to-world transformations
    world_R_cam: Float[ndarray, "3 3"] | None = None
    world_t_cam: Float[ndarray, "3"] | None = None
    cam_R_world: Float[ndarray, "3 3"] | None = None
    cam_t_world: Float[ndarray, "3"] | None = None
    # The projection matrix and transformation matrices will be computed in post-init
    world_T_cam: Float[ndarray, "4 4"] = field(init=False)
    cam_T_world: Float[ndarray, "4 4"] = field(init=False)

    def __post_init__(self) -> None:
        self.compute_transformation_matrices()

    def compute_transformation_matrices(self) -> None:
        # If world-to-camera is provided, compute the transformation matrix and its inverse
        if self.world_R_cam is not None and self.world_t_cam is not None:
            self.world_T_cam: Float[ndarray, "4 4"] = self.compose_transformation_matrix(
                self.world_R_cam, self.world_t_cam
            )
            self.cam_T_world: Float[ndarray, "4 4"] = np.linalg.inv(self.world_T_cam)
            # Extract camera-to-world rotation and translation from the inverse matrix
            self.cam_R_world, self.cam_t_world = self.decompose_transformation_matrix(self.cam_T_world)
        # If camera-to-world is provided, compute the transformation matrix and its inverse
        elif self.cam_R_world is not None and self.cam_t_world is not None:
            self.cam_T_world: Float[ndarray, "4 4"] = self.compose_transformation_matrix(
                self.cam_R_world, self.cam_t_world
            )
            self.world_T_cam: Float[ndarray, "4 4"] = np.linalg.inv(self.cam_T_world)
            # Extract world-to-camera rotation and translation from the inverse matrix
            self.world_R_cam, self.world_t_cam = self.decompose_transformation_matrix(self.world_T_cam)
        else:
            raise ValueError("Either world-to-camera or camera-to-world rotation and translation must be provided.")

    def compose_transformation_matrix(self, R: Float[ndarray, "3 3"], t: Float[ndarray, "3"]) -> Float[ndarray, "4 4"]:
        Rt: Float[ndarray, "3 4"] = np.hstack([R, rearrange(t, "c -> c 1")])
        T: Float[ndarray, "4 4"] = np.vstack([Rt, np.array([0, 0, 0, 1])])
        return T

    def decompose_transformation_matrix(
        self, T: Float[ndarray, "4 4"]
    ) -> tuple[Float[ndarray, "3 3"], Float[ndarray, "3"]]:
        R: Float[ndarray, "3 3"] = T[:3, :3]
        t: Float[ndarray, "3 "] = T[:3, 3]
        return R, t


@dataclass(kw_only=True)
class Intrinsics:
    """Camera intrinsics with two construction paths.

    Usage
    -----
    - From focal lengths and principal point (legacy path):
        ``Intrinsics.from_focal_principal_point(camera_conventions="RDF", fl_x=fx, fl_y=fy, cx=cx, cy=cy, height=H, width=W)``
    - From a full 3x3 K matrix (new path):
        ``Intrinsics.from_k_matrix(camera_conventions="RDF", k_matrix=K, height=H, width=W)``

    Exactly one of these sets must be provided. Passing both will raise a ``ValueError``.
    The class stores both the individual scalars and the derived ``k_matrix`` so downstream
    callers can continue to use either representation.
    """

    camera_conventions: Literal["RDF", "RUB"]
    """RDF(OpenCV): X Right - Y Down - Z Front | RUB (OpenGL): X Right- Y Up - Z Back"""
    height: int
    width: int
    # Either (fl_x, fl_y, cx, cy) OR k_matrix must be provided by the caller.
    fl_x: float | np.floating | None = None
    fl_y: float | np.floating | None = None
    cx: float | np.floating | None = None
    cy: float | np.floating | None = None
    k_matrix: Float[ndarray, "3 3"] | None = None

    def __post_init__(self) -> None:
        provided_k: bool = self.k_matrix is not None
        provided_components: bool = all(value is not None for value in (self.fl_x, self.fl_y, self.cx, self.cy))

        if provided_k and provided_components:
            raise ValueError("Provide either k_matrix or fl_x/fl_y/cx/cy, not both.")
        if not provided_k and not provided_components:
            raise ValueError("You must provide k_matrix or fl_x/fl_y/cx/cy.")

        if provided_k:
            k_matrix: Float[ndarray, "3 3"] = np.asarray(self.k_matrix, dtype=float)
            if k_matrix.shape != (3, 3):
                raise ValueError("k_matrix must have shape (3, 3).")

            self.k_matrix = k_matrix
            self.fl_x = float(k_matrix[0, 0])
            self.fl_y = float(k_matrix[1, 1])
            self.cx = float(k_matrix[0, 2])
            self.cy = float(k_matrix[1, 2])
        else:
            assert self.fl_x is not None and self.fl_y is not None and self.cx is not None and self.cy is not None
            k_matrix: Float[ndarray, "3 3"] = np.array(
                [
                    [self.fl_x, 0, self.cx],
                    [0, self.fl_y, self.cy],
                    [0, 0, 1],
                ],
                dtype=float,
            )
            self.k_matrix = k_matrix

    @classmethod
    def from_k_matrix(
        cls,
        *,
        camera_conventions: Literal["RDF", "RUB"],
        k_matrix: Float[ndarray, "3 3"],
        height: int,
        width: int,
    ) -> "Intrinsics":
        """Construct intrinsics directly from a 3x3 camera matrix.

        Use this when you already have the calibrated K matrix; the focal lengths and
        principal point will be derived from it.
        """

        return cls(
            camera_conventions=camera_conventions,
            k_matrix=k_matrix,
            height=height,
            width=width,
        )

    @classmethod
    def from_focal_principal_point(
        cls,
        *,
        camera_conventions: Literal["RDF", "RUB"],
        fl_x: float | np.floating,
        fl_y: float | np.floating,
        cx: float | np.floating,
        cy: float | np.floating,
        height: int,
        width: int,
    ) -> "Intrinsics":
        """Construct intrinsics from focal lengths and principal point.

        Use this when you have separate fx/fy/cx/cy values; the 3x3 K matrix will
        be built during initialization.
        """

        return cls(
            camera_conventions=camera_conventions,
            fl_x=fl_x,
            fl_y=fl_y,
            cx=cx,
            cy=cy,
            height=height,
            width=width,
        )

    def __repr__(self) -> str:
        return (
            f"Intrinsics(camera_conventions={self.camera_conventions}, "
            f"fl_x={self.fl_x}, fl_y={self.fl_y}, cx={self.cx}, cy={self.cy}, "
            f"height={self.height}, width={self.width})"
        )


@dataclass
class PinholeParameters:
    name: str
    extrinsics: Extrinsics
    intrinsics: Intrinsics
    projection_matrix: Float[ndarray, "3 4"] = field(init=False)
    distortion: BrownConradyDistortion | None = None

    def __post_init__(self) -> None:
        self.compute_projection_matrix()

    def compute_projection_matrix(self) -> None:
        # Compute the projection matrix using k_matrix and world_T_cam
        self.projection_matrix: Float[ndarray, "3 4"] = self.intrinsics.k_matrix @ self.extrinsics.cam_T_world[:3, :]


@dataclass
class Fisheye62Parameters:
    """Kannala–Brandt fisheye camera described by up to 6 radial coefficients."""

    name: str
    extrinsics: Extrinsics
    intrinsics: Intrinsics
    distortion: KannalaBrandtDistortion | None = None
    projection_matrix: Float[ndarray, "3 4"] = field(init=False)

    def __post_init__(self) -> None:
        self.compute_projection_matrix()

    def compute_projection_matrix(self) -> None:
        # Compute the projection matrix using k_matrix and world_T_cam
        self.projection_matrix: Float[ndarray, "3 4"] = self.intrinsics.k_matrix @ self.extrinsics.cam_T_world[:3, :]


@dataclass(frozen=True, slots=True)
class Fisheye624Parameters:
    """Project Aria's FISHEYE624 camera: Fisheye62 plus four thin-prism terms, and its field of view.

    projectaria-tools calls the model ``FisheyeRadTanThinPrism``: six radial
    Kannala–Brandt terms, two tangential and four thin-prism terms, with one
    focal length for both axes. ``simplecv.sensors.camera.fisheye624`` projects
    and unprojects through it.
    """

    name: str
    """Sensor label, e.g. ``"camera-slam-left"``."""
    width: int
    """Image width in pixels."""
    height: int
    """Image height in pixels."""
    params: Float64[ndarray, "15"]
    """``[f, cx, cy, k1..k6, p1, p2, s1..s4]``, projectaria-tools' own order."""
    rig_T_cam: SE3
    """Camera pose in the rig (Aria: device) frame."""
    max_solid_angle: float
    """Half-angle in radians of the cone a point must lie in to project."""
    valid_radius: float | None = None
    """Pixel radius about the principal point that a projection must fall inside, if the sensor has one."""

    def to_fisheye62(self, rig_T_cam: Float64[ndarray, "4 4"] | None = None) -> Fisheye62Parameters:
        """Drop the thin-prism terms and the field of view, for consumers of simplecv's Fisheye62 camera.

        Extrinsics are stored as ``world_R_cam``/``world_t_cam`` holding ``rig_T_cam``:
        this camera's own unless another pose (the same camera in another rig frame) is given.
        """
        if rig_T_cam is None:
            rig_T_cam = self.rig_T_cam.matrix()
        f, cx, cy, k1, k2, k3, k4, k5, k6, p1, p2 = (float(value) for value in self.params[:11])
        return Fisheye62Parameters(
            name=self.name,
            extrinsics=Extrinsics(world_R_cam=rig_T_cam[:3, :3].copy(), world_t_cam=rig_T_cam[:3, 3].copy()),
            intrinsics=Intrinsics.from_focal_principal_point(
                camera_conventions="RDF", fl_x=f, fl_y=f, cx=cx, cy=cy, width=self.width, height=self.height
            ),
            distortion=KannalaBrandtDistortion(k1=k1, k2=k2, k3=k3, k4=k4, k5=k5, k6=k6, p1=p1, p2=p2),
        )


MeiModel: TypeAlias = Literal["mei_v3", "mei_v6"]
"""The Insta360 factory distortion on the unified sphere: ``mei_v3`` (``k1 k2 k3 p1 p2``) or ``mei_v6`` (``d0 .. d12``)."""
MEI_DISTORTION_COUNT: dict[MeiModel, int] = {"mei_v3": 5, "mei_v6": 13}
"""How many factory distortion coefficients each Mei model has."""
MEI_CATALOG_MODELS: tuple[str, ...] = ("mei_v3", "mei_v6", "mei_v3_unified", "mei_v6_unified")
"""The ``DistortionModel`` names a Mei camera is logged under (exoego:v2 section 4).

``mei_v3`` / ``mei_v6``: coefficients ``xi, half_field_rad, distortion...``, and the ``Pinhole`` focal is gamma: what a
base written before the unified names states. ``*_unified``: coefficients ``gamma_x, gamma_y, xi, half_field_rad,
distortion...``, and the ``Pinhole`` focal is the paraxial ``gamma / (1 + xi)``, the viewer's pinhole approximation only."""


@dataclass(frozen=True, slots=True)
class MeiParameters:
    """A unified-sphere (Mei) camera with the Insta360 factory distortion: the model's own intrinsics, typed.

    The projection (kornia-staging ``camera::mei``, insta360-io ``MeiCamera``): a ray ``(x, y, z)`` goes to the sphere,
    shifts by ``xi`` along the axis, is distorted, then ``u = gamma_x * x / (z + xi * r) + cx``. At the axis that is
    ``gamma / (1 + xi)`` pixels per radian, so a plain pinhole with that focal (``paraxial_intrinsics``) is what a
    viewer draws; ``gamma`` itself would draw the frustum ``1 + xi`` times too narrow.
    """

    name: str
    """Sensor label, e.g. ``"cam_00"``."""
    model: MeiModel
    """The factory distortion it carries."""
    width: int
    """Image width in pixels."""
    height: int
    """Image height in pixels."""
    gamma_x: float
    """The model's focal along x, pixels (not a pinhole focal)."""
    gamma_y: float
    """The model's focal along y, pixels."""
    cx: float
    """Principal point x, pixels."""
    cy: float
    """Principal point y, pixels."""
    xi: float
    """The sphere's shift along the optical axis."""
    half_field_rad: float
    """Half the field of view the calibration holds over, radians."""
    distortion: Float64[ndarray, " n"]
    """The factory distortion in the model's order (``MEI_DISTORTION_COUNT``)."""

    def __post_init__(self) -> None:
        """Check the distortion length matches the model."""
        if len(self.distortion) != MEI_DISTORTION_COUNT[self.model]:
            raise ValueError(f"{self.model} takes {MEI_DISTORTION_COUNT[self.model]} distortion coefficients, got {len(self.distortion)}")

    def __eq__(self, other: object) -> bool:
        """Equal fields, the distortion compared by value."""
        if not isinstance(other, MeiParameters):
            return NotImplemented
        scalars = ("name", "model", "width", "height", "gamma_x", "gamma_y", "cx", "cy", "xi", "half_field_rad")
        return all(getattr(self, key) == getattr(other, key) for key in scalars) and bool(np.array_equal(self.distortion, other.distortion))

    def paraxial_intrinsics(self) -> "Intrinsics":
        """The pinhole a viewer draws this camera with: focal ``gamma / (1 + xi)``, the same principal point and size."""
        return Intrinsics.from_focal_principal_point(
            camera_conventions="RDF",
            fl_x=self.gamma_x / (1.0 + self.xi),
            fl_y=self.gamma_y / (1.0 + self.xi),
            cx=self.cx,
            cy=self.cy,
            width=self.width,
            height=self.height,
        )

    def sphere_coefficients(self) -> list[float]:
        """``xi, half_field_rad, distortion...``: the coefficient list a Mei camera takes beside ``gamma_x gamma_y cx cy``
        (insta360-io's ``MeiCamera``, posekit's ``MeiCamera.from_calibration``)."""
        return [self.xi, self.half_field_rad, *(float(value) for value in self.distortion)]

    def catalog_distortion(self) -> tuple[str, Float64[ndarray, " n"]]:
        """The ``DistortionModel`` name and ``DistortionCoefficients`` this camera is logged with: the unified name."""
        coefficients: Float64[ndarray, " n"] = np.array([self.gamma_x, self.gamma_y, self.xi, self.half_field_rad, *self.distortion], dtype=np.float64)
        return f"{self.model}_unified", coefficients

    @classmethod
    def from_catalog(
        cls, model: str, coefficients: Float64[ndarray, " n"], k_matrix: Float64[ndarray, "3 3"], width: int, height: int, name: str = ""
    ) -> "MeiParameters":
        """Read a Mei camera from its logged ``DistortionModel``, ``DistortionCoefficients`` and ``Pinhole`` (any name in
        ``MEI_CATALOG_MODELS``).

        Raises:
            ValueError: If ``model`` is not a Mei name, or the coefficients are too few for it.
        """
        if model not in MEI_CATALOG_MODELS:
            raise ValueError(f"{model!r} is not a Mei model ({', '.join(MEI_CATALOG_MODELS)})")
        unified: bool = model.endswith("_unified")
        base: MeiModel = "mei_v3" if model.startswith("mei_v3") else "mei_v6"
        head: int = 4 if unified else 2
        needed: int = head + MEI_DISTORTION_COUNT[base]
        values: Float64[ndarray, " n"] = np.asarray(coefficients, dtype=np.float64).reshape(-1)
        if len(values) < needed:
            raise ValueError(f"{model} needs {needed} coefficients, got {len(values)}")
        gamma_x, gamma_y = (float(values[0]), float(values[1])) if unified else (float(k_matrix[0, 0]), float(k_matrix[1, 1]))
        return cls(
            name=name,
            model=base,
            width=width,
            height=height,
            gamma_x=gamma_x,
            gamma_y=gamma_y,
            cx=float(k_matrix[0, 2]),
            cy=float(k_matrix[1, 2]),
            xi=float(values[head - 2]),
            half_field_rad=float(values[head - 1]),
            distortion=values[head:needed].copy(),
        )


def to_homogeneous(
    points: Float[np.ndarray, "num_points _"],
) -> Float[np.ndarray, "num_points _"]:
    """
    Converts a set of 3D points to homogeneous coordinates.

    Args:
        points (Float[np.ndarray, "num_points 3"]): A numpy array containing the 3D coordinates of the points.

    Returns:
        Float[np.ndarray, "num_points 4"]: A numpy array containing the homogeneous coordinates of the points.
    """
    ones_column: Float[ndarray, "num_points 1"] = np.ones((points.shape[0], 1), dtype=points.dtype)
    return np.hstack([points, ones_column])


def from_homogeneous(
    points_hom: Float[np.ndarray, "num_points _"],
) -> Float[np.ndarray, "num_points _"]:
    """
    Converts a set of 3D points from homogeneous coordinates to Euclidean coordinates.

    Args:
        points (Float[np.ndarray, "num_points 4"]): A numpy array containing the homogeneous coordinates of the points.

    Returns:
        Float[np.ndarray, "num_points 3"]: A numpy array containing the 3D coordinates of the points.
    """
    points = points_hom / points_hom[:, 3:]
    return points[:, :3]


def rescale_intri(camera_intrinsics: Intrinsics, *, target_width: int, target_height: int) -> Intrinsics:
    """
    Rescales the input image and intrinsic matrix by a given scale factor.

    Args:
        camera_intrinsics: The camera intrinsics to rescale.

    Returns:
        Intrinsics: Rescaled copy of the input intrinsics.
    """
    assert camera_intrinsics.height is not None, "Set Camera Height, currently None"
    assert camera_intrinsics.width is not None, "Set Camera Width, currently None"
    x_scale: float = target_width / camera_intrinsics.width
    y_scale: float = target_height / camera_intrinsics.height

    new_fl_x: float = camera_intrinsics.fl_x * x_scale
    new_fl_y: float = camera_intrinsics.fl_y * y_scale

    rescaled_intri = Intrinsics(
        camera_conventions=camera_intrinsics.camera_conventions,
        fl_x=new_fl_x,
        fl_y=new_fl_y,
        cx=camera_intrinsics.cx * x_scale,
        cy=camera_intrinsics.cy * y_scale,
        height=target_height,
        width=target_width,
    )

    return rescaled_intri


def perspective_projection(
    points_3d: Float[np.ndarray, "num_points 3"], K: Float[np.ndarray, "3 3"]
) -> Float[np.ndarray, "num_points 2"]:
    """
    Project 3D points in camera coordinates to 2D using perspective projection

    Args:
        points_3d: A numpy array of shape (num_points, 3) representing the 3D points in camera coordinates to project
        K: A numpy array of shape (3, 3) representing the camera intrinsic matrix

    Returns:
        A numpy array of shape (num_points, 2) representing the 2D image coordinates of the projected points
    """
    # Apply the camera intrinsic matrix to the 3D points to obtain the 2D image coordinates in homogeneous coordinates
    points_2d_hom = (K @ points_3d.T).T
    # Convert the homogeneous coordinates to Euclidean coordinates by dividing by the third coordinate
    points_2d = points_2d_hom[:, :2] / points_2d_hom[:, 2:]
    return points_2d


def arctan_projection(
    points_3d_cam: Float[np.ndarray, "num_points 3"], K: Float[np.ndarray, "3 3"]
) -> Float[np.ndarray, "num_points 2"]:
    """
    Project 3D points in camera coordinates to 2D using arctan projection

    Args:
        points_3d: A numpy array of shape (num_points, 3) representing the 3D points in camera coordinates to project
        K: A numpy array of shape (3, 3) representing the camera intrinsic matrix

    Returns:
        A numpy array of shape (num_points, 2) representing the 2D image coordinates of the projected points
    """
    # Compute the radial distance of each 3D point from the camera center
    r: Float[ndarray, "num_points"] = np.sqrt(np.sum(np.square(points_3d_cam[:, :2]), axis=-1))
    eps: float = 2.0**-128
    # Compute the angles of the 2D image coordinates with respect to the camera center using arctan2
    s: Float[ndarray, "num_points"] = np.arctan2(r, points_3d_cam[:, 2]) / np.maximum(r, eps)
    # Scale the angles by the radial distance to obtain the final 2D image coordinates in camera coordinates
    points_2d_cam: Float[ndarray, "num_points 2"] = np.zeros((points_3d_cam.shape[0], 2))
    points_2d_cam[:, 0] = points_3d_cam[:, 0] * s
    points_2d_cam[:, 1] = points_3d_cam[:, 1] * s
    # Convert the camera coordinates to homogeneous coordinates
    points_2d_hom: Float[ndarray, "num_points 3"] = to_homogeneous(points_2d_cam)
    # Apply the camera intrinsic matrix to the homogeneous coordinates to obtain the final 2D image coordinates in homogeneous coordinates
    points_2d: Float[ndarray, "num_points 3"] = (K @ points_2d_hom.T).T
    # Convert the homogeneous coordinates to Euclidean coordinates by dividing by the third coordinate
    points_2d: Float[ndarray, "num_points 2"] = points_2d[:, :2] / points_2d[:, 2:]
    return points_2d


def apply_radial_tangential_distortion(
    distortion: KannalaBrandtDistortion, points2d: Float[np.ndarray, "num_points 2"]
) -> Float[np.ndarray, "num_points 2"]:
    """
    Applies radial and tangential distortion to normalized 2D points.

    Args:
        dist_coeffs (Float[np.ndarray, "8"]): The distortion coefficients.
        points2d (Float[np.ndarray, "num_points 2"]): A numpy array containing the normalized 2D coordinates of the points.

    Returns:
        Float[np.ndarray, "num_points 2"]: A numpy array containing the 2D coordinates of the distorted points.

    Note:
        The points2d input should be normalized before being passed to this function.
    """
    k1, k2, p1, p2, k3, k4, k5, k6 = (
        distortion.k1,
        distortion.k2,
        distortion.p1,
        distortion.p2,
        distortion.k3,
        distortion.k4,
        distortion.k5,
        distortion.k6,
    )
    # radial component
    r2 = (points2d * points2d).sum(axis=-1, keepdims=True)
    r2 = np.clip(r2, -(np.pi**2), np.pi**2)
    r4 = r2 * r2
    r6 = r2 * r4
    r8 = r4 * r4
    r10 = r4 * r6
    r12 = r6 * r6
    radial = 1 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8 + k5 * r10 + k6 * r12
    uv = points2d * radial

    # tangential component
    x, y = uv[..., 0], uv[..., 1]
    x2 = x * x
    y2 = y * y
    xy = x * y
    r2 = x2 + y2
    x += 2 * p2 * xy + p1 * (r2 + 2 * x2)
    y += 2 * p1 * xy + p2 * (r2 + 2 * y2)
    return np.stack((x, y), axis=-1)
