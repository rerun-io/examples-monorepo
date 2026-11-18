"""Cube-face rig geometry: join a lens's half-face with the other lens's half-face into a full 90 deg face.

The cube-faces layer (casa-loma CONTRACT "Cube faces") gives each lens one N x N ``front`` face and four
half-faces (``left``/``right``/``up``/``down``) of 0.6N x N or N x 0.6N: the part of a full 90 deg cube face
that this lens sees, with an off-centre principal point. MoGe expects a centred pinhole image, so a half-face
is completed to the full N x N face: its own pixels at their place in the full face, and the remaining part
warped in from the other lens's half-face that looks the same way (context only: rotation-only homography,
the 33 mm lens baseline is ignored there). A prediction on the full face is cropped back to the half-face's own
pixels, so prior pixels are exactly the training image's pixels.
"""

import re
from dataclasses import dataclass

import cv2
import numpy as np
from jaxtyping import Bool, Float32, Float64, Shaped, UInt8, UInt16
from numpy import ndarray

PIXEL_CENTRE_PX: float = 0.0
"""Where pixel centres sit in K's image coordinates: 0.0 = integers (OpenCV, as the cube-faces layer writes
K: the front face has cx = (N - 1) / 2), 0.5 = half-integers (COLMAP, Rerun)."""

FACE_NAME_PATTERN: re.Pattern[str] = re.compile(r"^cube_(?P<lens>\d+)_(?P<direction>front|left|right|up|down)$")
"""Face names of the cube layer, ``cube_<lens>_<direction>``."""


@dataclass(frozen=True, slots=True)
class CubeFace:
    """One virtual pinhole of the cube-faces layer."""

    entity: str
    """Camera node, e.g. ``/world/rig_00/cam_03``."""
    name: str
    """Face name ``cube_<lens>_<direction>`` (the node's static ``name``)."""
    k_matrix: Float64[ndarray, "3 3"]
    """Pinhole intrinsics, pixels."""
    width: int
    """Image width, pixels."""
    height: int
    """Image height, pixels."""
    face_R_rig: Float64[ndarray, "3 3"]
    """Rotation taking rig (IMU) coordinates to face camera coordinates (RDF)."""

    def __post_init__(self) -> None:
        """Validate the name pattern."""
        if FACE_NAME_PATTERN.match(self.name) is None:
            raise ValueError(f"{self.entity}: face name {self.name!r} is not cube_<lens>_<front|left|right|up|down>")

    @property
    def lens(self) -> int:
        """Lens index parsed from the name."""
        match: re.Match[str] | None = FACE_NAME_PATTERN.match(self.name)
        assert match is not None
        return int(match["lens"])

    @property
    def is_front(self) -> bool:
        """Whether this is the lens's full front face (no join needed)."""
        return self.name.endswith("_front")


@dataclass(frozen=True, slots=True)
class FullFaceWindow:
    """Where a half-face's pixels sit inside its full N x N face (centred principal point, same focal)."""

    side_px: int
    """N, the full face's width and height."""
    x0_px: int
    """Column of the half-face's pixel (0, 0) in the full face."""
    y0_px: int
    """Row of the half-face's pixel (0, 0) in the full face."""
    width: int
    """Half-face width."""
    height: int
    """Half-face height."""

    def crop(self, full: Shaped[ndarray, "n n *channels"]) -> Shaped[ndarray, "h w *channels"]:
        """Cut the half-face's own region out of a full-face array."""
        return full[self.y0_px : self.y0_px + self.height, self.x0_px : self.x0_px + self.width]


def _integral(value: float, what: str) -> int:
    rounded: int = round(value)
    if abs(value - rounded) > 1e-3:
        raise ValueError(f"{what} = {value} is not an integer number of pixels")
    return rounded


def full_face_centre_px(side_px: int, pixel_centre_px: float) -> float:
    """The principal point coordinate of a centred N x N face in the given pixel-centre convention."""
    return (side_px - 1) / 2.0 + pixel_centre_px


def full_face_window(face: CubeFace, pixel_centre_px: float = PIXEL_CENTRE_PX) -> FullFaceWindow:
    """Place a face inside its full 90 deg face: N = 2 fx, principal point at the image centre.

    A wrong ``pixel_centre_px`` shows up here: a front face then lands half a pixel off its full face.

    Raises:
        ValueError: If fx != fy, the offsets are not whole pixels, or the face does not fit inside N x N.
    """
    fx: float = float(face.k_matrix[0, 0])
    fy: float = float(face.k_matrix[1, 1])
    if abs(fx - fy) > 1e-6 * fx:
        raise ValueError(f"{face.name}: fx {fx} != fy {fy}; a cube face has square pixels")
    side_px: int = _integral(2.0 * fx, f"{face.name} 2 fx")
    centre_px: float = full_face_centre_px(side_px, pixel_centre_px)
    x0_px: int = _integral(centre_px - float(face.k_matrix[0, 2]), f"{face.name} full-face centre - cx")
    y0_px: int = _integral(centre_px - float(face.k_matrix[1, 2]), f"{face.name} full-face centre - cy")
    if x0_px < 0 or y0_px < 0 or x0_px + face.width > side_px or y0_px + face.height > side_px:
        raise ValueError(f"{face.name}: {face.width}x{face.height} at ({x0_px}, {y0_px}) does not fit the {side_px}x{side_px} full face")
    return FullFaceWindow(side_px=side_px, x0_px=x0_px, y0_px=y0_px, width=face.width, height=face.height)


def opposite_half_face(face: CubeFace, faces: tuple[CubeFace, ...]) -> CubeFace:
    """The other lens's half-face whose optical axis is closest to this face's (it completes the full face).

    Raises:
        ValueError: For a front face, or when no other-lens half-face looks within 10 deg of the same way.
    """
    if face.is_front:
        raise ValueError(f"{face.name} is a front face; it needs no join")
    axis_rig: Float64[ndarray, "3"] = face.face_R_rig[2]
    candidates: list[CubeFace] = [other for other in faces if other.lens != face.lens and not other.is_front]
    if not candidates:
        raise ValueError(f"{face.name}: no half-face of another lens in the layer")
    alignments: list[float] = [float(axis_rig @ other.face_R_rig[2]) for other in candidates]
    best: int = int(np.argmax(alignments))
    if alignments[best] < np.cos(np.deg2rad(10.0)):
        raise ValueError(
            f"{face.name}: the best other-lens half-face {candidates[best].name} is {np.rad2deg(np.arccos(min(alignments[best], 1.0))):.1f} deg off"
        )
    return candidates[best]


def full_from_other_homography(face: CubeFace, other: CubeFace, pixel_centre_px: float = PIXEL_CENTRE_PX) -> Float64[ndarray, "3 3"]:
    """Homography from the other half-face's pixels to this face's full-face pixels (rotation only).

    ``x_full ~ K_full face_R_rig other_R_rig^T K_other^-1 x_other``, with ``K_full`` = this face's focal and a
    centred principal point.
    """
    window: FullFaceWindow = full_face_window(face, pixel_centre_px)
    full_k: Float64[ndarray, "3 3"] = face.k_matrix.copy()
    full_k[0, 2] = full_face_centre_px(window.side_px, pixel_centre_px)
    full_k[1, 2] = full_face_centre_px(window.side_px, pixel_centre_px)
    face_R_other: Float64[ndarray, "3 3"] = face.face_R_rig @ other.face_R_rig.T
    return full_k @ face_R_other @ np.linalg.inv(other.k_matrix)


def join_half_faces(
    own_rgb: UInt8[ndarray, "h w 3"],
    other_rgb: UInt8[ndarray, "h2 w2 3"],
    face: CubeFace,
    other: CubeFace,
    pixel_centre_px: float = PIXEL_CENTRE_PX,
) -> UInt8[ndarray, "n n 3"]:
    """Complete a half-face to its full N x N face: other lens warped in as context, own pixels pasted on top.

    Args:
        own_rgb: This half-face's image, ``face.height x face.width``.
        other_rgb: The opposite half-face's image (see :func:`opposite_half_face`).
        face: The half-face being completed.
        other: The other lens's half-face.
        pixel_centre_px: Pixel-centre convention of the faces' K (see :data:`PIXEL_CENTRE_PX`).

    Returns:
        The full face image; where the other lens does not reach (a sliver at the edge), its border pixels repeat.
    """
    window: FullFaceWindow = full_face_window(face, pixel_centre_px)
    # OpenCV's warp puts pixel centres at integers; shift K's convention onto it.
    cv_from_k: Float64[ndarray, "3 3"] = np.array([[1.0, 0.0, -pixel_centre_px], [0.0, 1.0, -pixel_centre_px], [0.0, 0.0, 1.0]])
    homography: Float64[ndarray, "3 3"] = cv_from_k @ full_from_other_homography(face, other, pixel_centre_px) @ np.linalg.inv(cv_from_k)
    full_rgb: UInt8[ndarray, "n n 3"] = cv2.warpPerspective(
        other_rgb, homography, (window.side_px, window.side_px), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    full_rgb[window.y0_px : window.y0_px + window.height, window.x0_px : window.x0_px + window.width] = own_rgb
    return full_rgb


def depth_png_mm(depth_m: Float32[ndarray, "h w"], valid: Bool[ndarray, "h w"]) -> UInt16[ndarray, "h w"]:
    """16-bit z-depth in millimetres for spirula's ``depths/``: 0 = invalid, beyond 65.535 m = invalid.

    Args:
        depth_m: Float32 z-depth in metres, shape ``h w``.
        valid: Bool ``h w`` validity mask.

    Returns:
        UInt16 ``h w`` depth in millimetres.
    """
    depth_mm: Float32[ndarray, "h w"] = np.round(depth_m * 1000.0).astype(np.float32)
    keep: Bool[ndarray, "h w"] = valid & np.isfinite(depth_mm) & (depth_mm > 0.0) & (depth_mm <= 65535.0)
    return np.where(keep, depth_mm, 0.0).astype(np.uint16)


def normal_png_rgb(normals: Float32[ndarray, "h w 3"], valid: Bool[ndarray, "h w"]) -> UInt8[ndarray, "h w 3"]:
    """RGB normals for spirula's ``normals/``: ``127.5 + 127.5 n`` in the OpenCV camera frame, black = none.

    Args:
        normals: Float32 unit normals ``h w 3`` in the camera frame (RDF), facing the camera.
        valid: Bool ``h w`` validity mask.

    Returns:
        UInt8 ``h w 3`` RGB image.
    """
    encoded: Float32[ndarray, "h w 3"] = np.clip(np.round(127.5 + 127.5 * normals), 0.0, 255.0).astype(np.float32)
    return np.where(valid[..., None], encoded, 0.0).astype(np.uint8)
