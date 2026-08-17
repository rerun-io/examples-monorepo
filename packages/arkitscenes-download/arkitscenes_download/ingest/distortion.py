"""Interpret Apple stream-10 distortion and fit its standard interop model."""

from dataclasses import dataclass

import cv2
import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Float, Float32, Float64
from numpy import ndarray
from scipy.optimize import least_squares
from simplecv.camera_parameters import BrownConradyDistortion

APPLE_FORWARD_DISTORTION_COMPONENT: str = "arkitscenes.components.AppleForwardDistortionCoefficients"
"""Exact Apple distorted-to-rectified polynomial provenance component."""
APPLE_INVERSE_DISTORTION_COMPONENT: str = "arkitscenes.components.AppleInverseDistortionCoefficients"
"""Apple's supplied rectified-to-distorted polynomial provenance component."""


class _AppleDistortionCoefficientsBatch(rr.ComponentBatchMixin):
    """One variable-length Apple coefficient vector with an explicit descriptor."""

    def __init__(self, coefficients: Float[ndarray, "n"], component: str) -> None:
        self.coefficients: Float32[ndarray, "n"] = np.asarray(coefficients, dtype=np.float32).reshape(-1)
        self.component: str = component

    def component_descriptor(self) -> rr.ComponentDescriptor:
        """Describe the provenance component without using the generic distortion slot."""
        return rr.ComponentDescriptor(component=self.component, component_type=self.component)

    def as_arrow_array(self) -> pa.Array:
        """Serialize one coefficient vector as a list-valued component."""
        coefficients: list[float] = self.coefficients.tolist()
        return pa.array([coefficients], type=pa.list_(pa.float32()))


class AppleDistortionProvenance(rr.AsComponents):
    """Exact forward and inverse Apple polynomials in provenance-only components."""

    def __init__(self, forward_coefficients_8: Float[ndarray, "8"], inverse_coefficients_8: Float[ndarray, "8"]) -> None:
        self.forward_coefficients_8: Float32[ndarray, "8"] = np.asarray(forward_coefficients_8, dtype=np.float32).reshape(-1)
        self.inverse_coefficients_8: Float32[ndarray, "8"] = np.asarray(inverse_coefficients_8, dtype=np.float32).reshape(-1)
        if self.forward_coefficients_8.shape != (8,) or self.inverse_coefficients_8.shape != (8,):
            raise ValueError("Apple distortion provenance requires two eight-coefficient vectors")

    def as_component_batches(self) -> list[rr.DescribedComponentBatch]:
        """Return separately described forward and inverse provenance batches."""
        forward_batch: _AppleDistortionCoefficientsBatch = _AppleDistortionCoefficientsBatch(
            self.forward_coefficients_8, APPLE_FORWARD_DISTORTION_COMPONENT
        )
        inverse_batch: _AppleDistortionCoefficientsBatch = _AppleDistortionCoefficientsBatch(
            self.inverse_coefficients_8, APPLE_INVERSE_DISTORTION_COMPONENT
        )
        return [
            forward_batch.described(forward_batch.component_descriptor()),
            inverse_batch.described(inverse_batch.component_descriptor()),
        ]


@dataclass(frozen=True, slots=True)
class AppleRadialPolynomial:
    """Apple's eight-coefficient radial magnification polynomial."""

    coefficients_8: Float32[ndarray, "8"]
    """Percent magnification coefficients for normalized powers t² through t¹⁶."""

    def __post_init__(self) -> None:
        """Normalize and validate the coefficient vector."""
        coefficients_8: Float32[ndarray, "8"] = np.asarray(self.coefficients_8, dtype=np.float32).reshape(-1)
        if coefficients_8.shape != (8,) or not np.all(np.isfinite(coefficients_8)):
            raise ValueError("Apple radial calibration requires eight finite coefficients")
        object.__setattr__(self, "coefficients_8", coefficients_8)

    def distorted_to_rectified_radius(self, distorted_radius: Float32[ndarray, "..."]) -> Float32[ndarray, "..."]:
        """Apply the empirically selected distorted-to-rectified radial map.

        Args:
            distorted_radius: float32 radii normalized by the farthest reference-image corner, with arbitrary shape.

        Returns:
            float32 normalized rectified radii with the input shape.
        """
        radii: Float32[ndarray, "..."] = np.asarray(distorted_radius, dtype=np.float32)
        powers_8: Float32[ndarray, "8"] = (2.0 * np.arange(1, 9, dtype=np.float32)).astype(np.float32)
        magnification: Float32[ndarray, "..."] = (
            0.01 * np.sum(self.coefficients_8 * radii[..., None] ** powers_8, axis=-1)
        ).astype(np.float32, copy=False)
        rectified_radius: Float32[ndarray, "..."] = (radii * (1.0 + magnification)).astype(np.float32, copy=False)
        return rectified_radius

    def rectified_to_distorted_radius(self, rectified_radius: Float32[ndarray, "..."]) -> Float32[ndarray, "..."]:
        """Numerically invert the radial map for OpenCV destination-to-source remapping.

        Args:
            rectified_radius: float32 radii normalized by the farthest reference-image corner, with arbitrary shape.

        Returns:
            float32 normalized distorted radii with the input shape.
        """
        requested: Float32[ndarray, "..."] = np.asarray(rectified_radius, dtype=np.float32)
        sample_distorted_n: Float32[ndarray, "n_samples=100001"] = np.linspace(0.0, 1.25, 100_001, dtype=np.float32)
        sample_rectified_n: Float32[ndarray, "n_samples=100001"] = self.distorted_to_rectified_radius(sample_distorted_n)
        if np.any(sample_rectified_n[1:] <= sample_rectified_n[:-1]):
            raise ValueError("Apple radial polynomial is not invertible over the calibrated frame")
        if np.any(requested < 0.0) or np.any(requested > sample_rectified_n[-1]):
            raise ValueError("Requested rectified radius lies outside the invertible calibration domain")
        distorted_flat_n: Float32[ndarray, "n"] = np.interp(  # pyrefly: ignore[bad-assignment]
            requested.reshape(-1),
            sample_rectified_n,
            sample_distorted_n,
        ).astype(np.float32)
        distorted: Float32[ndarray, "..."] = distorted_flat_n.reshape(requested.shape)
        return distorted


@dataclass(frozen=True, slots=True)
class OpenCVRationalFit:
    """A validated OpenCV rational-model approximation of one Apple calibration."""

    distortion: BrownConradyDistortion
    """Rational radial and tangential terms; thin-prism and tilt terms remain zero."""
    K_33: Float32[ndarray, "3 3"]
    """Unchanged pinhole intrinsics whose principal point is the standard model's only center."""
    max_residual_px: float
    """Measured worst-case distorted-to-rectified error against the exact Apple field."""


def opencv_rational_from_apple(
    K_native_33: Float[ndarray, "3 3"],
    coefficients_8: Float[ndarray, "8"],
    distortion_center_reference_xy: Float[ndarray, "2"],
    reference_dimensions_wh: tuple[int, int],
    image_wh: tuple[int, int],
    residual_bound_px: float = 0.01,
) -> OpenCVRationalFit:
    """Fit OpenCV's rational Brown-Conrady model around the pinhole principal point.

    The exact Apple field remains centered at Apple's distinct distortion center.
    Tangential terms model that small center offset while the returned K stays
    unchanged, matching simplecv's one-center pinhole convention.

    Args:
        K_native_33: Floating-point native-resolution pinhole intrinsics shaped ``3 3``.
        coefficients_8: Floating-point Apple forward coefficients shaped ``8``.
        distortion_center_reference_xy: Floating-point Apple distortion center shaped ``2`` in reference pixels.
        reference_dimensions_wh: Calibration reference width and height.
        image_wh: Native image width and height at which to validate the fit.
        residual_bound_px: Reject the fit if its worst-case error exceeds this value.

    Returns:
        Validated simplecv Brown-Conrady distortion and the unchanged pinhole K.

    Raises:
        ValueError: On anisotropic reference scaling, an invalid source or fitted
            map, or a residual beyond ``residual_bound_px``.
    """
    width: int = image_wh[0]
    height: int = image_wh[1]
    reference_per_image_xy: Float64[ndarray, "2"] = np.array(
        [reference_dimensions_wh[0] / width, reference_dimensions_wh[1] / height], dtype=np.float64
    )
    if abs(reference_per_image_xy[0] - reference_per_image_xy[1]) > 1e-6 * reference_per_image_xy[0]:
        raise ValueError(f"Anisotropic reference scaling {reference_per_image_xy.tolist()} is unsupported")
    model: AppleRadialPolynomial = AppleRadialPolynomial(np.asarray(coefficients_8, dtype=np.float32))
    apple_center_reference_xy: Float64[ndarray, "2"] = np.asarray(distortion_center_reference_xy, dtype=np.float64)
    apple_center_native_xy: Float64[ndarray, "2"] = apple_center_reference_xy / reference_per_image_xy
    K_native: Float64[ndarray, "3 3"] = np.asarray(K_native_33, dtype=np.float64)
    fit_center_native_xy: Float64[ndarray, "2"] = K_native[:2, 2].copy()
    corners_reference_42: Float64[ndarray, "4 2"] = np.array(
        [[0.0, 0.0], [reference_dimensions_wh[0], 0.0], [0.0, reference_dimensions_wh[1]], [*map(float, reference_dimensions_wh)]],
        dtype=np.float64,
    )
    radius_max_reference: float = float(np.max(np.linalg.norm(corners_reference_42 - apple_center_reference_xy, axis=1)))
    focal_fit_px: float = radius_max_reference / float(reference_per_image_xy[0])

    def apple_rectified(distorted_n2: Float64[ndarray, "n 2"]) -> Float64[ndarray, "n 2"]:
        """Evaluate the exact Apple distorted-to-rectified native-pixel field."""
        delta_n2: Float64[ndarray, "n 2"] = (distorted_n2 - apple_center_native_xy) * reference_per_image_xy
        radii_n: Float64[ndarray, "n"] = np.linalg.norm(delta_n2, axis=1)
        mapped_n: Float64[ndarray, "n"] = np.asarray(
            model.distorted_to_rectified_radius((radii_n / radius_max_reference).astype(np.float32)), dtype=np.float64
        )
        scale_n: Float64[ndarray, "n"] = np.divide(
            mapped_n * radius_max_reference, radii_n, out=np.ones_like(radii_n), where=radii_n > 0.0
        )
        return apple_center_native_xy + delta_n2 * scale_n[:, None] / reference_per_image_xy

    def rational_distorted(
        rectified_n2: Float64[ndarray, "n 2"], distortion_8: Float64[ndarray, "8"]
    ) -> Float64[ndarray, "n 2"]:
        """Evaluate OpenCV's rational forward field around the pinhole center."""
        normalized_n2: Float64[ndarray, "n 2"] = (rectified_n2 - fit_center_native_xy) / focal_fit_px
        x_n: Float64[ndarray, "n"] = normalized_n2[:, 0]
        y_n: Float64[ndarray, "n"] = normalized_n2[:, 1]
        r2_n: Float64[ndarray, "n"] = np.sum(np.square(normalized_n2), axis=1)
        numerator_n: Float64[ndarray, "n"] = 1.0 + distortion_8[0] * r2_n + distortion_8[1] * r2_n**2 + distortion_8[4] * r2_n**3
        denominator_n: Float64[ndarray, "n"] = 1.0 + distortion_8[5] * r2_n + distortion_8[6] * r2_n**2 + distortion_8[7] * r2_n**3
        radial_n: Float64[ndarray, "n"] = numerator_n / denominator_n
        tangential_n2: Float64[ndarray, "n 2"] = np.stack(
            [
                2.0 * distortion_8[2] * x_n * y_n + distortion_8[3] * (r2_n + 2.0 * x_n**2),
                distortion_8[2] * (r2_n + 2.0 * y_n**2) + 2.0 * distortion_8[3] * x_n * y_n,
            ],
            axis=1,
        )
        return fit_center_native_xy + focal_fit_px * (normalized_n2 * radial_n[:, None] + tangential_n2)

    x_n: Float64[ndarray, "nx"] = np.unique(np.append(np.arange(0.0, width, 4.0), float(width - 1)))
    y_n: Float64[ndarray, "ny"] = np.unique(np.append(np.arange(0.0, height, 4.0), float(height - 1)))
    x_hw: Float64[ndarray, "ny nx"]
    y_hw: Float64[ndarray, "ny nx"]
    x_hw, y_hw = np.meshgrid(x_n, y_n)
    distorted_fit_n2: Float64[ndarray, "n 2"] = np.stack([x_hw.reshape(-1), y_hw.reshape(-1)], axis=1)
    rectified_fit_n2: Float64[ndarray, "n 2"] = apple_rectified(distorted_fit_n2)

    def residuals(distortion_8: Float64[ndarray, "8"]) -> Float64[ndarray, "m"]:
        """Return forward-direction component residuals on the fit grid."""
        return (rational_distorted(rectified_fit_n2, distortion_8) - distorted_fit_n2).reshape(-1)

    brown_3: Float64[ndarray, "3"] = least_squares(
        lambda k_3: residuals(np.array([k_3[0], k_3[1], 0.0, 0.0, k_3[2], 0.0, 0.0, 0.0])),
        np.zeros(3),
        ftol=1e-13,
        xtol=1e-13,
        gtol=1e-13,
    ).x
    distortion_fit_8: Float64[ndarray, "8"] = least_squares(
        residuals,
        np.array([brown_3[0], brown_3[1], 0.0, 0.0, brown_3[2], 0.0, 0.0, 0.0]),
        ftol=1e-13,
        xtol=1e-13,
        gtol=1e-13,
    ).x

    corners_native_42: Float64[ndarray, "4 2"] = np.array(
        [[0.0, 0.0], [width - 1.0, 0.0], [0.0, height - 1.0], [width - 1.0, height - 1.0]], dtype=np.float64
    )
    rectified_corners_42: Float64[ndarray, "4 2"] = apple_rectified(corners_native_42)
    radius_domain: float = float(np.max(np.linalg.norm(rectified_corners_42 - fit_center_native_xy, axis=1))) / focal_fit_px
    r2_probe_n: Float64[ndarray, "n=4096"] = np.square(np.linspace(0.0, 1.05 * radius_domain, 4096))
    denominator_probe_n: Float64[ndarray, "n=4096"] = (
        1.0 + distortion_fit_8[5] * r2_probe_n + distortion_fit_8[6] * r2_probe_n**2 + distortion_fit_8[7] * r2_probe_n**3
    )
    if np.any(denominator_probe_n <= 0.0):
        raise ValueError("Fitted rational denominator is not positive over the calibrated frame")
    probe_n2: Float64[ndarray, "n=4096 2"] = np.stack(
        [fit_center_native_xy[0] + np.sqrt(r2_probe_n) * focal_fit_px, np.full(r2_probe_n.shape, fit_center_native_xy[1])], axis=1
    )
    distorted_radii_n: Float64[ndarray, "n=4096"] = np.linalg.norm(
        rational_distorted(probe_n2, distortion_fit_8) - fit_center_native_xy, axis=1
    )
    if np.any(np.diff(distorted_radii_n) <= 0.0):
        raise ValueError("Fitted rational radius map is not strictly increasing over the calibrated frame")

    K_fit_33: Float64[ndarray, "3 3"] = np.array(
        [[focal_fit_px, 0.0, fit_center_native_xy[0]], [0.0, focal_fit_px, fit_center_native_xy[1]], [0.0, 0.0, 1.0]]
    )
    x2_n: Float64[ndarray, "nx"] = np.unique(np.append(np.arange(0.0, width, 2.0), float(width - 1)))
    y2_n: Float64[ndarray, "ny"] = np.unique(np.append(np.arange(0.0, height, 2.0), float(height - 1)))
    x2_hw: Float64[ndarray, "ny nx"]
    y2_hw: Float64[ndarray, "ny nx"]
    x2_hw, y2_hw = np.meshgrid(x2_n, y2_n)
    distorted_check_n2: Float64[ndarray, "n 2"] = np.stack([x2_hw.reshape(-1), y2_hw.reshape(-1)], axis=1)
    criteria: tuple[int, int, float] = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-13)
    rectified_check_n12: Float64[ndarray, "n 1 2"] = cv2.undistortPoints(  # pyrefly: ignore  # no-matching-overload
        distorted_check_n2.reshape(-1, 1, 2), K_fit_33, distortion_fit_8, P=K_fit_33, criteria=criteria
    )
    max_residual_px: float = float(
        np.max(np.linalg.norm(rectified_check_n12.reshape(-1, 2) - apple_rectified(distorted_check_n2), axis=1))
    )
    if max_residual_px > residual_bound_px:
        raise ValueError(f"Rational fit residual {max_residual_px:.6f} px exceeds the {residual_bound_px} px bound")

    ratio: float = float(K_native[0, 0]) / focal_fit_px
    distortion_physical_8: Float64[ndarray, "8"] = distortion_fit_8 * np.array(
        [ratio**2, ratio**4, ratio, ratio, ratio**6, ratio**2, ratio**4, ratio**6]
    )
    distortion: BrownConradyDistortion = BrownConradyDistortion(
        k1=float(distortion_physical_8[0]),
        k2=float(distortion_physical_8[1]),
        p1=float(distortion_physical_8[2]),
        p2=float(distortion_physical_8[3]),
        k3=float(distortion_physical_8[4]),
        k4=float(distortion_physical_8[5]),
        k5=float(distortion_physical_8[6]),
        k6=float(distortion_physical_8[7]),
    )
    return OpenCVRationalFit(distortion=distortion, K_33=K_native.astype(np.float32), max_residual_px=max_residual_px)
