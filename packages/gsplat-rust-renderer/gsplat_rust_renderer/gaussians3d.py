"""Tiny Python-side Gaussian splat logging helper.

Provides the :class:`Gaussians3D` dataclass that implements
:class:`rerun.AsComponents` so it can be passed directly to :func:`rerun.log`.
The companion Rust viewer registers a custom visualizer that knows how to
render these component batches as GPU-accelerated Gaussian splats.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rerun as rr
from jaxtyping import Float32, UInt8
from plyfile import PlyData

SH_C0: np.float32 = np.float32(0.2820948)
"""Zeroth spherical-harmonic coefficient (1 / (2 * sqrt(pi)))."""


def _component_descriptor(component: str, component_type: str) -> rr.ComponentDescriptor:
    """Build a Rerun component descriptor for the GaussianSplats3D archetype.

    Args:
        component: Component name within the archetype.
        component_type: Rerun component type string.

    Returns:
        A :class:`rerun.ComponentDescriptor` instance.
    """
    return rr.ComponentDescriptor(
        archetype="GaussianSplats3D",
        component=component,
        component_type=component_type,
    )


def _as_float32(name: str, values: npt.ArrayLike, shape_tail: tuple[int, ...]) -> Float32[np.ndarray, "..."]:
    """Validate and cast *values* to a contiguous float32 array with the expected trailing shape.

    Args:
        name: Human-readable name for error messages.
        values: Input array-like.
        shape_tail: Expected trailing dimensions (e.g. ``(3,)`` for ``[N, 3]``).

    Returns:
        A contiguous ``float32`` array.

    Raises:
        ValueError: If the shape does not match.
    """
    array: Float32[np.ndarray, "..."] = np.asarray(values, dtype=np.float32)
    if array.ndim != len(shape_tail) + 1 or tuple(array.shape[1:]) != shape_tail:
        msg: str = f"{name} must have shape [N, {', '.join(map(str, shape_tail))}]"
        raise ValueError(msg)
    return np.ascontiguousarray(array)


def _as_float32_1d(name: str, values: npt.ArrayLike) -> Float32[np.ndarray, "n"]:
    """Validate and cast *values* to a 1-D contiguous float32 array.

    Args:
        name: Human-readable name for error messages.
        values: Input array-like.

    Returns:
        A 1-D contiguous ``float32`` array.

    Raises:
        ValueError: If the array is not 1-D.
    """
    array: Float32[np.ndarray, "n"] = np.asarray(values, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape [N]")
    return np.ascontiguousarray(array)


def _normalize_quaternions_xyzw(quaternions_xyzw: Float32[np.ndarray, "n 4"]) -> Float32[np.ndarray, "n 4"]:
    """Normalize quaternions to unit length, replacing near-zero quaternions with identity.

    Args:
        quaternions_xyzw: Quaternions in ``[x, y, z, w]`` order.

    Returns:
        Unit-length quaternions with the same shape.
    """
    norms: Float32[np.ndarray, "n 1"] = np.linalg.norm(quaternions_xyzw, axis=1, keepdims=True)
    identity: Float32[np.ndarray, "1 4"] = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    return np.where(norms > 1e-12, quaternions_xyzw / np.maximum(norms, 1e-12), identity)


def _sigmoid(x: Float32[np.ndarray, "n"]) -> Float32[np.ndarray, "n"]:
    """Element-wise sigmoid activation.

    Args:
        x: Input array.

    Returns:
        Sigmoid of *x*.
    """
    return 1.0 / (1.0 + np.exp(-x))


def _sh_dc_to_rgb(dc_coefficients: Float32[np.ndarray, "n 3"]) -> Float32[np.ndarray, "n 3"]:
    """Convert zeroth-order SH DC coefficients to linear RGB.

    Args:
        dc_coefficients: Raw DC coefficients from the PLY file.

    Returns:
        Linear RGB values clamped to ``[0, inf)``.
    """
    return np.maximum(dc_coefficients * SH_C0 + 0.5, 0.0)


def _normalized_color_channels(
    vertex: np.ndarray, names: set[str]
) -> Float32[np.ndarray, "n 3"] | None:
    """Extract and normalize ``red/green/blue`` (or ``r/g/b``) vertex fields.

    Integer channels are mapped to ``[0, 1]`` via the dtype maximum; float
    channels are clipped to ``[0, 1]``.

    Args:
        vertex: Structured NumPy array from the PLY vertex element.
        names: Set of field names present in the vertex data.

    Returns:
        An ``(N, 3)`` float32 array, or ``None`` if no color fields are found.
    """
    color_names: tuple[str, str, str] | None = (
        ("red", "green", "blue")
        if {"red", "green", "blue"} <= names
        else ("r", "g", "b")
        if {"r", "g", "b"} <= names
        else None
    )
    if color_names is None:
        return None

    channels: list[Float32[np.ndarray, "n"]] = []
    for name in color_names:
        values: np.ndarray = np.asarray(vertex[name])
        if np.issubdtype(values.dtype, np.integer):
            dtype_info: np.iinfo = np.iinfo(values.dtype)
            channel: Float32[np.ndarray, "n"] = values.astype(np.float32) / np.float32(dtype_info.max)
        else:
            channel = values.astype(np.float32)
        channels.append(np.clip(channel, 0.0, 1.0))

    return np.stack(channels, axis=1).astype(np.float32)


@dataclass(frozen=True)
class Gaussians3D(rr.AsComponents):
    """Minimal Python logging wrapper for the Rust Gaussian splat visualizer.

    Implements ``rr.AsComponents`` so instances can be passed directly to
    ``rr.log()``.  The companion Rust viewer queries these components and
    renders them as GPU-accelerated Gaussian splats.
    """

    centers: Float32[np.ndarray, "n 3"]
    """World-space positions of each Gaussian center."""
    quaternions_xyzw: Float32[np.ndarray, "n 4"]
    """Rotation quaternions in ``[x, y, z, w]`` order (normalized)."""
    scales: Float32[np.ndarray, "n 3"]
    """Per-axis scale factors (already exponentiated, clamped >= 1e-6)."""
    opacities: Float32[np.ndarray, "n"]
    """Per-splat opacity in ``[0, 1]``."""
    colors_dc: Float32[np.ndarray, "n 3"]
    """Base RGB color derived from the zeroth SH coefficient."""
    sh_coefficients: Float32[np.ndarray, "n coeffs 3"] | None = None
    """Optional higher-order spherical harmonic coefficients."""

    def __post_init__(self) -> None:
        """Validate shapes, normalize quaternions, and clamp values."""
        centers: Float32[np.ndarray, "n 3"] = _as_float32("centers", self.centers, (3,))
        quaternions: Float32[np.ndarray, "n 4"] = _normalize_quaternions_xyzw(
            _as_float32("quaternions_xyzw", self.quaternions_xyzw, (4,))
        )
        scales: Float32[np.ndarray, "n 3"] = np.maximum(_as_float32("scales", self.scales, (3,)), 1e-6)
        opacities: Float32[np.ndarray, "n"] = np.clip(_as_float32_1d("opacities", self.opacities), 0.0, 1.0)
        colors_dc: Float32[np.ndarray, "n 3"] = np.clip(_as_float32("colors_dc", self.colors_dc, (3,)), 0.0, None)

        num_splats: int = centers.shape[0]
        for name, array in {
            "quaternions_xyzw": quaternions,
            "scales": scales,
            "opacities": opacities,
            "colors_dc": colors_dc,
        }.items():
            if array.shape[0] != num_splats:
                raise ValueError(f"{name} must have the same leading dimension as centers")

        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "quaternions_xyzw", quaternions)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "opacities", opacities)
        object.__setattr__(self, "colors_dc", colors_dc)

        if self.sh_coefficients is not None:
            sh: Float32[np.ndarray, "n coeffs 3"] = np.asarray(self.sh_coefficients, dtype=np.float32)
            if sh.ndim != 3 or sh.shape[0] != num_splats or sh.shape[2] != 3:
                raise ValueError("sh_coefficients must have shape [N, coeffs_per_channel, 3]")
            object.__setattr__(self, "sh_coefficients", np.ascontiguousarray(sh))

    def as_component_batches(self) -> list[rr.DescribedComponentBatch]:
        """Convert Gaussian data into Rerun component batches.

        Returns:
            A list of described component batches ready for ``rr.log()``.
        """
        color_bytes: UInt8[np.ndarray, "n 3"] = np.round(np.clip(self.colors_dc, 0.0, 1.0) * 255.0).astype(np.uint8)
        batches: list[rr.DescribedComponentBatch] = [
            rr.components.Translation3DBatch(self.centers).described(
                _component_descriptor("GaussianSplats3D:centers", "rerun.components.Translation3D")
            ),
            rr.components.RotationQuatBatch(self.quaternions_xyzw).described(
                _component_descriptor(
                    "GaussianSplats3D:quaternions", "rerun.components.RotationQuat"
                )
            ),
            rr.components.Scale3DBatch(self.scales).described(
                _component_descriptor("GaussianSplats3D:scales", "rerun.components.Scale3D")
            ),
            rr.components.OpacityBatch(self.opacities).described(
                _component_descriptor("GaussianSplats3D:opacities", "rerun.components.Opacity")
            ),
            rr.components.ColorBatch(color_bytes).described(
                _component_descriptor("GaussianSplats3D:colors", "rerun.components.Color")
            ),
        ]

        if self.sh_coefficients is not None:
            batches.append(
                rr.components.TensorDataBatch(
                    [
                        rr.datatypes.TensorData(
                            array=self.sh_coefficients,
                            dim_names=["splat", "coefficient", "channel"],
                        )
                    ]
                ).described(
                    _component_descriptor(
                        "GaussianSplats3D:sh_coefficients",
                        "rerun.components.TensorData",
                    )
                )
            )

        return batches

    @classmethod
    def from_ply(cls, path: Path) -> Gaussians3D:
        """Load Gaussian splat data from a PLY file.

        Supports standard 3DGS PLY files with fields ``x/y/z``,
        ``scale_0/1/2``, ``rot_0/1/2/3``, ``opacity``, and optional
        ``f_dc_*`` / ``f_rest_*`` spherical harmonic coefficients.

        Args:
            path: Path to the ``.ply`` file.

        Returns:
            A :class:`Gaussians3D` instance ready for ``rr.log()``.
        """
        ply: PlyData = PlyData.read(path)
        vertex: np.ndarray = ply["vertex"].data
        names: set[str] = set(vertex.dtype.names or ())

        centers: Float32[np.ndarray, "n 3"] = np.stack(
            [vertex["x"], vertex["y"], vertex["z"]], axis=1
        ).astype(np.float32)
        scales: Float32[np.ndarray, "n 3"] = np.exp(
            np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1).astype(np.float32)
        )
        quaternions_xyzw: Float32[np.ndarray, "n 4"] = np.stack(
            [vertex["rot_1"], vertex["rot_2"], vertex["rot_3"], vertex["rot_0"]], axis=1
        ).astype(np.float32)
        opacities: Float32[np.ndarray, "n"] = _sigmoid(
            np.asarray(vertex["opacity"], dtype=np.float32)
        ).astype(np.float32)

        dc_coefficients: Float32[np.ndarray, "n 3"] | None = None
        if {"f_dc_0", "f_dc_1", "f_dc_2"} <= names:
            dc_coefficients = np.stack(
                [vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1
            ).astype(np.float32)
            colors_dc: Float32[np.ndarray, "n 3"] = _sh_dc_to_rgb(dc_coefficients)
        elif (colors := _normalized_color_channels(vertex, names)) is not None:
            colors_dc = colors
        else:
            colors_dc = np.ones((len(vertex), 3), dtype=np.float32)

        rest_fields: dict[int, Float32[np.ndarray, "n"]] = {
            int(name[len("f_rest_") :]): np.asarray(vertex[name], dtype=np.float32)
            for name in names
            if name.startswith("f_rest_") and name[len("f_rest_") :].isdigit()
        }

        sh_coefficients: Float32[np.ndarray, "n coeffs 3"] | None = None
        if dc_coefficients is not None or rest_fields:
            extra_coefficients: int = len(rest_fields) // 3
            raw_coeffs_per_channel: int = extra_coefficients + 1
            # The Rust viewer only supports SH sizes in {1, 4, 9, 16, 25}
            # (degrees 0-4).  Round up to the next supported size and zero-pad,
            # matching Brush's behavior for partial PLY payloads.
            supported_sh_sizes: tuple[int, ...] = (1, 4, 9, 16, 25)
            coeffs_per_channel: int = next(
                (s for s in supported_sh_sizes if s >= raw_coeffs_per_channel),
                raw_coeffs_per_channel,  # fall through if larger than degree 4
            )
            sh_coefficients = np.zeros((len(vertex), coeffs_per_channel, 3), dtype=np.float32)
            if dc_coefficients is not None:
                sh_coefficients[:, 0, :] = dc_coefficients

            # ``f_rest_*`` is channel-major: all red coefficients, then green, then blue.
            # Missing coefficients are treated as zero so partial payloads degrade gracefully.
            zeros: Float32[np.ndarray, "n"] = np.zeros(len(vertex), dtype=np.float32)
            for coefficient_index in range(extra_coefficients):
                sh_coefficients[:, coefficient_index + 1, 0] = rest_fields.get(
                    coefficient_index, zeros
                )
                sh_coefficients[:, coefficient_index + 1, 1] = rest_fields.get(
                    extra_coefficients + coefficient_index, zeros
                )
                sh_coefficients[:, coefficient_index + 1, 2] = rest_fields.get(
                    extra_coefficients * 2 + coefficient_index, zeros
                )

        return cls(
            centers=centers,
            quaternions_xyzw=quaternions_xyzw,
            scales=scales,
            opacities=opacities,
            colors_dc=colors_dc,
            sh_coefficients=sh_coefficients,
        )
