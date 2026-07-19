"""Gaussians3D wire schema for the in-development rerun ``Gaussians3D`` archetype.

Emits the exact upstream component layout as *custom* components on top of the
released ``rerun-sdk`` (0.34.1), which does not yet ship the archetype natively.
Each field is tagged with a :class:`rerun.ComponentDescriptor` whose
``archetype``/``component``/``component_type`` strings match the upstream
contract verbatim, and every batch is a hand-built ``pyarrow`` array so the
Arrow datatypes line up byte-for-byte with what the Rust deserializer checks
(FixedSizeList child fields named ``"item"`` and non-nullable, ``f16`` SH
coefficients, ``UInt32`` packed colors).

Semantics (upstream): opacity lives in the color alpha channel; the SH
degree-0 (DC) term is folded into the color RGB; ``sh_coefficients`` therefore
carries ONLY degrees 1-3 as 45 ``f16`` values in coefficient-major layout
``[c1.rgb, c2.rgb, ..., c15.rgb]`` (i.e. ``coefficients[3 * coeff + channel]``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import rerun as rr
from jaxtyping import Float16, Float32, UInt8, UInt32
from plyfile import PlyData
from rerun._baseclasses import DescribedComponentBatch

ARCHETYPE: str = "rerun.archetypes.Gaussians3D"

SPLATS_ENTITY: str = "/world/splats"
"""Canonical entity path both CLIs log splats under (and blueprints bind)."""

SPLATS_VISUALIZER: str = "Gaussians3D"
"""Identifier of the custom Rust visualizer — must match `IdentifiedViewSystem` in gaussian_visualizer.rs."""
"""Fully-qualified upstream archetype name for every emitted component."""

SH_C0: float = 0.28209479177387814
"""Zeroth spherical-harmonic coefficient, ``1 / (2 * sqrt(pi))``."""

SH_REST_COEFFS: int = 15
"""Number of degree 1-3 SH coefficients per channel ((3+1)^2 - 1)."""

SH_REST_VALUES: int = SH_REST_COEFFS * 3
"""Flattened length of the degree 1-3 SH block (45 = 15 coeffs * 3 channels)."""


class _RawArrowBatch:
    """Minimal ``ComponentBatchLike`` wrapping a pre-built ``pyarrow`` array.

    Rerun's ``DescribedComponentBatch`` accepts any object exposing
    ``as_arrow_array``; this lets us attach an upstream ``ComponentDescriptor``
    to arrays whose Arrow datatype the stock Python SDK cannot express (``f16``
    fixed-size lists, ``UInt32`` colors).
    """

    def __init__(self, array: pa.Array) -> None:
        self._array: pa.Array = array

    def as_arrow_array(self) -> pa.Array:
        """Return the wrapped ``pyarrow`` array."""
        return self._array


def _descriptor(component: str, component_type: str) -> rr.ComponentDescriptor:
    """Build the ``Gaussians3D`` component descriptor for one field.

    Args:
        component: Field-qualified component name, e.g. ``"Gaussians3D:centers"``.
        component_type: Upstream component-type string, e.g. ``"rerun.components.Position3D"``.

    Returns:
        A :class:`rerun.ComponentDescriptor` tagged with :data:`ARCHETYPE`.
    """
    return rr.ComponentDescriptor(archetype=ARCHETYPE, component=component, component_type=component_type)


def _fixed_size_list(values: npt.ArrayLike, dtype: pa.DataType, size: int) -> pa.FixedSizeListArray:
    """Build a ``FixedSizeList<dtype, size>`` whose child field is ``"item"`` and non-nullable.

    Args:
        values: Flattenable array-like of the child values (length must be a multiple of ``size``).
        dtype: Arrow child datatype (e.g. ``pa.float32()`` / ``pa.float16()``).
        size: Fixed list length.

    Returns:
        A :class:`pyarrow.FixedSizeListArray` with the exact datatype the Rust deserializer expects.
    """
    flat: np.ndarray = np.ascontiguousarray(values).reshape(-1)
    list_type: pa.DataType = pa.list_(pa.field("item", dtype, nullable=False), size)
    return pa.FixedSizeListArray.from_arrays(pa.array(flat, type=dtype), type=list_type)


def _as_float32(name: str, values: npt.ArrayLike, width: int) -> Float32[np.ndarray, "n width"]:
    """Validate and cast *values* to a contiguous ``[N, width]`` float32 array."""
    array: Float32[np.ndarray, "n width"] = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape [N, {width}]")
    return np.ascontiguousarray(array)


def _normalize_quaternions_xyzw(quaternions_xyzw: Float32[np.ndarray, "n 4"]) -> Float32[np.ndarray, "n 4"]:
    """Normalize quaternions to unit length, substituting identity for near-zero inputs.

    Args:
        quaternions_xyzw: Quaternions in ``[x, y, z, w]`` order.

    Returns:
        Unit-length quaternions with the same shape.
    """
    norms: Float32[np.ndarray, "n 1"] = np.linalg.norm(quaternions_xyzw, axis=1, keepdims=True)
    identity: Float32[np.ndarray, "1 4"] = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    return np.where(norms > 1e-12, quaternions_xyzw / np.maximum(norms, 1e-12), identity).astype(np.float32)


def _sigmoid(x: Float32[np.ndarray, "n"]) -> Float32[np.ndarray, "n"]:
    """Element-wise sigmoid activation."""
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


@dataclass(frozen=True)
class Gaussians3D(rr.AsComponents):
    """Upstream ``Gaussians3D`` splat payload, logged as custom rerun components.

    Implements ``rr.AsComponents`` so instances pass straight to ``rr.log``.
    ``centers`` is required; ``scales``/``quaternions_xyzw``/``colors_rgba`` are
    recommended; ``sh_coefficients``/``show_spherical_harmonics`` are optional
    and omitted from the wire (never logged empty) when ``None``.
    """

    centers: Float32[np.ndarray, "n 3"]
    """World-space Gaussian centers (Position3D)."""
    scales: Float32[np.ndarray, "n 3"] | None = None
    """Per-axis scale factors, already exponentiated (Scale3D)."""
    quaternions_xyzw: Float32[np.ndarray, "n 4"] | None = None
    """Unit rotation quaternions in ``[x, y, z, w]`` order (RotationQuat)."""
    colors_rgba: UInt8[np.ndarray, "n 4"] | None = None
    """Per-splat RGBA color; alpha carries opacity, RGB carries the SH DC term (Color)."""
    sh_coefficients: Float32[np.ndarray, "n 45"] | None = None
    """Degree 1-3 SH block, coefficient-major ``[c1.rgb, ..., c15.rgb]`` (SphericalHarmonics3)."""
    show_spherical_harmonics: bool | None = None
    """Whether the viewer should evaluate the higher-order SH (ShowSphericalHarmonics)."""

    def __post_init__(self) -> None:
        """Validate shapes/lengths, normalize quaternions, and store contiguous arrays."""
        centers: Float32[np.ndarray, "n 3"] = _as_float32("centers", self.centers, 3)
        num_splats: int = centers.shape[0]
        object.__setattr__(self, "centers", centers)

        if self.scales is not None:
            scales: Float32[np.ndarray, "n 3"] = _as_float32("scales", self.scales, 3)
            if scales.shape[0] != num_splats:
                raise ValueError("scales must share the leading dimension of centers")
            object.__setattr__(self, "scales", scales)

        if self.quaternions_xyzw is not None:
            quaternions: Float32[np.ndarray, "n 4"] = _normalize_quaternions_xyzw(_as_float32("quaternions_xyzw", self.quaternions_xyzw, 4))
            if quaternions.shape[0] != num_splats:
                raise ValueError("quaternions_xyzw must share the leading dimension of centers")
            object.__setattr__(self, "quaternions_xyzw", quaternions)

        if self.colors_rgba is not None:
            colors: UInt8[np.ndarray, "n 4"] = np.ascontiguousarray(np.asarray(self.colors_rgba, dtype=np.uint8))
            if colors.ndim != 2 or colors.shape[1] != 4 or colors.shape[0] != num_splats:
                raise ValueError("colors_rgba must have shape [N, 4] matching centers")
            object.__setattr__(self, "colors_rgba", colors)

        if self.sh_coefficients is not None:
            sh: Float32[np.ndarray, "n 45"] = _as_float32("sh_coefficients", self.sh_coefficients, SH_REST_VALUES)
            if sh.shape[0] != num_splats:
                raise ValueError("sh_coefficients must share the leading dimension of centers")
            object.__setattr__(self, "sh_coefficients", sh)

    def as_component_batches(self) -> list[DescribedComponentBatch]:
        """Emit the upstream ``Gaussians3D`` components as described Arrow batches.

        Returns:
            ``centers`` first, then any of ``scales``/``quaternions``/``colors``/
            ``sh_coefficients``/``show_spherical_harmonics`` that are present.
        """
        batches: list[DescribedComponentBatch] = [
            DescribedComponentBatch(
                _RawArrowBatch(_fixed_size_list(self.centers, pa.float32(), 3)),
                _descriptor("Gaussians3D:centers", "rerun.components.Position3D"),
            )
        ]

        if self.scales is not None:
            batches.append(
                DescribedComponentBatch(
                    _RawArrowBatch(_fixed_size_list(self.scales, pa.float32(), 3)),
                    _descriptor("Gaussians3D:scales", "rerun.components.Scale3D"),
                )
            )

        if self.quaternions_xyzw is not None:
            batches.append(
                DescribedComponentBatch(
                    _RawArrowBatch(_fixed_size_list(self.quaternions_xyzw, pa.float32(), 4)),
                    _descriptor("Gaussians3D:quaternions", "rerun.components.RotationQuat"),
                )
            )

        if self.colors_rgba is not None:
            rgba: UInt32[np.ndarray, "n 4"] = self.colors_rgba.astype(np.uint32)
            packed: UInt32[np.ndarray, "n"] = (rgba[:, 0] << 24) | (rgba[:, 1] << 16) | (rgba[:, 2] << 8) | rgba[:, 3]
            batches.append(
                DescribedComponentBatch(
                    _RawArrowBatch(pa.array(np.ascontiguousarray(packed), type=pa.uint32())),
                    _descriptor("Gaussians3D:colors", "rerun.components.Color"),
                )
            )

        if self.sh_coefficients is not None:
            sh_f16: Float16[np.ndarray, "n 45"] = self.sh_coefficients.astype(np.float16)
            batches.append(
                DescribedComponentBatch(
                    _RawArrowBatch(_fixed_size_list(sh_f16, pa.float16(), SH_REST_VALUES)),
                    _descriptor("Gaussians3D:sh_coefficients", "rerun.components.SphericalHarmonics3"),
                )
            )

        if self.show_spherical_harmonics is not None:
            batches.append(
                DescribedComponentBatch(
                    _RawArrowBatch(pa.array([bool(self.show_spherical_harmonics)], type=pa.bool_())),
                    _descriptor("Gaussians3D:show_spherical_harmonics", "rerun.components.ShowSphericalHarmonics"),
                )
            )

        return batches

    @classmethod
    def from_ply(cls, path: Path) -> Gaussians3D:
        """Load an INRIA 3DGS PLY into the upstream ``Gaussians3D`` schema.

        Converts the standard 3DGS fields (``x/y/z``, ``scale_0..2``,
        ``rot_0..3`` wxyz, ``opacity``, ``f_dc_0..2``, ``f_rest_*``) into
        centers, exponentiated scales, normalized xyzw quaternions, packed RGBA
        color (DC folded into RGB, opacity into alpha), and the degree 1-3 SH
        block in coefficient-major layout.

        Args:
            path: Path to the ``.ply`` file.

        Returns:
            A :class:`Gaussians3D` instance ready for ``rr.log``.
        """
        ply: PlyData = PlyData.read(path)
        vertex: np.ndarray = ply["vertex"].data
        names: set[str] = set(vertex.dtype.names or ())
        num_splats: int = len(vertex)

        centers: Float32[np.ndarray, "n 3"] = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
        scales: Float32[np.ndarray, "n 3"] = np.exp(
            np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1).astype(np.float32)
        )
        # PLY rot_0..3 is wxyz; reorder to xyzw (normalization happens in __post_init__).
        quaternions_xyzw: Float32[np.ndarray, "n 4"] = np.stack(
            [vertex["rot_1"], vertex["rot_2"], vertex["rot_3"], vertex["rot_0"]], axis=1
        ).astype(np.float32)

        # Color: DC SH term -> RGB, sigmoid(opacity) -> alpha, packed to u8 RGBA.
        if {"f_dc_0", "f_dc_1", "f_dc_2"} <= names:
            dc: Float32[np.ndarray, "n 3"] = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1).astype(np.float32)
            rgb: Float32[np.ndarray, "n 3"] = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)
        else:
            rgb = np.ones((num_splats, 3), dtype=np.float32)
        alpha: Float32[np.ndarray, "n"] = _sigmoid(np.asarray(vertex["opacity"], dtype=np.float32)) if "opacity" in names else np.ones(num_splats, dtype=np.float32)
        colors_rgba: UInt8[np.ndarray, "n 4"] = np.round(
            np.concatenate([rgb, alpha[:, None]], axis=1) * 255.0
        ).astype(np.uint8)

        # SH degrees 1-3: f_rest_* is channel-major (3, K) per splat -> coefficient-major (K, 3).
        # Strict acceptance (mirrors src/ply_loader.rs): the fields must be a
        # contiguous f_rest_0..N-1 run, N divisible by 3, and N/3 must be a
        # complete SH band count (3/8/15 rest coeffs = degree 1/2/3).  Reject
        # anything else loudly instead of silently compacting or truncating.
        rest_count: int = sum(1 for name in names if name.startswith("f_rest_"))
        sh_coefficients: Float32[np.ndarray, "n 45"] | None = None
        show_spherical_harmonics: bool | None = None
        if rest_count:
            missing: list[str] = [f"f_rest_{i}" for i in range(rest_count) if f"f_rest_{i}" not in names]
            if missing:
                raise ValueError(f"{path}: non-contiguous f_rest_* fields (count {rest_count}, missing {missing[:3]}...)")
            if rest_count % 3 != 0 or rest_count // 3 not in (3, 8, 15):
                raise ValueError(
                    f"{path}: unsupported f_rest_* count {rest_count}; expected 9/24/45 (complete SH degree 1/2/3 bands)"
                )
            rest: Float32[np.ndarray, "n rest"] = np.stack([np.asarray(vertex[f"f_rest_{i}"], dtype=np.float32) for i in range(rest_count)], axis=1)
            coeffs_per_channel: int = rest_count // 3
            # (N, 3, K) channel-major -> (N, K, 3) coefficient-major -> flat,
            # zero-padded to the fixed 45-value wire block (lower degrees pad
            # exactly by construction of the coefficient-major layout).
            coefficient_major: Float32[np.ndarray, "n flat"] = (
                rest.reshape(num_splats, 3, coeffs_per_channel).transpose(0, 2, 1).reshape(num_splats, -1)
            )
            sh_coefficients = np.zeros((num_splats, SH_REST_VALUES), dtype=np.float32)
            sh_coefficients[:, : coefficient_major.shape[1]] = coefficient_major
            show_spherical_harmonics = True

        return cls(
            centers=centers,
            scales=scales,
            quaternions_xyzw=quaternions_xyzw,
            colors_rgba=colors_rgba,
            sh_coefficients=sh_coefficients,
            show_spherical_harmonics=show_spherical_harmonics,
        )
