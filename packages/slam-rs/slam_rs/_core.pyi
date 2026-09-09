"""Types for the compiled ``slam_rs._core`` extension (built by the ``slam-rs-build`` task).

The signatures here are the only static check on the FFI boundary, so they stay
exact: no ``Any``, and every array carries its dtype.
"""

__version__: str

class VioConfig:
    """basalt's ``VioConfig``, as ``data/**/*_config.json`` carries it."""

    def __init__(self) -> None:
        """basalt's own defaults, the ones its constructor sets."""

    @staticmethod
    def from_json(text: str) -> VioConfig:
        """Read one of basalt's config files; keys it omits keep their default."""

    def to_json(self) -> str:
        """Write the config back in basalt's shape, ``value0`` wrapper and all."""

    optical_flow_image_safe_radius: float
    """Circular mask that hides a fisheye's black corners, in pixels; 0 disables it."""

    def __repr__(self) -> str: ...

class Calibration:
    """basalt's camera-IMU calibration: extrinsics, intrinsics and the noise model."""

    @staticmethod
    def from_json(text: str) -> Calibration:
        """Read one of basalt's calibration files."""

    def to_json(self) -> str:
        """Write the calibration back in basalt's shape, ``value0`` wrapper and all."""

    @property
    def camera_count(self) -> int: ...
    @property
    def resolution(self) -> list[tuple[int, int]]:
        """Each camera's ``(width, height)`` in pixels, in rig order."""

    def __repr__(self) -> str: ...