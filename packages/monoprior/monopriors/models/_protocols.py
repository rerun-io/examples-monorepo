from typing import Literal, Protocol, Self, runtime_checkable


@runtime_checkable
class DeviceMovable(Protocol):
    """Model that can move between supported inference devices."""

    def to(self, device: Literal["cpu", "cuda"], /) -> Self:
        """Move the model to an inference device.

        Args:
            device: Target inference device.

        Returns:
            The moved model.
        """
        ...
