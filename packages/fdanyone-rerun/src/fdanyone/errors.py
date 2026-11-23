"""Project-specific errors with actionable user-facing messages."""

import threading


class FourDAnyoneError(RuntimeError):
    """Base class for expected pipeline failures."""


class ConfigurationError(FourDAnyoneError):
    """Raised when a frozen inference contract is violated."""


class AssetError(FourDAnyoneError):
    """Raised when a model or gated asset is missing or invalid."""


class VideoContractError(FourDAnyoneError):
    """Raised when an input cannot provide the canonical 121-frame clip."""


class RunCancelled(FourDAnyoneError):
    """The run's cancel event was set; raised where the run notices it (a worker subprocess is reaped first)."""


def raise_if_stopped(cancel: threading.Event | None) -> None:
    """``RunCancelled`` once ``cancel`` is set (a Stop)."""
    if cancel is not None and cancel.is_set():
        raise RunCancelled("stopped")
