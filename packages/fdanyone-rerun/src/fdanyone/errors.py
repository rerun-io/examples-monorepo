"""Project-specific errors with actionable user-facing messages."""


class FourDAnyoneError(RuntimeError):
    """Base class for expected pipeline failures."""


class ConfigurationError(FourDAnyoneError):
    """Raised when a frozen inference contract is violated."""


class AssetError(FourDAnyoneError):
    """Raised when a model or gated asset is missing or invalid."""


class VideoContractError(FourDAnyoneError):
    """Raised when an input cannot provide the canonical 121-frame clip."""
