"""Report the version of the compiled Rust core, which also proves it imports."""

from dataclasses import dataclass

from slam_rs import _core


@dataclass(slots=True)
class Config:
    """Options for the slam-rs version CLI."""

    verbose: bool = False
    """Also print where the compiled extension was loaded from."""


def main(config: Config) -> None:
    """Print the core version, and its file path when asked.

    Args:
        config: Parsed CLI options.
    """
    print(f"slam-rs core {_core.__version__}")
    if config.verbose:
        print(f"slam_rs._core loaded from {_core.__file__}")
