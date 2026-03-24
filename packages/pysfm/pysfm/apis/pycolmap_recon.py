from dataclasses import dataclass
from pathlib import Path

from simplecv.rerun_log_utils import RerunTyroConfig


# ---------------------------------------------------------------------------
# CLI config
# ---------------------------------------------------------------------------
@dataclass
class SfMCLIConfig:
    """CLI configuration for COLMAP SfM reconstruction."""

    rr_config: RerunTyroConfig
    """Rerun logging configuration (--save, --connect, --spawn)."""
    videos_dir: Path = Path("data/examples/unknown-rig")
    """Directory containing Videos."""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main(config: SfMCLIConfig) -> None:
    print(f"hi: {config}")
