"""Download FlexGlue's three ONNX graphs (cvg/FlexGlue release v0.1) into `MODEL_DIR/flexglue`.

COLMAP's FlexGlue options default to these release URLs and would fetch them at run time; colsfm passes local paths
instead, so a run never touches the network. The URLs and sha256 sums are the ones COLMAP pins in
`src/colmap/feature/resources.h` at the recipe's revision. A file already present with the right sum is kept.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from colsfm.matching import FLEXGLUE_MODEL_DIR

RELEASE_URL: Final[str] = "https://github.com/cvg/FlexGlue/releases/download/v0.1"
"""Where COLMAP's default FlexGlue URIs point."""

MODELS: Final[dict[str, str]] = {
    "flexglue_descriptor.onnx": "93fab8a4ccde1b7331640915357d96cff0929ea87134872f37f7b0150d9914ff",
    "flexglue_matcher_early_stop.onnx": "c86714cd8ffac70c17ee48c346270497e21846831336e12d307390591aa12ea7",
    "flexglue_matcher_early_stop_fp16.onnx": "01286e692e6396cdfe6eaf5944d3af60752f5043275b338a6419a97efa339e20",
}
"""File name to sha256, as pinned by COLMAP (dense descriptor, matcher, its float16 variant)."""

@dataclass
class Config:
    """Where to put the graphs."""

    output_dir: Path = field(default_factory=lambda: FLEXGLUE_MODEL_DIR)
    """Destination directory."""


def sha256(path: Path) -> str:
    """The sha256 of a file, read in 16 MiB blocks.

    Args:
        path: The file.

    Returns:
        The hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def main(config: Config) -> None:
    """Fetch every graph that is missing or has the wrong sum.

    Args:
        config: Destination.

    Raises:
        RuntimeError: When a downloaded file does not match its pinned sha256.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in MODELS.items():
        destination: Path = config.output_dir / name
        if destination.is_file() and sha256(destination) == expected:
            print(f"[colsfm] {name}: present")
            continue
        partial: Path = destination.with_suffix(".onnx.part")
        request = urllib.request.Request(f"{RELEASE_URL}/{name}", headers={"User-Agent": "colsfm-download"})
        with urllib.request.urlopen(request, timeout=600) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1 << 24)
        actual: str = sha256(partial)
        if actual != expected:
            partial.unlink()
            raise RuntimeError(f"{name}: sha256 {actual}, expected {expected}")
        partial.rename(destination)
        print(f"[colsfm] {name}: {destination.stat().st_size / 1e6:.0f} MB")
