"""Log a SOG v2/v3 scene (bundled `.sog`, `meta.json`, or its directory) as `SogSplats3D` + a `GaussianSplats3D` preview."""

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import rerun as rr
from simplecv.rerun_log_utils import RerunTyroConfig

from gsplat_rust_renderer.sog3d import PREVIEW_ROWS, log_sog
from gsplat_rust_renderer.sog_decode import SogFile, read_sog

SAMPLES_DIR: Path = Path(__file__).resolve().parents[2] / "data/sog"


@dataclass(frozen=True, slots=True)
class Sample:
    url: str
    sha256: str


SAMPLES: dict[str, Sample] = {
    # PlayCanvas engine example: v2, bundled, SH (bands inferred), 246,821 splats, 5,431,881 B.
    "skull": Sample(
        "https://raw.githubusercontent.com/playcanvas/engine/main/examples/assets/splats/skull.sog",
        "f28305aec60d409b5d1ef294b1f8f285641485c3a68a343a14db4a43496bc175",
    ),
}


def fetch_sample(name: Literal["skull"], directory: Path = SAMPLES_DIR) -> Path:
    """Download a pinned public sample once (checked by sha256) and return its path."""
    sample: Sample = SAMPLES[name]
    path: Path = directory / Path(sample.url).name
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(sample.url, timeout=120) as response:
            data: bytes = response.read()
        digest: str = hashlib.sha256(data).hexdigest()
        if digest != sample.sha256:
            raise ValueError(f"{sample.url}: sha256 {digest}, expected {sample.sha256}")
        path.write_bytes(data)
    return path


@dataclass
class LogSogConfig:
    """Log a SOG scene to the custom Rust viewer (stock viewers draw the preview only)."""

    rr_config: RerunTyroConfig
    """Rerun connection/output configuration. Use --rr-config.connect to send to the Rust viewer."""
    path: Path | None = None
    """A bundled .sog, a meta.json, or a directory holding one; None takes --sample."""
    sample: Literal["skull"] = "skull"
    """Pinned public sample fetched into data/sog when --path is not given."""
    entity: str = "/world/splats"
    """Entity of the payload and its preview."""
    preview_rows: int = PREVIEW_ROWS
    """Stock-viewer preview rows (every k-th splat); 0 logs no preview."""


def main(config: LogSogConfig) -> None:
    path: Path = config.path if config.path is not None else fetch_sample(config.sample)
    sog: SogFile = read_sog(path)
    # SOG keeps the 3DGS PLY frame (x right, y down, z forward).
    rr.log("/", rr.ViewCoordinates.RDF, static=True)
    log_sog(config.entity, sog, config.preview_rows)
    stored: int = sum(len(b) for b in sog.planes.values())
    print(f"logged {path} (SOG v{sog.meta.version}, {sog.meta.count} splats, {stored} plane bytes) as static {config.entity}")
