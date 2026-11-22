"""Log a SOG v2/v3 scene (bundled `.sog`, `meta.json`, or its directory) as `SogSplats3D`, or an animated-sog `.asog`
clip as `AnimatedSogSplats3D`, each with a `GaussianSplats3D` preview on the same entity."""

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
from simplecv.rerun_log_utils import RerunTyroConfig

from gsplat_rust_renderer.asog_decode import TEMPORAL_THRESHOLD, AsogFile, read_asog
from gsplat_rust_renderer.sog3d import PREVIEW_ROWS, log_asog, log_sog
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
    """A bundled .sog, a meta.json, a directory holding one, or an .asog clip; None takes --sample."""
    sample: Literal["skull"] = "skull"
    """Pinned public sample fetched into data/sog when --path is not given."""
    entity: str = "/world/splats"
    """Entity of the payload and its preview."""
    preview_rows: int = PREVIEW_ROWS
    """Stock-viewer preview rows (every k-th splat; .asog: of the mid-clip pose); 0 logs no preview (then the custom
    viewer has no GaussianSplats3D rows to attach the payload to)."""
    frame_origin: float = 0.0
    """.asog only: the `frame` timeline value at clip time 0 (the clip plays on `frame` at the manifest frameRate)."""
    active_threshold: float = TEMPORAL_THRESHOLD
    """.asog only: temporal cut the viewer applies (0.05 = the format's constant, what other players show)."""


def main(config: LogSogConfig) -> None:
    path: Path = config.path if config.path is not None else fetch_sample(config.sample)
    if path.suffix == ".asog":
        asog: AsogFile = read_asog(path)
        rate: float = asog.manifest.timeline.frame_rate
        # .asog positions are in the trainer frame (TRiGS / COLMAP, y down); players rotate for display only.
        rr.log("/", rr.ViewCoordinates.RDF, static=True)
        log_asog(config.entity, asog, config.frame_origin, config.active_threshold, config.preview_rows, fps=rate)
        # The payload is static: a clip-time scalar per frame gives the recording its `frame` timeline.
        frames: int = round(asog.manifest.duration * rate)
        rr.send_columns(
            "/clip/seconds",
            indexes=[rr.TimeColumn("frame", sequence=round(config.frame_origin) + np.arange(frames))],
            columns=rr.Scalars.columns(scalars=np.arange(frames) / rate),
        )
        print(
            f"logged {path} (asog v{asog.manifest.version}, {asog.manifest.count} splats, {len(asog.data)} B, {rate} fps, "
            f"cut {config.active_threshold}) as static {config.entity}; play it on the `frame` timeline"
        )
        return
    sog: SogFile = read_sog(path)
    # SOG keeps the 3DGS PLY frame (x right, y down, z forward).
    rr.log("/", rr.ViewCoordinates.RDF, static=True)
    log_sog(config.entity, sog, config.preview_rows)
    stored: int = sum(len(b) for b in sog.planes.values())
    print(f"logged {path} (SOG v{sog.meta.version}, {sog.meta.count} splats, {stored} plane bytes) as static {config.entity}")
