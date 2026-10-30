"""Single-segment prototype entry point."""

from dataclasses import dataclass
from pathlib import Path

from serde.json import to_json

from selfcap_soma.catalog import read_window
from selfcap_soma.fitting import fit
from selfcap_soma.output import save


@dataclass(frozen=True, slots=True)
class Config:
    catalog: str = "rerun+http://dgx-spark:51235"
    """Read-only catalog endpoint."""
    dataset: str = "selfcap-post-dev"
    """Source dataset."""
    segment: str = "selfcap__2025-12-03__d31bf1da__ep001"
    """Only segment to query."""
    start: float = 2.0
    """First video second."""
    end: float = 6.0
    """Last video second."""
    stride: int = 2
    """Keep every nth exo observation frame."""
    iterations: int = 400
    """Total Adam iterations."""
    shape_iterations: int = 60
    """Initial shared-shape fitting iterations."""
    assets: Path = Path("/tmp/soma-fit/assets")
    """Official SOMA-X asset directory."""
    output: Path = Path("/tmp/soma-fit/window")
    """Local RRD and report directory; never registered."""
    target: float = 4.144
    """Video time used for diagnostics."""


def main(config: Config) -> None:
    if not 0 < config.shape_iterations < config.iterations:
        raise ValueError("shape iterations must be between zero and total iterations")
    data = read_window(config.catalog, config.dataset, config.segment, config.start, config.end, config.stride)
    result = fit(data, config.assets, config.iterations, config.shape_iterations)
    print(to_json(save(config.output, data, result, config.target)))
