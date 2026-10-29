"""dataforge-swap-layer: replace one layer behind the catalog idle gate."""

from dataclasses import dataclass
from pathlib import Path

from rerun.catalog import CatalogClient

from dataforge.catalog_swap import LayerEntry, catalog_idle, swap_layer


@dataclass
class SwapConfig:
    """Swap one layer behind the catalog idle gate, without publishing blueprints."""

    layer: str
    """Layer to unregister and register with SKIP."""
    files: tuple[Path, ...]
    """New .rrd files named for their segment IDs."""
    catalog_name: str
    """Existing catalog dataset to update."""
    catalog_url: str = "rerun+http://127.0.0.1:51235"
    """Catalog gRPC URL."""
    batch_size: int = 500
    """Maximum segments per gated batch."""
    gate_timeout_s: float = 600.0
    """Maximum seconds each batch waits for an idle catalog."""
    local_catalog: bool = True
    """Probe the local listening backlog; disable for a remote catalog."""
    nfs_mount: Path | None = None
    """Optional NFS mount whose READ latency is checked over 20 seconds."""

    def __post_init__(self) -> None:
        if not self.files:
            raise ValueError("swap-layer requires files")



def main(config: SwapConfig) -> None:
    """Swap only the requested segment layer; retain all catalog_swap guards."""
    entry: LayerEntry = CatalogClient(config.catalog_url).get_dataset(config.catalog_name)
    swap_layer(
        entry, config.layer, config.files, batch_size=config.batch_size, gate_timeout_s=config.gate_timeout_s,
        idle_probe=lambda: catalog_idle(config.catalog_url, config.nfs_mount, local=config.local_catalog),
    )
