"""Reproduce slow REPLACE with only three small synthetic RRDs.

Run on the test server host, using either the 0.38.1 or 0.41.0 Python environment:
    REPRO_URL=rerun+http://127.0.0.1:51241 python repro_replace_scaling.py

This creates scratch datasets, preserves its generated files, and never accesses
port 51235. More layers amplify the schema-cache cost without needing large data.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from time import perf_counter
from urllib.parse import urlparse

import rerun as rr
from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer
from rerun.chunk import Chunk, LazyChunkStream


def main() -> None:
    """Compare insertion, replacement, and unregister/reinsert on the same schemas."""
    url: str = os.environ.get("REPRO_URL", "rerun+http://127.0.0.1:51241")
    if urlparse(url).port == 51235:
        raise ValueError("The shared catalog port 51235 is forbidden")
    root: Path = Path(tempfile.mkdtemp(prefix="rerun-replace-repro-"))
    width: int = 64
    paths: list[str] = []
    for segment in range(3):
        path: Path = root / f"segment-{segment}.rrd"
        chunk: Chunk = Chunk.from_property("wide", rr.AnyValues(**{f"field_{field}": float(field) for field in range(width)}))
        LazyChunkStream.from_iter([chunk]).write_rrd(path, application_id="schema-scale", recording_id=f"segment-{segment}")
        paths.append(path.as_uri())
    client: CatalogClient = CatalogClient(url)
    print(f"SDK {rr.__version__}; server {url}; files {root}", flush=True)
    for layers in (1, 8, 32, 64):
        dataset: DatasetEntry = client.create_dataset(f"schema-scale-{rr.__version__}-{root.name}-{layers}")
        start: float = perf_counter()
        for layer in range(layers):
            dataset.register(paths, layer_name=f"layer-{layer}").wait()
        inserted: float = perf_counter() - start
        start = perf_counter()
        for layer in range(layers):
            dataset.register(paths, layer_name=f"layer-{layer}", on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
        replaced: float = perf_counter() - start
        start = perf_counter()
        dataset.unregister(
            segments_to_drop=[f"segment-{segment}" for segment in range(3)],
            layers_to_drop=[f"layer-{layer}" for layer in range(layers)],
        ).wait()
        for layer in range(layers):
            dataset.register(paths, layer_name=f"layer-{layer}").wait()
        rebuilt: float = perf_counter() - start
        print(f"layers={layers} sources={3 * layers} insert_s={inserted:.6f} replace_s={replaced:.6f} rebuild_s={rebuilt:.6f}", flush=True)


if __name__ == "__main__":
    main()
